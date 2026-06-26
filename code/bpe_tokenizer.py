import heapq
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

import regex as re
from base_tokenizer import BaseTokenizer

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
VOCAB_SIZE = 5000   # target vocab size: specials + 256 bytes + merges
MAX_TOKEN_WORDS = 2  # a token may span at most this many words (bigram)
MIN_PAIR_FREQ = 2  # ignore merges of pairs rarer than this (generalize)
BIGRAM_RESERVE_FRAC = 0.08  # fraction of vocab reserved for stage-B bigrams
TEXT_ENCODING = "utf-8"  # byte encoding used for all text <-> bytes

# cl100k-style pre-tokenization pattern (requires `regex` for \p{} Unicode
# properties). Improvements over GPT-2 pattern: case-insensitive contractions,
# \p{L}/\p{N} capture all Unicode letters/digits (not just ASCII-ish ranges),
# digits grouped ≤3 to avoid excessively long number tokens.
PRETOKENIZE_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Social-media variant: @handles and #hashtags matched atomically before the
# generic letter rule, keeping them as single BPE chunks.
PRETOKENIZE_PATTERN_SOCIAL = r"""@\p{L}\w*|#\p{L}\w*|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# @handle matcher — used by the domain-3 (mixed) flow to strip Twitter mentions
# from the combined corpus before training (they don't transfer to the hidden domain).
HANDLE_RE = re.compile(r"@\w+")


@lru_cache()
def bytes_to_unicode():
    """
    Returns a mapping of utf-8 byte -> printable unicode char.
    The reversible bpe codes work on unicode strings, so this avoids mapping
    bytes to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return the set of adjacent symbol pairs in a word (tuple of symbols)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.vocab_size = vocab_size

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.pat = re.compile(PRETOKENIZE_PATTERN, re.UNICODE)

        # Stage-A within-word merges: pair -> rank (lower = merged earlier).
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        # Stage-B bigram merges: (word_token_a, word_token_b) -> merged token.
        self.bigram_merges: Dict[Tuple[str, str], str] = {}
        # Reporting: merged bigram token -> corpus frequency.
        self.bigram_stats: Dict[str, int] = {}

        self.cache: Dict[str, List[str]] = {}  # chunk string -> bpe symbols
        self._encode_cache: Dict[str, List[int]] = {}  # full text -> token ids

        # Byte-level space marker; also a base-vocab token, required by NER.
        self.space_token = self.byte_encoder[ord(" ")]  # 'Ġ'

        # Training hyperparams — overridden by _auto_configure() at train() time.
        self._min_pair_freq: int = MIN_PAIR_FREQ
        self._bigram_reserve_frac: float = BIGRAM_RESERVE_FRAC
        self._lbpe_exp: float = 0.0
        self._min_bigram_freq: int = 8  # tuned: domain 1 & 2 both optimal at 8

    # ------------------------------------------------------------------ #
    # vocab helpers
    # ------------------------------------------------------------------ #
    def _add_token(self, token: str) -> int:
        """Register a token string, returning its id (existing or new)."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def _chunk_key(self, chunk: str) -> Tuple[str, ...]:
        """Byte-encode a pre-tokenized chunk into a tuple of base symbols."""
        return tuple(self.byte_encoder[b] for b in chunk.encode(TEXT_ENCODING))

    @staticmethod
    def _merge_symbols(symbols: List[str], pair) -> List[str]:
        """Replace every adjacent occurrence of pair with the joined symbol."""
        first, second = pair
        merged = first + second
        out = []
        i = 0
        n = len(symbols)
        while i < n:
            if symbols[i] == first and i < n - 1 and symbols[i + 1] == second:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return out

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _auto_configure(self, texts: List[str]) -> str:
        """Apply per-domain hyperparams and return the detected mode.

        Three cases (params from Optuna composite search; 'mixed' pending tuning):
          - 'mixed'  : combined domain_1+domain_2 corpus for the hidden domain-3
                       tokenizer. Only the union exceeds 1.4M lines, so size is the
                       discriminator (at_frac alone can't separate it from domain 1).
                       Generalist params + standard pattern; train() also strips
                       @handles — Twitter noise that won't transfer to the hidden domain.
          - 'social' : domain 1 (Twitter-like), ~0.44 of lines start with @handle.
          - 'formal' : domain 2 (news/long-form), ~0 @handles.
        """
        n        = len(texts) or 1
        at_frac  = sum(1 for t in texts if t.lstrip().startswith("@")) / n

        if n > 1_400_000:
            # Domain 3 (hidden): combined corpus. Generalist params (placeholder —
            # retune via cross-domain proxy on dev_1 + dev_2).
            self._min_pair_freq       = 5
            self._bigram_reserve_frac = 0.02
            self._lbpe_exp            = 1.0
            return "mixed"
        if at_frac > 0.05:
            # Domain 1 (social/Twitter)
            self._min_pair_freq       = 9
            self._bigram_reserve_frac = 0.0068
            self._lbpe_exp            = 1.4908
            self.pat = re.compile(PRETOKENIZE_PATTERN_SOCIAL, re.UNICODE)
            return "social"
        # Domain 2 (formal/news)
        self._min_pair_freq       = 4
        self._bigram_reserve_frac = 0.0909
        self._lbpe_exp            = 0.7771
        return "formal"

    def train(self, texts: List[str]) -> None:
        """Learn merges from texts (only the provided data is used)."""
        mode = self._auto_configure(texts)

        # Domain-3 flow: strip @handles from the combined corpus before training.
        # ~99.6% of @-tokens are non-entities in the NER data; the unknown eval
        # domain is unlikely to be Twitter, so handle merges waste vocab budget.
        if mode == "mixed":
            texts = [HANDLE_RE.sub("", t) for t in texts]

        # Base vocabulary: every byte-level unicode char (=> no OOV ever).
        for char in self.byte_encoder.values():
            self._add_token(char)

        line_counts = Counter(t for t in texts if t.strip())

        reserve = max(1, int(self.vocab_size * self._bigram_reserve_frac))
        within_budget = self.vocab_size - reserve

        line_keys = self._train_within_word(line_counts, within_budget)
        self._train_bigrams(line_counts, line_keys)

        # Drop the bulky atomic-word map; only needed during stage B.
        self._atomic = {}

    def _train_within_word(
        self, line_counts: Counter, budget: int
    ) -> Dict[str, List[Tuple[str, ...]]]:
        """Stage A: classic per-word BPE with an incremental pair-count index.

        Returns each line's pre-tokenized chunk keys so stage B can skip a second
        regex + byte-encode pass over the whole corpus.
        """
        # Deduplicate to unique pre-tokenized words (huge reduction vs. lines).
        word_freqs: Counter = Counter()
        line_keys: Dict[str, List[Tuple[str, ...]]] = {}
        for line, freq in line_counts.items():
            keys = [self._chunk_key(chunk) for chunk in self.pat.findall(line)]
            line_keys[line] = keys
            for k in keys:
                word_freqs[k] += freq

        # Parallel arrays: each word is a mutable symbol list with a frequency.
        word_syms: List[List[str]] = []
        word_freq: List[int] = []
        for key, freq in word_freqs.items():
            word_syms.append(list(key))
            word_freq.append(freq)

        # Global pair frequencies + inverted index pair -> word indices.
        pair_freqs: Dict[Tuple[str, str], int] = {}
        pair_to_words: Dict[Tuple[str, str], set] = {}
        for idx, syms in enumerate(word_syms):
            f = word_freq[idx]
            for p in zip(syms, syms[1:]):
                pair_freqs[p] = pair_freqs.get(p, 0) + f
                pair_to_words.setdefault(p, set()).add(idx)

        # LBPE: score = freq * merged_length^lbpe_exp (0 = pure freq, 1 = full LBPE).
        # Longer tokens carry more information per vocab slot; weighting by length
        # biases toward merges that reduce tokens/char most aggressively.
        lbpe_exp = getattr(self, "_lbpe_exp", 0.5)

        def _score(p: Tuple[str, str], f: int) -> float:
            return f * (len(p[0]) + len(p[1])) ** lbpe_exp

        heap = [(-_score(p, f), p) for p, f in pair_freqs.items()]
        heapq.heapify(heap)

        def push(p):
            f = pair_freqs.get(p, 0)
            if f > 0:
                heapq.heappush(heap, (-_score(p, f), p))

        rank = 0
        while len(self.token_to_id) < budget and heap:
            neg_sc, best = heapq.heappop(heap)
            # Skip stale entries: recompute score with current freq and compare.
            if _score(best, pair_freqs.get(best, 0)) != -neg_sc:
                continue
            if pair_freqs.get(best, 0) < self._min_pair_freq:
                break

            self.bpe_ranks[best] = rank
            rank += 1
            self._add_token(best[0] + best[1])

            # Re-merge only the words that actually contain `best`.
            for idx in list(pair_to_words.get(best, ())):
                syms = word_syms[idx]
                f = word_freq[idx]
                old = Counter(zip(syms, syms[1:]))
                new_syms = self._merge_symbols(syms, best)
                new = Counter(zip(new_syms, new_syms[1:]))
                word_syms[idx] = new_syms

                # Subtract old contribution, add new; fix the inverted index.
                for p, c in old.items():
                    pair_freqs[p] = pair_freqs.get(p, 0) - f * c
                    if pair_freqs[p] <= 0:
                        pair_freqs.pop(p, None)
                    if p not in new:
                        s = pair_to_words.get(p)
                        if s is not None:
                            s.discard(idx)
                    if p != best:
                        push(p)
                for p, c in new.items():
                    pair_freqs[p] = pair_freqs.get(p, 0) + f * c
                    pair_to_words.setdefault(p, set()).add(idx)
                    push(p)

            pair_freqs.pop(best, None)
            pair_to_words.pop(best, None)

        # Map each unique word to its final single token, if it became atomic.
        # word_syms[idx] already holds each word's fully merged form (index-
        # aligned with word_freqs), so no need to re-run _bpe per word.
        self._atomic: Dict[Tuple[str, ...], str] = {}
        for idx, key in enumerate(word_freqs):
            syms = word_syms[idx]
            if len(syms) == 1:
                self._atomic[key] = syms[0]

        return line_keys

    def _surface(self, token: str) -> str:
        """Decode a byte-level token back to its readable text."""
        return bytearray(self.byte_decoder[c] for c in token).decode(
            TEXT_ENCODING, errors="ignore"
        )

    def _train_bigrams(
        self, line_counts: Counter, line_keys: Dict[str, List[Tuple[str, ...]]]
    ) -> None:
        """Stage B: collapse the most frequent adjacent whole-word pairs.

        Only pairs of *atomic* words (each already a single token, and carrying
        an actual letter so punctuation/whitespace runs are skipped) are merged,
        so every bigram token spans exactly two words. Guarantees >= 1 bigram.
        Reuses stage A's cached chunk keys (line_keys) instead of re-tokenizing.
        """
        # Restrict to atomic tokens that contain a letter -> real word bigrams.
        word_atoms = {
            key: tok
            for key, tok in self._atomic.items()
            if any(c.isalpha() for c in self._surface(tok))
        }

        bigram_freqs: Counter = Counter()
        for line, freq in line_counts.items():
            atoms = [word_atoms.get(k) for k in line_keys[line]]
            for a, b in zip(atoms, atoms[1:]):
                if a is not None and b is not None:
                    bigram_freqs[(a, b)] += freq

        # Hard guarantee: if no adjacent atomic pair exists, synthesize one from
        # the two most frequent atomic tokens so the spec's bigram rule holds.
        if not bigram_freqs and word_atoms:
            tokens = sorted(set(word_atoms.values()))[:2]
            if len(tokens) == 2:
                bigram_freqs[(tokens[0], tokens[1])] = 1

        for (a, b), freq in bigram_freqs.most_common():
            if len(self.token_to_id) >= self.vocab_size:
                break
            if freq < self._min_bigram_freq:
                break  # most_common() is sorted; all remaining are also below threshold
            merged = a + b
            self.bigram_merges[(a, b)] = merged
            self.bigram_stats[merged] = freq
            self._add_token(merged)

        # Guarantee ≥1 bigram even if all candidates fell below min_bigram_freq.
        if not self.bigram_merges and bigram_freqs:
            (a, b), freq = bigram_freqs.most_common(1)[0]
            merged = a + b
            self.bigram_merges[(a, b)] = merged
            self.bigram_stats[merged] = freq
            self._add_token(merged)

    # ------------------------------------------------------------------ #
    # bpe (heap + doubly-linked list: O(n log n) vs GPT-2's O(n²))
    # ------------------------------------------------------------------ #
    def _bpe(self, symbols: List[str]) -> List[str]:
        n = len(symbols)
        if n < 2:
            return list(symbols)

        syms = list(symbols)
        # Array-based doubly linked list; nxt[n-1] = n acts as right sentinel.
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        alive = [True] * n

        ranks = self.bpe_ranks
        INF = float("inf")

        heap: List[Tuple[float, int]] = []
        for i in range(n - 1):
            r = ranks.get((syms[i], syms[i + 1]), INF)
            if r < INF:
                heapq.heappush(heap, (r, i))

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j >= n or not alive[j]:
                continue
            # Stale entry: syms[i] or syms[j] changed since this was pushed.
            if ranks.get((syms[i], syms[j]), INF) != r:
                continue

            syms[i] = syms[i] + syms[j]
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i

            # Left neighbour's right-pair changed.
            p = prev[i]
            if p >= 0:
                r2 = ranks.get((syms[p], syms[i]), INF)
                if r2 < INF:
                    heapq.heappush(heap, (r2, p))

            # i's new right-pair.
            k = nxt[i]
            if k < n:
                r2 = ranks.get((syms[i], syms[k]), INF)
                if r2 < INF:
                    heapq.heappush(heap, (r2, i))

        return [syms[i] for i in range(n) if alive[i]]

    def _encode_chunk(self, chunk: str) -> List[str]:
        symbols = self.cache.get(chunk)
        if symbols is None:
            symbols = self._bpe(
                [self.byte_encoder[b] for b in chunk.encode(TEXT_ENCODING)]
            )
            self.cache[chunk] = symbols
        return symbols

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        """Convert a text string into a list of token ids."""
        # Lazy-init for backward compat with tokenizers pickled without _encode_cache.
        if not hasattr(self, "_encode_cache"):
            self._encode_cache = {}
        cached = self._encode_cache.get(text)
        if cached is not None:
            return cached

        chunks = [self._encode_chunk(ch) for ch in self.pat.findall(text)]

        # Stage-B bigram pass: local refs avoid repeated attribute lookups.
        bm = self.bigram_merges
        out: List[str] = []
        i = 0
        n = len(chunks)
        while i < n:
            cur = chunks[i]
            if (
                i + 1 < n
                and len(cur) == 1
                and len(chunks[i + 1]) == 1
                and (cur[0], chunks[i + 1][0]) in bm
            ):
                out.append(bm[(cur[0], chunks[i + 1][0])])
                i += 2
            else:
                out.extend(cur)
                i += 1

        unk = self.special_tokens["[UNK]"]
        t2id = self.token_to_id
        result = [t2id.get(s, unk) for s in out]
        self._encode_cache[text] = result
        return result

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token ids back into a text string."""
        chars = []
        for idx in token_ids:
            token = self.id_to_token.get(idx)
            if token is None or token in self.special_tokens:
                continue
            chars.append(token)
        text = "".join(chars)
        data = bytearray(self.byte_decoder[c] for c in text)
        return data.decode(TEXT_ENCODING, errors="replace")

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Encode text and return (token_ids, char_offsets).

        char_offsets[i] = (start, end) character positions in `text` for token i.
        Used by train_ner_model.py (Strategy 1) for exact token-to-word alignment,
        replacing the slower O(n²) decode-based fallback.
        """
        # Pretokenize, preserving match positions in the original text.
        matches = list(self.pat.finditer(text))
        encoded_chunks: List[Tuple[int, int, List[str]]] = []
        for m in matches:
            symbols = self._encode_chunk(m.group())
            encoded_chunks.append((m.start(), m.end(), symbols))

        token_ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        unk = self.special_tokens["[UNK]"]

        i = 0
        n = len(encoded_chunks)
        while i < n:
            start, end, syms = encoded_chunks[i]
            # Stage-B bigram check (same logic as encode).
            if (
                i + 1 < n
                and len(syms) == 1
                and len(encoded_chunks[i + 1][2]) == 1
                and (syms[0], encoded_chunks[i + 1][2][0]) in self.bigram_merges
            ):
                merged = self.bigram_merges[(syms[0], encoded_chunks[i + 1][2][0])]
                token_ids.append(self.token_to_id.get(merged, unk))
                offsets.append((start, encoded_chunks[i + 1][1]))
                i += 2
            else:
                # Distribute char positions across subtokens of this chunk.
                chunk_str = text[start:end]
                chunk_bytes = chunk_str.encode(TEXT_ENCODING)
                byte_pos = 0
                for sym in syms:
                    sym_bytes = bytearray(self.byte_decoder[c] for c in sym)
                    nb = len(sym_bytes)
                    # Map byte range to char range within the chunk.
                    c_start = len(
                        chunk_bytes[:byte_pos].decode(TEXT_ENCODING, errors="replace")
                    )
                    c_end = len(
                        chunk_bytes[: byte_pos + nb].decode(
                            TEXT_ENCODING, errors="replace"
                        )
                    )
                    token_ids.append(self.token_to_id.get(sym, unk))
                    offsets.append((start + c_start, start + c_end))
                    byte_pos += nb
                i += 1

        return token_ids, offsets

    # ------------------------------------------------------------------ #
    # reporting (for the write-up)
    # ------------------------------------------------------------------ #
    def get_bigrams(self) -> List[Tuple[str, int]]:
        """Learned bigrams as (surface form, frequency), most frequent first."""
        items = sorted(self.bigram_stats.items(), key=lambda kv: -kv[1])
        return [(tok.replace(self.space_token, " ").strip(), f) for tok, f in items]
