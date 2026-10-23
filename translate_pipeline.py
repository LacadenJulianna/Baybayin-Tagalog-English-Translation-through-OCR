"""
translate_pipeline.py
=====================
Baybayin -> Tagalog -> English, after segmentation and glyph recognition.

Stages (each stage's output is kept in the returned log dict)
-------------------------------------------------------------
  0  offline resources   lexicon + prefix set, smoothed bigram LM, English lookup
  1  glyph observations  CNN class probabilities (top-k) + mark probabilities
                         -> syllable options with log-probabilities
  2  candidate search    prefix-pruned search per word; an omitted coda consonant is
                         allowed after any syllable (penalty); edit-distance fallback
                         marked "unverified"
  3  ranking             beam search over words:
                         alpha*visual + beta*log P_LM(w_i | w_{i-1}),
                         P_LM = lam*P_bigram + (1-lam)*P_unigram(add-k)
  4  English             dictionary lookup for single words, NLLB-200 for phrases
  5  evaluation          CER / WER / exact match / top-k word identification

Usage
-----
  # 1) sanity check of stages 2-3 with PERFECT recognition (no CNN needed)
  python translate_pipeline.py selftest --data dataset --n 2000

  # 2) word-image evaluation (needs models/cnn.pt from train_models.py)
  python translate_pipeline.py words --data dataset --cnn models/cnn.pt --limit 200

  # 3) one image (word or sentence photo)
  python translate_pipeline.py image path.png --data dataset --cnn models/cnn.pt [--mode photo]

Dependencies: numpy, pandas. Optional: rapidfuzz (fast fuzzy fallback),
torch (CNN), transformers (NLLB).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Reading table (Section 3.5)
# --------------------------------------------------------------------------- #
CLASS_CONSONANT: Dict[str, List[str]] = {
    "ba": ["b"], "dara": ["d", "r"], "ga": ["g"], "ha": ["h"], "ka": ["k"],
    "la": ["l"], "ma": ["m"], "na": ["n"], "nga": ["ng"], "pa": ["p"],
    "sa": ["s"], "ta": ["t"], "wa": ["w"], "ya": ["y"],
}
VOWEL_CLASS: Dict[str, List[str]] = {"a": ["a"], "ei": ["e", "i"], "ou": ["o", "u"]}
MARK_VOWELS: Dict[str, List[str]] = {
    "none": ["a"], "above": ["e", "i"], "below": ["o", "u"], "virama": [""],
}
MARKS = ("none", "above", "below", "virama")
# consonants that Baybayin commonly leaves unwritten at the end of a syllable
CODAS = ["n", "ng", "s", "t", "g", "m", "l", "k", "d", "p", "b", "r", "h", "w", "y"]


@dataclass
class TParams:
    alpha: float = 1.0          # weight of the visual score
    beta: float = 1.0           # weight of the language-model score
    lam: float = 0.7            # bigram / unigram interpolation
    k: float = 0.1              # add-k smoothing for unigrams
    coda_penalty: float = 2.0   # log-penalty per omitted coda consonant
    fuzzy_penalty: float = 1.5  # log-penalty per edit in the fallback
    search_beam: int = 400      # partial spellings kept per glyph position
    sentence_beam: int = 8      # hypotheses kept in the word-sequence beam
    n_best: int = 5
    min_class_prob: float = 0.05
    top_k_classes: int = 2


# --------------------------------------------------------------------------- #
# Step 0: offline resources
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zñ]+", text.lower())


class Lexicon:
    """Tagalog word set + set of every prefix (a flat trie)."""

    def __init__(self, words: Iterable[str]):
        self.words = {w for w in (str(x).strip().lower() for x in words)
                      if w and re.fullmatch(r"[a-zñ]+", w)}
        self.prefixes = set()
        for w in self.words:
            for i in range(1, len(w) + 1):
                self.prefixes.add(w[:i])
        self.by_len: Dict[int, List[str]] = defaultdict(list)
        for w in self.words:
            self.by_len[len(w)].append(w)

    @classmethod
    def from_data_dir(cls, data_dir: str) -> "Lexicon":
        import pandas as pd
        p = os.path.join(data_dir, "Tagalog Dictionary", "tagalog words.csv")
        # NOTE: this file has NO header row; the first word would be lost otherwise.
        df = pd.read_csv(p, header=None, usecols=[0])
        words = set(df[0].dropna().astype(str))
        q = os.path.join(data_dir, "Tagalog Dictionary", "diksyo_tagalog_vocabulary.csv")
        if os.path.exists(q):
            words |= set(pd.read_csv(q, usecols=["word"])["word"].dropna().astype(str))
        return cls(words)


FREQ_PRIOR = {"Everyday": 20, "Common": 20, "Occasional": 8, "Slang": 3,
              "Rare": 2, "Obsolete": 1}   # pseudo-counts for words found in diksyo


class BigramLM:
    """Unigram (add-k) + bigram counts, interpolated:
         P(w|prev) = lam * c(prev,w)/c(prev) + (1-lam) * P_uni(w)
    Tokens outside the lexicon are mapped to <unk> so the tables stay small."""

    def __init__(self, vocab: Iterable[str], k: float = 0.1, lam: float = 0.7):
        self.vocab = set(vocab)
        self.k, self.lam = k, lam
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.prev_count: Counter = Counter()
        self.total = 0

    def _map(self, toks: List[str]) -> List[str]:
        return [t if t in self.vocab else "<unk>" for t in toks]

    def fit_texts(self, texts: Iterable[str]) -> None:
        for text in texts:
            for sent in re.split(r"[.!?\n;:]+", str(text)):
                toks = self._map(tokenize(sent))
                if not toks:
                    continue
                seq = ["<s>"] + toks
                self.uni.update(toks)
                self.total += len(toks)
                for a, b in zip(seq, seq[1:]):
                    self.bi[(a, b)] += 1
                    self.prev_count[a] += 1

    def add_prior(self, prior: Dict[str, float]) -> None:
        for w, c in prior.items():
            if w in self.vocab:
                self.uni[w] += c
                self.total += c

    def p_uni(self, w: str) -> float:
        V = len(self.vocab) + 1
        return (self.uni.get(w, 0) + self.k) / (self.total + self.k * V)

    def logp(self, w: str, prev: Optional[str]) -> float:
        """prev=None -> unigram only (first word of a phrase)."""
        pu = self.p_uni(w)
        if prev is None:
            return math.log(pu)
        cprev = self.prev_count.get(prev, 0)
        pb = self.bi.get((prev, w), 0) / cprev if cprev else 0.0
        lam = self.lam if cprev else 0.0
        return math.log(lam * pb + (1.0 - lam) * pu)

    def key(self, w: str) -> str:
        return w if w in self.vocab else "<unk>"

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(path: str) -> "BigramLM":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_lm(data_dir: str, lex: Lexicon, p: TParams, cache: Optional[str] = None,
             max_rows: int = 0) -> BigramLM:
    """Bigram counts from the Tagalog side of 'Tagalog to English.csv' (a modern,
    partly machine-translated corpus - state that limitation in the paper),
    plus frequency-label pseudo-counts from diksyo."""
    import pandas as pd
    if cache and os.path.exists(cache):
        lm = BigramLM.load(cache)
        lm.k, lm.lam = p.k, p.lam
        return lm
    lm = BigramLM(lex.words, p.k, p.lam)
    cpath = os.path.join(data_dir, "Tagalog English Dictionary Dataset", "Tagalog to English.csv")
    if os.path.exists(cpath):
        df = pd.read_csv(cpath, usecols=["tagalog"])
        if max_rows:
            df = df.head(max_rows)
        lm.fit_texts(df["tagalog"].dropna().astype(str))
    dpath = os.path.join(data_dir, "Tagalog Dictionary", "diksyo_tagalog_vocabulary.csv")
    if os.path.exists(dpath):
        d = pd.read_csv(dpath, usecols=["word", "frequency"]).dropna()
        prior = {}
        for w, fq in zip(d["word"].astype(str).str.lower(), d["frequency"]):
            prior[w] = max(prior.get(w, 0), FREQ_PRIOR.get(fq, 1))
        lm.add_prior(prior)
    if cache:
        lm.save(cache)
    return lm


# --------------------------------------------------------------------------- #
# Step 1: glyph observations -> syllable options
# --------------------------------------------------------------------------- #
@dataclass
class GlyphObs:
    class_logp: Dict[str, float]            # class name -> log P(class | image)
    mark_logp: Dict[str, float] = field(default_factory=lambda: {
        "none": 0.0, "above": -20.0, "below": -20.0, "virama": -20.0})


def _lse(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def syllable_options(g: GlyphObs) -> List[Tuple[str, float]]:
    """All Latin readings of one glyph with log-probabilities.
    Vowel ambiguity (e/i, o/u) and d/r are NOT split: every reading gets the full
    probability, so the language model decides between them."""
    opts: Dict[str, float] = {}

    def add(s: str, lp: float) -> None:
        opts[s] = _lse(opts[s], lp) if s in opts else lp

    for cls, clp in g.class_logp.items():
        if cls in VOWEL_CLASS:
            for v in VOWEL_CLASS[cls]:
                add(v, clp)
        elif cls in CLASS_CONSONANT:
            for mark, mlp in g.mark_logp.items():
                for cons in CLASS_CONSONANT[cls]:
                    for v in MARK_VOWELS[mark]:
                        add(cons + v, clp + mlp)
    return sorted(opts.items(), key=lambda t: -t[1])


def perfect_obs(cls: str, mark: str = "none", conf: float = 0.97) -> GlyphObs:
    rest = (1 - conf) / 3
    return GlyphObs({cls: math.log(conf)},
                    {m: math.log(conf if m == mark else rest) for m in MARKS})


def obs_from_probs(probs: np.ndarray, classes: Sequence[str], marks, p: TParams,
                   mark_fn=None) -> GlyphObs:
    """probs: softmax output for one glyph; marks: segmentation Marks attached to it."""
    order = np.argsort(-probs)[:p.top_k_classes]
    cl = {classes[i]: float(math.log(max(probs[i], 1e-9)))
          for i in order if probs[i] >= p.min_class_prob or i == order[0]}
    return GlyphObs(cl, (mark_fn or rule_mark_logp)(marks))


def rule_mark_logp(marks) -> Dict[str, float]:
    """PLACEHOLDER until the diacritic classifier is trained: soft probabilities from
    segmentation geometry (above/below) and the dot/cross shape hint."""
    if not marks:
        pr = {"none": 0.88, "above": 0.04, "below": 0.04, "virama": 0.04}
    else:
        m = max(marks, key=lambda x: x.bbox[2] * x.bbox[3])
        vir = 0.55 if m.shape_hint == "cross" else 0.15
        kud = 0.40 if m.shape_hint == "cross" else 0.80
        pr = {"none": 0.03, "above": 0.03, "below": 0.03, "virama": vir}
        pr[m.position] += kud
    s = sum(pr.values())
    return {k: math.log(v / s) for k, v in pr.items()}


# --------------------------------------------------------------------------- #
# Step 2: prefix-pruned candidate search (+ fuzzy fallback)
# --------------------------------------------------------------------------- #
@dataclass
class Cand:
    word: str
    visual: float
    unverified: bool = False
    n_coda: int = 0


def search_word(obs: Sequence[GlyphObs], lex: Lexicon, p: TParams) -> List[Cand]:
    """Syllable-by-syllable search. A partial spelling survives only if it is a prefix of
    some lexicon word. After each syllable an omitted coda consonant may be inserted."""
    states: Dict[str, Tuple[float, int]] = {"": (0.0, 0)}
    for g in obs:
        opts = syllable_options(g)
        new: Dict[str, Tuple[float, int]] = {}
        for s, (sc, nc) in states.items():
            for syl, lp in opts:
                t = s + syl
                if t in lex.prefixes:
                    v = sc + lp
                    if t not in new or v > new[t][0]:
                        new[t] = (v, nc)
        grown = dict(new)
        for t, (sc, nc) in new.items():
            for c in CODAS:
                u = t + c
                if u in lex.prefixes:
                    v = sc - p.coda_penalty
                    if u not in grown or v > grown[u][0]:
                        grown[u] = (v, nc + 1)
        if not grown:
            states = {}
            break
        states = dict(sorted(grown.items(), key=lambda kv: -kv[1][0])[:p.search_beam])

    out = [Cand(w, sc, False, nc) for w, (sc, nc) in states.items() if w in lex.words]
    if out:
        return sorted(out, key=lambda c: -c.visual)
    return fuzzy_fallback(obs, lex, p)


def greedy_reading(obs: Sequence[GlyphObs]) -> Tuple[str, float]:
    s, sc = "", 0.0
    for g in obs:
        opts = syllable_options(g)
        if opts:
            s += opts[0][0]
            sc += opts[0][1]
    return s, sc


def _lev(a: str, b: str, cutoff: int) -> int:
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


def fuzzy_fallback(obs: Sequence[GlyphObs], lex: Lexicon, p: TParams,
                   max_dist: int = 3, top: int = 3) -> List[Cand]:
    reading, sc = greedy_reading(obs)
    if not reading:
        return []
    pool: List[str] = []
    for L in range(max(1, len(reading) - max_dist), len(reading) + max_dist + 1):
        pool.extend(lex.by_len.get(L, []))
    scored: List[Tuple[int, str]] = []
    try:
        from rapidfuzz.distance import Levenshtein
        for w in pool:
            d = Levenshtein.distance(reading, w, score_cutoff=max_dist)
            if d <= max_dist:
                scored.append((d, w))
    except ImportError:
        for w in pool:
            d = _lev(reading, w, max_dist)
            if d <= max_dist:
                scored.append((d, w))
    scored.sort()
    cands = [Cand(w, sc - p.fuzzy_penalty * d, True) for d, w in scored[:top]]
    # always keep the raw reading so the output is never empty
    cands.append(Cand(reading, sc - p.fuzzy_penalty * (max_dist + 1), True))
    return cands


# --------------------------------------------------------------------------- #
# Step 3: ranking (beam search over word sequences)
# --------------------------------------------------------------------------- #
def rank_sequence(word_cands: List[List[Cand]], lm: BigramLM, p: TParams
                  ) -> List[Tuple[float, List[Cand]]]:
    """alpha*visual + beta*logP_LM. The first word uses the unigram only, because an
    isolated word has no context. Beam search is not guaranteed optimal."""
    hyps: List[Tuple[float, List[Cand]]] = [(0.0, [])]
    for cands in word_cands:
        if not cands:
            continue
        nxt = []
        for sc, seq in hyps:
            prev = lm.key(seq[-1].word) if seq else None
            for c in cands:
                s = sc + p.alpha * c.visual + p.beta * lm.logp(lm.key(c.word), prev)
                nxt.append((s, seq + [c]))
        hyps = sorted(nxt, key=lambda t: -t[0])[:p.sentence_beam]
    return hyps


# --------------------------------------------------------------------------- #
# Step 4: English
# --------------------------------------------------------------------------- #
class EnglishTranslator:
    def __init__(self, data_dir: str, use_nllb: bool = True,
                 nllb_name: str = "facebook/nllb-200-distilled-600M"):
        import pandas as pd
        self.dict: Dict[str, str] = {}
        base = os.path.join(data_dir, "Tagalog English Dictionary Dataset")
        p2 = os.path.join(base, "tagalog__english_dictionary.csv")
        if os.path.exists(p2):               # lower priority: filled first, overwritten below
            df = pd.read_csv(p2).dropna()
            for w, d in zip(df["tagalog_word"].astype(str).str.lower(),
                            df["english_definition"].astype(str)):
                gloss = re.sub(r"^\S+\s+(?:n|v|adj|adv|prep|conj|pron|interj|part)\.?\s+", "", d)
                self.dict.setdefault(w, gloss.strip())
        p1 = os.path.join(data_dir, "Tagalog Dictionary", "diksyo_tagalog_vocabulary.csv")
        if os.path.exists(p1):               # preferred source (short English glosses)
            df = pd.read_csv(p1, usecols=["word", "translation"]).dropna()
            for w, t in zip(df["word"].astype(str).str.lower(), df["translation"].astype(str)):
                self.dict[w] = t
        self.use_nllb, self.nllb_name = use_nllb, nllb_name
        self._tok = self._model = None

    def _load_nllb(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.nllb_name, src_lang="tgl_Latn")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_name)
            return True
        except Exception as e:          # no transformers / no internet / model not found
            print(f"[translator] NLLB unavailable ({type(e).__name__}); using dictionary glosses")
            self.use_nllb = False
            return False

    def translate(self, words: List[str]) -> Dict[str, str]:
        if len(words) == 1 and words[0] in self.dict:
            return {"english": self.dict[words[0]], "method": "dictionary"}
        if self.use_nllb and self._load_nllb():
            inp = self._tok(" ".join(words), return_tensors="pt")
            out = self._model.generate(
                **inp, forced_bos_token_id=self._tok.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=128, num_beams=4)
            return {"english": self._tok.batch_decode(out, skip_special_tokens=True)[0],
                    "method": "nllb-200"}
        glosses = [self.dict.get(w, f"[{w}]").split(";")[0] for w in words]
        return {"english": " ".join(glosses), "method": "word-by-word glosses"}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class CNNClassifier:
    """Wraps models/cnn.pt from train_models.py."""

    def __init__(self, path: str):
        from train_models import load_cnn
        self.model, self.classes = load_cnn(path)

    def predict_proba(self, images_uint8: np.ndarray) -> np.ndarray:
        from train_models import predict_glyphs
        return predict_glyphs(self.model, self.classes, images_uint8)[1]


class BaybayinTranslator:
    def __init__(self, lex: Lexicon, lm: BigramLM, params: Optional[TParams] = None,
                 translator: Optional[EnglishTranslator] = None, classifier=None,
                 mark_fn=None):
        self.lex, self.lm = lex, lm
        self.p = params or TParams()
        self.translator, self.clf, self.mark_fn = translator, classifier, mark_fn

    # stages 2-4 on observations: List[word] of List[GlyphObs]
    def translate_observations(self, words_obs: List[List[GlyphObs]],
                               do_english: bool = True) -> Dict:
        t0 = time.time()
        word_cands = [search_word(w, self.lex, self.p) for w in words_obs]
        ranked = rank_sequence(word_cands, self.lm, self.p)
        best_sc, best = ranked[0] if ranked else (0.0, [])
        words = [c.word for c in best]
        log = {
            "stage2_candidates": [[asdict(c) for c in cs[:10]] for cs in word_cands],
            "stage3_ranked": [{"score": round(s, 3), "words": [c.word for c in seq],
                               "unverified": [c.unverified for c in seq]}
                              for s, seq in ranked[:self.p.n_best]],
            "tagalog": " ".join(words),
            "unverified_words": [c.word for c in best if c.unverified],
        }
        if do_english and self.translator and words:
            log["stage4_english"] = self.translator.translate(words)
        log["seconds"] = round(time.time() - t0, 3)
        return log

    # stage 1 from a segmentation Result
    def observations_from_result(self, res, baybayin_only: bool = True):
        lines = []
        for line in res.lines:
            if baybayin_only and line.script != "baybayin":
                continue
            words_obs = []
            for w in line.words:
                imgs = np.stack([g.image for g in w.glyphs])
                probs = self.clf.predict_proba(imgs)
                words_obs.append([obs_from_probs(probs[i], self.clf.classes, g.marks,
                                                 self.p, self.mark_fn)
                                  for i, g in enumerate(w.glyphs)])
            lines.append(words_obs)
        return lines

    def translate_image(self, path: str, mode: str = "clean",
                        single_word: bool = False, roi=None) -> Dict:
        from baybayin_preprocess import process_image
        res = process_image(path, mode=mode, single_word=single_word, roi=roi)
        out_lines = []
        for words_obs in self.observations_from_result(res):
            out_lines.append(self.translate_observations(words_obs))
        return {"image": path, "lines": out_lines}


# --------------------------------------------------------------------------- #
# Step 5: metrics
# --------------------------------------------------------------------------- #
def edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(1, len(r))


def best_cer(refs: Sequence[str], hyp: str) -> float:
    """Several valid readings are allowed (vowel ambiguity): score against the closest."""
    return min(cer(r, hyp) for r in refs)


def label_from_filename(fn: str) -> str:
    """'1 Gwapo.PNG' -> 'gwapo'"""
    stem = os.path.splitext(os.path.basename(fn))[0]
    return re.sub(r"^\d+[\s_.\-]*", "", stem).strip().lower()


# --------------------------------------------------------------------------- #
# Oracle self-test (perfect recognition) - isolates stages 2-3
# --------------------------------------------------------------------------- #
def word_to_glyphs(word: str) -> Optional[List[Tuple[str, str]]]:
    """Idealised Baybayin spelling of a Tagalog word: (class, mark) per glyph, with coda
    consonants omitted (traditional rule). None if the word needs letters Baybayin lacks."""
    vow = "aeiou"
    out: List[Tuple[str, str]] = []
    i, n = 0, len(word)
    mark_of = {"a": "none", "e": "above", "i": "above", "o": "below", "u": "below"}
    while i < n:
        ch = word[i]
        if ch in vow:
            out.append(({"a": "a", "e": "ei", "i": "ei", "o": "ou", "u": "ou"}[ch], "none"))
            i += 1
            continue
        if word.startswith("ng", i) and i + 2 < n and word[i + 2] in vow:
            cons, step = "nga", 2
        else:
            cons, step = ch + "a", 1
        key = cons
        if key not in CLASS_CONSONANT and key != "dara":
            if ch in "dr" and step == 1:
                key = "dara"
            else:
                return None
        if ch in "dr" and step == 1:
            key = "dara"
        if i + step < n and word[i + step] in vow:
            out.append((key, mark_of[word[i + step]]))
            i += step + 1
        else:
            i += step           # coda consonant: not written
    return out


def selftest(data_dir: str, n: int, p: TParams, lm_cache: Optional[str], seed: int = 0) -> Dict:
    lex = Lexicon.from_data_dir(data_dir)
    t0 = time.time()
    lm = build_lm(data_dir, lex, p, cache=lm_cache)
    print(f"lexicon: {len(lex.words)} words, {len(lex.prefixes)} prefixes; "
          f"LM: {lm.total} tokens, {len(lm.bi)} bigrams ({time.time() - t0:.0f}s)")
    rnd = random.Random(seed)
    pool = sorted(lex.words)
    rnd.shuffle(pool)
    tests = []
    for w in pool:
        g = word_to_glyphs(w)
        if g:
            tests.append((w, g))
        if len(tests) >= n:
            break
    tr = BaybayinTranslator(lex, lm, p)
    top1 = top5 = recall = 0
    ncand = []
    misses = []
    t0 = time.time()
    for w, glyphs in tests:
        obs = [perfect_obs(c, m) for c, m in glyphs]
        cands = search_word(obs, lex, p)
        ncand.append(len(cands))
        ranked = rank_sequence([cands], lm, p)
        order = [seq[0].word for _, seq in ranked]
        # full n-best (sentence_beam may be < number of candidates)
        allr = sorted(((p.alpha * c.visual + p.beta * lm.logp(lm.key(c.word), None), c.word)
                       for c in cands), reverse=True)
        order = [x[1] for x in allr]
        recall += w in order
        top1 += bool(order) and order[0] == w
        top5 += w in order[:5]
        if w not in order[:1] and len(misses) < 8:
            misses.append((w, order[:3]))
    N = len(tests)
    res = {"words_tested": N, "candidate_recall": recall / N, "top1": top1 / N,
           "top5": top5 / N, "mean_candidates": float(np.mean(ncand)),
           "median_candidates": float(np.median(ncand)), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=2))
    print("sample misses (true word -> top-3 predicted):")
    for w, o in misses:
        print(f"  {w:>16} -> {o}")
    return res


# --------------------------------------------------------------------------- #
# Word-image evaluation
# --------------------------------------------------------------------------- #
def evaluate_word_images(tr: BaybayinTranslator, img_dir: str, limit: int = 0,
                         seed: int = 0, out_json: Optional[str] = None) -> Dict:
    from baybayin_preprocess import process_image
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    random.Random(seed).shuffle(files)
    if limit:
        files = files[:limit]
    rows, ex1, ex5, cers = [], 0, 0, []
    for fn in files:
        ref = label_from_filename(fn)
        try:
            res = process_image(os.path.join(img_dir, fn), mode="clean", single_word=True)
            lines = tr.observations_from_result(res)
            if not lines or not lines[0]:
                raise ValueError("no glyphs found")
            words_obs = lines[0]
            wobs = [g for w in words_obs for g in w]      # single word: merge groups
            cands = search_word(wobs, tr.lex, tr.p)
            allr = sorted(((tr.p.alpha * c.visual + tr.p.beta *
                            tr.lm.logp(tr.lm.key(c.word), None), c) for c in cands),
                          key=lambda t: -t[0], reverse=True)
            order = [c.word for _, c in allr]
        except Exception as e:
            order = []
            rows.append({"file": fn, "ref": ref, "error": str(e)})
            cers.append(1.0)
            continue
        hyp = order[0] if order else ""
        ex1 += hyp == ref
        ex5 += ref in order[:5]
        cers.append(cer(ref, hyp))
        rows.append({"file": fn, "ref": ref, "top5": order[:5],
                     "unverified": bool(cands and cands[0].unverified)})
    N = len(files)
    summary = {"images": N, "exact_match_top1": ex1 / N, "top5_word_identification": ex5 / N,
               "mean_cer_vs_filename": float(np.mean(cers))}
    print(json.dumps(summary, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build(args) -> BaybayinTranslator:
    p = TParams()
    lex = Lexicon.from_data_dir(args.data)
    lm = build_lm(args.data, lex, p, cache=args.lm_cache)
    tr_en = EnglishTranslator(args.data, use_nllb=not args.no_nllb)
    clf = CNNClassifier(args.cnn)
    return BaybayinTranslator(lex, lm, p, tr_en, clf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--data", default="dataset")
        s.add_argument("--lm-cache", default="models/lm.pkl")

    s = sub.add_parser("selftest")
    common(s)
    s.add_argument("--n", type=int, default=2000)

    s = sub.add_parser("words")
    common(s)
    s.add_argument("--cnn", default="models/cnn.pt")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-nllb", action="store_true")
    s.add_argument("--out", default="models/word_eval.json")

    s = sub.add_parser("image")
    common(s)
    s.add_argument("path")
    s.add_argument("--cnn", default="models/cnn.pt")
    s.add_argument("--mode", choices=["clean", "photo"], default="clean")
    s.add_argument("--word", action="store_true", help="treat the image as a single word")
    s.add_argument("--no-nllb", action="store_true")

    args = ap.parse_args()
    if args.cmd in ("selftest",):
        os.makedirs(os.path.dirname(args.lm_cache) or ".", exist_ok=True)
        selftest(args.data, args.n, TParams(), args.lm_cache)
    elif args.cmd == "words":
        os.makedirs(os.path.dirname(args.lm_cache) or ".", exist_ok=True)
        tr = _build(args)
        evaluate_word_images(tr, os.path.join(args.data, "Baybayin Word Images"),
                             args.limit, out_json=args.out)
    else:
        os.makedirs(os.path.dirname(args.lm_cache) or ".", exist_ok=True)
        tr = _build(args)
        out = tr.translate_image(args.path, mode=args.mode, single_word=args.word)
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
