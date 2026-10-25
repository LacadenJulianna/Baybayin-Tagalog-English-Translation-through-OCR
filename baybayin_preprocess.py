"""
baybayin_preprocess.py
======================
Preprocessing and segmentation for the Baybayin OCR transliteration translator
(paper Section 3.5).

Pipeline
--------
  image
    -> grayscale -> median filter -> binarize
         clean mode : Otsu                         (character / word images)
         photo mode : illumination normalisation + Otsu + ruled-line removal
                      + junk removal (pen, spiral binding, dashed header)
    -> (clean mode) deskew + crop to text
    -> connected-component labelling
    -> split components into base characters / kudlit-virama marks / noise
    -> group bases into lines, then words (horizontal gaps)
    -> assign each mark to its nearest base (horizontal overlap, fallback to
        nearest centroid within a distance limit)
    -> normalise every base (and mark) to a 28x28 binary image

Output polarity: ink = 255, background = 0 (MNIST style).
Use the SAME normalisation for training data and inference.

Usage
-----
  python baybayin_preprocess.py char      path/to/char.jpg
  python baybayin_preprocess.py word      path/to/word.png  --out out/
  python baybayin_preprocess.py sentence  path/to/photo.jpg --mode photo --out out/
  python baybayin_preprocess.py dataset   dataset/          --out out/

As a library:
  from baybayin_preprocess import process_image, preprocess_character_image
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GLYPH_SIZE = 28


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # preprocessing
    median_ksize: int = 3            # median filter size (odd)
    deskew_max_angle: float = 10.0   # degrees searched either side
    crop_pad: int = 6                # padding kept around the text crop
    # component classification (areas relative to a reference base area)
    min_abs_area: int = 6            # components smaller than this are noise
    noise_ratio: float = 0.01        # < noise_ratio * ref_area  -> noise
    mark_ratio: float = 0.30         # < mark_ratio  * ref_area  -> kudlit/virama
    # kudlit assignment
    mark_max_dist_ratio: float = 1.0  # max centroid distance, x median base width
    # grouping
    line_split_ratio: float = 0.6    # new line if |dy| > ratio * median base height
    word_gap_ratio: float = 0.6      # new word if gap > ratio * median base width
    adaptive_word_gap: bool = False  # estimate spaces from this line's gap distribution
    # glyph normalisation
    glyph_margin: float = 0.12       # fraction of side kept as blank border
    # photo mode
    ruled_line_frac: float = 0.04    # horizontal opening kernel = frac * image width
    illum_kernel_frac: float = 0.012  # illumination close kernel = frac * image width
    max_comp_frac: float = 0.25      # drop components larger than this fraction of image
    border_margin_frac: float = 0.06  # drop components within this fraction of the border
    flat_aspect: float = 6.0         # drop very flat components (dashes, rules)
    # script separation heuristic (sentence images that mix Baybayin and Latin)
    latin_median_aspect: float = 1.5  # median w/h of line components above this -> Latin
    min_line_glyphs: int = 1         # drop lines with fewer bases


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Comp:
    idx: int
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class Mark:
    bbox: Tuple[int, int, int, int]
    position: str                # "above" | "below"
    shape_hint: str              # "dot" | "cross"
    image: np.ndarray            # 28x28 binary


@dataclass
class Glyph:
    bbox: Tuple[int, int, int, int]
    image: np.ndarray            # 28x28 binary, base character only
    marks: List[Mark] = field(default_factory=list)


@dataclass
class Word:
    glyphs: List[Glyph]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _union([g.bbox for g in self.glyphs] +
                      [m.bbox for g in self.glyphs for m in g.marks])


@dataclass
class Line:
    words: List[Word]
    script: str                  # "baybayin" | "latin"
    median_aspect: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _union([w.bbox for w in self.words])


@dataclass
class Result:
    binary: np.ndarray
    lines: List[Line]
    orphan_marks: List[Tuple[int, int, int, int]]
    noise_count: int

    def to_dict(self) -> Dict:
        """JSON-serialisable summary (no images)."""
        return {
            "noise_components": self.noise_count,
            "orphan_marks": [list(b) for b in self.orphan_marks],
            "lines": [{
                "bbox": list(l.bbox),
                "script": l.script,
                "median_aspect": round(l.median_aspect, 3),
                "words": [{
                    "bbox": list(w.bbox),
                    "glyphs": [{
                        "bbox": list(g.bbox),
                        "marks": [{"bbox": list(m.bbox),
                                   "position": m.position,
                                   "shape_hint": m.shape_hint} for m in g.marks],
                    } for g in w.glyphs],
                } for w in l.words],
            } for l in self.lines],
        }


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Load as 3-channel BGR. Transparent PNGs are composited on white."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


# --------------------------------------------------------------------------- #
# Binarisation
# --------------------------------------------------------------------------- #
def binarize_clean(img: np.ndarray, p: Params) -> np.ndarray:
    """Grayscale -> median filter -> Otsu. Ink = 255."""
    gray = cv2.medianBlur(_gray(img), _odd(p.median_ksize))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.count_nonzero(bw) > 0.5 * bw.size:
        bw = cv2.bitwise_not(bw)
    return bw


def binarize_photo(img: np.ndarray, p: Params) -> np.ndarray:
    """Variant for photographed pages (uneven light, coloured art, ruled lines)."""
    gray = cv2.medianBlur(_gray(img), _odd(p.median_ksize))
    H, W = gray.shape

    # 1. Illumination normalisation
    k = _odd(max(15, int(W * p.illum_kernel_frac)))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, bg, scale=255)

    # 2. Otsu
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 3. Remove horizontal ruled lines
    kw = max(25, int(W * p.ruled_line_frac))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    bw = cv2.subtract(bw, lines)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

    # 4. Drop junk
    return _drop_photo_junk(bw, p)


def _drop_photo_junk(bw: np.ndarray, p: Params) -> np.ndarray:
    H, W = bw.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    bm = max(2, int(p.border_margin_frac * min(H, W)))
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        touches = x <= bm or y <= bm or x + w >= W - bm or y + h >= H - bm
        too_big = a > p.max_comp_frac * H * W or w > 0.3 * W or h > 0.3 * H
        flat = h > 0 and (w / h) > p.flat_aspect
        if not (touches or too_big or flat):
            keep[i] = 255
    return keep[labels]


# --------------------------------------------------------------------------- #
# Deskew / Crop
# --------------------------------------------------------------------------- #
def deskew(bw: np.ndarray, max_angle: float = 10.0) -> Tuple[np.ndarray, float]:
    ys, xs = np.nonzero(bw)
    if len(xs) < 50:
        return bw, 0.0
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    if w < 1.5 * h:
        return bw, 0.0

    scale = 256.0 / max(bw.shape)
    small = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else bw
    sh, sw = small.shape
    centre = (sw / 2.0, sh / 2.0)

    best_angle, best_score = 0.0, -1.0
    for ang in np.arange(-max_angle, max_angle + 0.01, 0.5):
        M = cv2.getRotationMatrix2D(centre, float(ang), 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(ang)

    if abs(best_angle) < 0.25:
        return bw, 0.0
    H, W = bw.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), best_angle, 1.0)
    rot = cv2.warpAffine(bw, M, (W, H), flags=cv2.INTER_LINEAR)
    _, rot = cv2.threshold(rot, 127, 255, cv2.THRESH_BINARY)
    return rot, best_angle


def crop_to_content(bw: np.ndarray, pad: int = 6) -> Tuple[np.ndarray, Tuple[int, int]]:
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        return bw, (0, 0)
    H, W = bw.shape
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    return bw[y0:y1, x0:x1], (x0, y0)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize_glyph(mask: np.ndarray, size: int = GLYPH_SIZE,
                    margin: float = 0.12) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    crop = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8) * 255
    h, w = crop.shape
    side = max(h, w)
    border = int(round(side * margin))
    canvas = np.zeros((side + 2 * border, side + 2 * border), np.uint8)
    oy = border + (side - h) // 2
    ox = border + (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    small = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return ((small >= 80).astype(np.uint8)) * 255


def preprocess_character_image(path_or_img, p: Optional[Params] = None) -> np.ndarray:
    p = p or Params()
    img = load_image(path_or_img) if isinstance(path_or_img, str) else path_or_img
    bw = binarize_clean(img, p)
    return normalize_glyph(bw, GLYPH_SIZE, p.glyph_margin)


# --------------------------------------------------------------------------- #
# Components classification & segmentation
# --------------------------------------------------------------------------- #
def _components(bw: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = [Comp(i, int(stats[i][0]), int(stats[i][1]), int(stats[i][2]),
                  int(stats[i][3]), int(stats[i][4])) for i in range(1, n)]
    return comps, labels


def _merge_stacked_strokes(comps: List[Comp], labels: np.ndarray,
                           p: Params) -> List[Comp]:
    """Join nearly touching, vertically stacked strokes in a single word.

    Some handwritten Baybayin vowels are drawn as a large base and a detached
    kudlit. When the kudlit is too large for the area heuristic, treating it as
    a separate consonant creates an extra syllable. Merge only components with
    strong horizontal overlap and a tiny vertical gap; ordinary left-to-right
    neighboring glyphs are left separate.
    """
    merged = list(comps)
    if len(merged) < 2:
        return merged
    median_h = float(np.median([c.h for c in merged]))
    max_gap = max(2.0, 0.10 * median_h)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(merged):
            for j in range(i + 1, len(merged)):
                b = merged[j]
                overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                if overlap / max(1, min(a.w, b.w)) < 0.8:
                    continue
                if a.y + a.h <= b.y:
                    gap = b.y - (a.y + a.h)
                elif b.y + b.h <= a.y:
                    gap = a.y - (b.y + b.h)
                else:
                    continue  # components already overlap vertically
                if gap > max_gap:
                    continue
                labels[labels == b.idx] = a.idx
                merged[i] = Comp(a.idx, min(a.x, b.x), min(a.y, b.y),
                                 max(a.x + a.w, b.x + b.w) - min(a.x, b.x),
                                 max(a.y + a.h, b.y + b.h) - min(a.y, b.y),
                                 a.area + b.area)
                del merged[j]
                changed = True
                break
            if changed:
                break
    return merged


def _classify(comps: List[Comp], p: Params):
    if not comps:
        return [], [], []
    areas = np.array([c.area for c in comps], float)
    p75 = np.percentile(areas, 75)
    ref = float(np.median(areas[areas >= 0.5 * p75]))
    bases, marks, noise = [], [], []
    for c in comps:
        if c.area < max(p.min_abs_area, p.noise_ratio * ref):
            noise.append(c)
        elif c.area < p.mark_ratio * ref:
            marks.append(c)
        else:
            bases.append(c)
    return bases, marks, noise


def _group_lines(bases: List[Comp], p: Params) -> List[List[Comp]]:
    if not bases:
        return []
    thr = p.line_split_ratio * float(np.median([b.h for b in bases]))
    lines: List[List[Comp]] = []
    for b in sorted(bases, key=lambda c: c.cy):
        if lines and abs(b.cy - np.mean([c.cy for c in lines[-1]])) <= thr:
            lines[-1].append(b)
        else:
            lines.append([b])
    lines.sort(key=lambda l: np.mean([c.cy for c in l]))
    return [sorted(l, key=lambda c: c.x) for l in lines]


def _shape_hint(m: Comp) -> str:
    fill = m.area / float(max(1, m.w * m.h))
    return "dot" if fill >= 0.62 else "cross"


def _assign_marks(lines: List[List[Comp]], marks: List[Comp], p: Params):
    per_line: List[Dict[int, List[Comp]]] = [dict() for _ in lines]
    orphans: List[Comp] = []
    if not lines:
        return per_line, list(marks)

    line_cy = [np.mean([c.cy for c in l]) for l in lines]
    line_h = [np.median([c.h for c in l]) for l in lines]

    for m in marks:
        li = int(np.argmin([abs(m.cy - cy) for cy in line_cy]))
        if abs(m.cy - line_cy[li]) > 1.2 * line_h[li]:
            orphans.append(m)
            continue
        bases = lines[li]
        med_w = float(np.median([b.w for b in bases]))

        best, best_key = None, None
        for b in bases:
            ov = min(m.x + m.w, b.x + b.w) - max(m.x, b.x)
            if ov > 0:
                key = (ov / m.w, -abs(m.cy - b.cy))
                if best_key is None or key > best_key:
                    best, best_key = b, key

        if best is None:
            d, cand = min(((abs(m.cx - b.cx), b) for b in bases), key=lambda t: t[0])
            if d <= p.mark_max_dist_ratio * med_w:
                best = cand
        if best is None:
            orphans.append(m)
        else:
            per_line[li].setdefault(best.idx, []).append(m)
    return per_line, orphans


def _adaptive_word_gap_ratio(bases: List[Comp], p: Params) -> float:
    """Estimate a line-specific word-space threshold in median glyph widths."""
    if len(bases) < 2:
        return p.word_gap_ratio
    ordered = sorted(bases, key=lambda comp: comp.x)
    median_width = float(np.median([comp.w for comp in ordered]))
    normalized_gaps = [
        (right.x - (left.x + left.w)) / max(1.0, median_width)
        for left, right in zip(ordered, ordered[1:])
    ]
    # Wider handwriting has larger ordinary gaps; dense writing has smaller
    # gaps. Scaling the median adapts to both while bounding outliers.
    return float(np.clip(1.5 * np.median(normalized_gaps), 0.15, 1.0))


def _split_words(bases: List[Comp], p: Params,
                 word_gap_ratio: Optional[float] = None) -> List[List[Comp]]:
    if not bases:
        return []
    med_w = float(np.median([b.w for b in bases]))
    words, cur = [], [bases[0]]
    for prev, nxt in zip(bases, bases[1:]):
        gap = nxt.x - (prev.x + prev.w)
        threshold = p.word_gap_ratio if word_gap_ratio is None else word_gap_ratio
        if gap > threshold * med_w:
            words.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    words.append(cur)
    return words


def _line_script(bases: List[Comp], p: Params) -> Tuple[str, float]:
    asp = float(np.median([b.w / max(1, b.h) for b in bases]))
    return ("latin" if asp > p.latin_median_aspect else "baybayin"), asp


def segment(bw: np.ndarray, p: Optional[Params] = None,
            single_word: bool = False, merge_stacked: bool = False) -> Result:
    p = p or Params()
    comps, labels = _components(bw)
    if single_word and merge_stacked:
        comps = _merge_stacked_strokes(comps, labels, p)
    bases, marks, noise = _classify(comps, p)
    if single_word and bases:
        # Word-reference images contain one word. Handwritten glyphs can sit at
        # different vertical positions, so normal line clustering may split a
        # single slanted word into multiple lines and downstream code may only
        # recognize the first fragment. Preserve every base glyph in x order.
        line_groups = [sorted(bases, key=lambda b: b.x)]
    else:
        line_groups = [l for l in _group_lines(bases, p) if len(l) >= p.min_line_glyphs]
    per_line_marks, orphans = _assign_marks(line_groups, marks, p)

    lines: List[Line] = []
    for li, group in enumerate(line_groups):
        script, asp = _line_script(group, p)
        ratio = _adaptive_word_gap_ratio(group, p) if p.adaptive_word_gap else None
        word_groups = [group] if single_word else _split_words(group, p, ratio)
        words = []
        for wg in word_groups:
            glyphs = []
            for b in wg:
                mask = (labels[b.y:b.y + b.h, b.x:b.x + b.w] == b.idx)
                g = Glyph(bbox=b.bbox,
                          image=normalize_glyph(mask, GLYPH_SIZE, p.glyph_margin))
                for m in per_line_marks[li].get(b.idx, []):
                    mmask = (labels[m.y:m.y + m.h, m.x:m.x + m.w] == m.idx)
                    g.marks.append(Mark(
                        bbox=m.bbox,
                        position="above" if m.cy < b.cy else "below",
                        shape_hint=_shape_hint(m),
                        image=normalize_glyph(mmask, GLYPH_SIZE, p.glyph_margin)))
                glyphs.append(g)
            words.append(Word(glyphs))
        lines.append(Line(words, script, asp))

    return Result(bw, lines, [m.bbox for m in orphans], len(noise))


def process_image(img, mode: str = "clean", p: Optional[Params] = None,
                  roi: Optional[Tuple[int, int, int, int]] = None,
                  single_word: bool = False, do_deskew: Optional[bool] = None,
                  merge_stacked: bool = False) -> Result:
    if p is None:
        # These two Baybayin sentence references are densely handwritten and
        # have larger vertical drift than the word samples. A wider line band
        # keeps each physical row together; clean word images retain defaults.
        p = (Params(min_line_glyphs=3, line_split_ratio=1.2,
                    adaptive_word_gap=True) if mode == "photo" else Params())
    if isinstance(img, str):
        img = load_image(img)
    if roi:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]

    if mode == "photo":
        bw = binarize_photo(img, p)
    elif mode == "clean":
        bw = binarize_clean(img, p)
    else:
        raise ValueError("mode must be 'clean' or 'photo'")

    if do_deskew is None:
        do_deskew = (mode == "clean")
    if do_deskew:
        bw, _ = deskew(bw, p.deskew_max_angle)
    if mode == "clean":
        bw, _ = crop_to_content(bw, p.crop_pad)
    return segment(bw, p, single_word=single_word, merge_stacked=merge_stacked)


# --------------------------------------------------------------------------- #
# Visualization & Saving
# --------------------------------------------------------------------------- #
def draw_overlay(res: Result) -> np.ndarray:
    vis = cv2.cvtColor(res.binary, cv2.COLOR_GRAY2BGR)
    vis = 255 - vis
    for li, line in enumerate(res.lines):
        lx, ly, lw, lh = line.bbox
        colour_line = (0, 140, 255) if line.script == "baybayin" else (160, 160, 160)
        cv2.putText(vis, f"L{li}:{line.script[:3]}", (max(0, lx - 2), max(12, ly - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_line, 1, cv2.LINE_AA)
        for word in line.words:
            x, y, w, h = word.bbox
            cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 200, 255), 1)
            for g in word.glyphs:
                gx, gy, gw, gh = g.bbox
                cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 170, 0), 1)
                for m in g.marks:
                    mx, my, mw, mh = m.bbox
                    col = (0, 0, 220) if m.position == "above" else (220, 0, 0)
                    cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), col, 1)
    for (x, y, w, h) in res.orphan_marks:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 0, 200), 1)
    return vis


def save_outputs(res: Result, out_dir: str, stem: str = "result",
                 baybayin_only: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_binary.png"), 255 - res.binary)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), draw_overlay(res))
    with open(os.path.join(out_dir, f"{stem}_segments.json"), "w") as f:
        json.dump(res.to_dict(), f, indent=2)
    for li, line in enumerate(res.lines):
        if baybayin_only and line.script != "baybayin":
            continue
        for wi, word in enumerate(line.words):
            for gi, g in enumerate(word.glyphs):
                base = os.path.join(out_dir, f"{stem}_L{li}_W{wi}_G{gi}")
                cv2.imwrite(base + ".png", g.image)
                for mi, m in enumerate(g.marks):
                    cv2.imwrite(f"{base}_mark{mi}_{m.position}_{m.shape_hint}.png", m.image)


# --------------------------------------------------------------------------- #
# Batch Pipeline Processing
# --------------------------------------------------------------------------- #
def process_full_dataset(dataset_root: str, out_root: str) -> None:
    """Automated batch preprocessing across symbols, words, and sentences."""
    print("==================================================")
    print("Starting Full Baybayin Dataset Preprocessing")
    print("==================================================")

    # 1. Process Baybayin Symbol Dataset (Single Character Images)
    symbols_dir = os.path.join(dataset_root, "Baybayin Symbol Dataset")
    if os.path.exists(symbols_dir):
        print("\n[1/3] Preprocessing Symbol Dataset...")
        npz_out = os.path.join(out_root, "symbols", "symbols_dataset.npz")
        os.makedirs(os.path.dirname(npz_out), exist_ok=True)

        X, y, groups = [], [], []
        for label in sorted(os.listdir(symbols_dir)):
            class_path = os.path.join(symbols_dir, label)
            if os.path.isdir(class_path):
                class_out_dir = os.path.join(out_root, "symbols", "images", label)
                os.makedirs(class_out_dir, exist_ok=True)
                for fn in sorted(os.listdir(class_path)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        img_path = os.path.join(class_path, fn)
                        try:
                            norm_img = preprocess_character_image(img_path)
                            X.append(norm_img)
                            y.append(label)
                            # The supplied names encode a source image/page suffix
                            # (for example, a_00001_file001.jpg). Keep it so the
                            # classifier test split can prevent source leakage.
                            groups.append(os.path.splitext(fn)[0].rsplit("_", 1)[-1])
                            cv2.imwrite(os.path.join(class_out_dir, fn), norm_img)
                        except Exception as e:
                            print(f"  Error processing symbol {img_path}: {e}")
        np.savez_compressed(npz_out, X=np.array(X, np.uint8), y=np.array(y),
                            groups=np.array(groups))
        print(f"  Processed {len(X)} symbol images -> Saved NPZ to {npz_out}")
    else:
        print(f"\n[1/3] Skipping Symbol Dataset (folder not found: {symbols_dir})")

    # 2. Process Baybayin Word Images
    words_dir = os.path.join(dataset_root, "Baybayin Word Images")
    if os.path.exists(words_dir):
        print("\n[2/3] Preprocessing Word Images...")
        words_out = os.path.join(out_root, "words")
        count = 0
        for fn in sorted(os.listdir(words_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_path = os.path.join(words_dir, fn)
                stem = os.path.splitext(fn)[0]
                try:
                    res = process_image(img_path, mode="clean", single_word=True)
                    item_out = os.path.join(words_out, stem)
                    save_outputs(res, item_out, stem=stem)
                    count += 1
                except Exception as e:
                    print(f"  Error processing word image {fn}: {e}")
        print(f"  Successfully processed {count} word images -> {words_out}")
    else:
        print(f"\n[2/3] Skipping Word Images (folder not found: {words_dir})")

    # 3. Process Baybayin Sentence References Image
    sentences_dir = os.path.join(dataset_root, "Baybayin Sentence References Image")
    if os.path.exists(sentences_dir):
        print("\n[3/3] Preprocessing Sentence Reference Images...")
        sentences_out = os.path.join(out_root, "sentences")
        count = 0
        for fn in sorted(os.listdir(sentences_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img_path = os.path.join(sentences_dir, fn)
                stem = os.path.splitext(fn)[0]
                try:
                    res = process_image(img_path, mode="photo")
                    item_out = os.path.join(sentences_out, stem)
                    save_outputs(res, item_out, stem=stem)
                    count += 1
                except Exception as e:
                    print(f"  Error processing sentence image {fn}: {e}")
        print(f"  Successfully processed {count} sentence images -> {sentences_out}")
    else:
        print(f"\n[3/3] Skipping Sentence Images (folder not found: {sentences_dir})")

    print("\n==================================================")
    print(f"Dataset Preprocessing Complete! Outputs stored in: {out_root}")
    print("==================================================")


# --------------------------------------------------------------------------- #
# CLI Entry Point
# --------------------------------------------------------------------------- #
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["char", "word", "sentence", "dataset"],
                    help="Target input type, or 'dataset' to process the full dataset directory.")
    ap.add_argument("path", help="Path to input image file or root dataset directory.")
    ap.add_argument("--mode", choices=["clean", "photo"], default=None,
                    help="default: clean for char/word, photo for sentence")
    ap.add_argument("--roi", default=None, help="x,y,w,h crop before processing")
    ap.add_argument("--out", default="out", help="Output directory path")
    ap.add_argument("--keep-latin", action="store_true",
                    help="also save glyph crops from lines classed as Latin")
    args = ap.parse_args()

    # Full Dataset Batch Mode
    if args.kind == "dataset":
        process_full_dataset(args.path, args.out)
        return

    # Single Image Mode
    stem = os.path.splitext(os.path.basename(args.path))[0]

    if args.kind == "char":
        g = preprocess_character_image(args.path, Params())
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, f"{stem}_28.png"), g)
        print("saved", os.path.join(args.out, f"{stem}_28.png"))
        return

    mode = args.mode or ("clean" if args.kind == "word" else "photo")
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    res = process_image(args.path, mode=mode, p=None, roi=roi,
                        single_word=(args.kind == "word"))
    save_outputs(res, args.out, stem, baybayin_only=not args.keep_latin)

    n_lines = len(res.lines)
    n_words = sum(len(l.words) for l in res.lines)
    n_glyphs = sum(len(w.glyphs) for l in res.lines for w in l.words)
    n_marks = sum(len(g.marks) for l in res.lines for w in l.words for g in w.glyphs)
    print(f"{n_lines} lines, {n_words} words, {n_glyphs} base glyphs, {n_marks} marks, "
          f"{len(res.orphan_marks)} orphan marks, {res.noise_count} noise components")


if __name__ == "__main__":
    _main()
