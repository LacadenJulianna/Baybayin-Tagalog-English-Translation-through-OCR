"""
train_models.py
===============
Train and evaluate the Baybayin base-character classifiers (paper Section 3.5):

  * CNN  (PyTorch)      - 17-way softmax, cross-entropy, augmentation, early stopping
  * SVM  (scikit-learn) - RBF baseline on the same data and the same split

Input
-----
  out/symbols/symbols_dataset.npz   produced by baybayin_preprocess.py
      X : (N, 28, 28) uint8, ink = 255, background = 0
      y : (N,) class names (a, ba, dara, ei, ga, ha, ka, la, ma, na, nga, ou, pa, sa, ta, wa, ya)

Output (in --out, default models/)
-------
  classes.json            index -> class name (needed at inference time)
  split.npz               train / val / test indices (so every model sees the same split)
  cnn.pt                  CNN weights + class list
  svm.joblib              fitted scaler + SVM
  metrics.json            accuracy, macro-F1, per-class precision/recall/F1 for both models
  history.json            CNN loss / accuracy per epoch
  confusion_cnn.png       confusion matrices (test split)
  confusion_svm.png

Usage
-----
  python train_models.py --npz out/symbols/symbols_dataset.npz --out models/
  python train_models.py --npz ... --skip-svm            # CNN only
  python train_models.py --npz ... --svm-max-train 0     # SVM on the full train split (slow)

Inference helpers (for the pipeline):
  from train_models import load_cnn, predict_glyphs
  model, classes = load_cnn("models/cnn.pt")
  labels, probs = predict_glyphs(model, classes, glyph_images_uint8)   # (n, 28, 28)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(npz_path: str, seed: int, val_frac: float = 0.15, test_frac: float = 0.15):
    """Stratified train/val/test split (70/15/15 by default)."""
    d = np.load(npz_path, allow_pickle=False)
    X = d["X"].astype(np.uint8)
    y_names = d["y"].astype(str)
    classes = sorted(set(y_names.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[c] for c in y_names], dtype=np.int64)

    idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(idx, test_size=test_frac, stratify=y,
                                           random_state=seed)
    rel_val = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(idx_trval, test_size=rel_val,
                                          stratify=y[idx_trval], random_state=seed)
    return X, y, classes, idx_train, idx_val, idx_test


def save_confusion(cm: np.ndarray, classes: List[str], title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(y_true, y_pred, classes) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True,
                                           zero_division=0),
    }


# --------------------------------------------------------------------------- #
# SVM baseline
# --------------------------------------------------------------------------- #
def train_svm(X, y, classes, idx_train, idx_val, idx_test, out_dir, seed, max_train):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def flat(ix):
        return X[ix].reshape(len(ix), -1).astype(np.float32) / 255.0

    rng = np.random.RandomState(seed)

    def subsample(ix, n):
        if n <= 0 or n >= len(ix):
            return ix
        sel, _ = train_test_split(np.arange(len(ix)), train_size=n,
                                  stratify=y[ix], random_state=seed)
        return ix[sel]

    # small grid on a subset of train, scored on the validation split
    tune_ix = subsample(idx_train, 4000)
    val_ix = subsample(idx_val, 2000)
    best, best_cfg = -1.0, None
    for C in (1, 10, 30):
        for gamma in ("scale", 0.002):
            m = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced"))
            m.fit(flat(tune_ix), y[tune_ix])
            acc = accuracy_score(y[val_ix], m.predict(flat(val_ix)))
            print(f"  [svm] C={C:<3} gamma={gamma!s:<6} val acc={acc:.4f}")
            if acc > best:
                best, best_cfg = acc, (C, gamma)
    print(f"  [svm] best C={best_cfg[0]} gamma={best_cfg[1]}")

    fit_ix = subsample(idx_train, max_train)
    print(f"  [svm] final fit on {len(fit_ix)} training images ...")
    t0 = time.time()
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=best_cfg[0], gamma=best_cfg[1],
                              class_weight="balanced"))
    model.fit(flat(fit_ix), y[fit_ix])
    print(f"  [svm] fit took {time.time() - t0:.0f}s")

    pred = model.predict(flat(idx_test))
    res = summarize(y[idx_test], pred, classes)
    res["best_params"] = {"C": best_cfg[0], "gamma": str(best_cfg[1])}
    res["train_images_used"] = int(len(fit_ix))
    joblib.dump({"model": model, "classes": classes}, os.path.join(out_dir, "svm.joblib"))
    save_confusion(confusion_matrix(y[idx_test], pred, labels=range(len(classes))),
                   classes, f"SVM (test acc {res['accuracy']:.3f})",
                   os.path.join(out_dir, "confusion_svm.png"))
    print(f"  [svm] TEST accuracy={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_cnn(n_classes: int):
    torch, nn, F = _torch()

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(block(1, 32), block(32, 64))   # 28 -> 14 -> 7
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.4),
                nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    return SmallCNN()


def augment(x):
    """Batch augmentation on (B,1,28,28) floats in [0,1]:
    small rotation, scaling, translation, and stroke thickness (dilate / erode)."""
    torch, _, F = _torch()
    B = x.size(0)
    ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(10)
    sc = 1.0 + (torch.rand(B, device=x.device) * 2 - 1) * 0.10
    tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.16   # normalised coords, ~8% of width
    ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.16
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                         torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.size()), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    r = torch.rand(B, 1, 1, 1, device=x.device)
    dilated = F.max_pool2d(x, 3, 1, 1)
    eroded = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < 0.15, dilated, torch.where(r > 0.90, eroded, x))
    return x.clamp(0, 1)


def _to_tensor(X_uint8, device):
    torch, _, _ = _torch()
    return torch.from_numpy(X_uint8.astype(np.float32) / 255.0).unsqueeze(1).to(device)


def _predict(model, X_uint8, device, batch=512):
    torch, _, _ = _torch()
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_uint8), batch):
            logits = model(_to_tensor(X_uint8[i:i + batch], device))
            outs.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 0))


def train_cnn(X, y, classes, idx_train, idx_val, idx_test, out_dir, seed,
              epochs, batch, lr, patience):
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [cnn] device={device}")

    model = build_cnn(len(classes)).to(device)
    counts = np.bincount(y[idx_train], minlength=len(classes)).astype(np.float32)
    weights = counts.sum() / (len(classes) * np.maximum(counts, 1))     # inverse frequency
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xtr = _to_tensor(X[idx_train], device)
    ytr = torch.from_numpy(y[idx_train]).to(device)
    best_acc, best_state, bad = -1.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        run_loss, n_seen = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb, yb = augment(Xtr[b]), ytr[b]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(b)
            n_seen += len(b)

        val_prob = _predict(model, X[idx_val], device)
        val_acc = float(accuracy_score(y[idx_val], val_prob.argmax(1)))
        history.append({"epoch": ep, "train_loss": run_loss / n_seen, "val_acc": val_acc})
        print(f"  [cnn] epoch {ep:>2}  loss={run_loss / n_seen:.4f}  val acc={val_acc:.4f}")

        if val_acc > best_acc:       # early stopping on validation accuracy
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  [cnn] early stop (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    prob = _predict(model, X[idx_test], device)
    pred = prob.argmax(1)
    res = summarize(y[idx_test], pred, classes)
    res["best_val_acc"] = best_acc
    res["epochs_run"] = len(history)
    torch.save({"state_dict": best_state, "classes": classes}, os.path.join(out_dir, "cnn.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_confusion(confusion_matrix(y[idx_test], pred, labels=range(len(classes))),
                   classes, f"CNN (test acc {res['accuracy']:.3f})",
                   os.path.join(out_dir, "confusion_cnn.png"))
    print(f"  [cnn] TEST accuracy={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    return res


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def load_cnn(path: str, device: str = "cpu"):
    torch, _, _ = _torch()
    ckpt = torch.load(path, map_location=device)
    classes = ckpt["classes"]
    model = build_cnn(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), classes


def predict_glyphs(model, classes: List[str], images_uint8: np.ndarray,
                   device: str = "cpu") -> Tuple[List[str], np.ndarray]:
    """images_uint8: (n, 28, 28), ink = 255 (same normalisation as training).
    Returns predicted class names and the full probability matrix (n, n_classes);
    keep the probabilities - candidate generation can use the runner-up classes."""
    prob = _predict(model, np.asarray(images_uint8, np.uint8), device)
    return [classes[i] for i in prob.argmax(1)], prob


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="out/symbols/symbols_dataset.npz")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--svm-max-train", type=int, default=12000,
                    help="max training images for the final SVM fit (0 = all; slow)")
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y, classes, i_tr, i_va, i_te = load_split(args.npz, args.seed)
    print(f"{len(y)} images, {len(classes)} classes; "
          f"train/val/test = {len(i_tr)}/{len(i_va)}/{len(i_te)}")
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)
    np.savez(os.path.join(args.out, "split.npz"), train=i_tr, val=i_va, test=i_te)

    metrics = {}
    if not args.skip_svm:
        print("SVM baseline")
        metrics["svm"] = train_svm(X, y, classes, i_tr, i_va, i_te, args.out,
                                   args.seed, args.svm_max_train)
    if not args.skip_cnn:
        print("CNN")
        metrics["cnn"] = train_cnn(X, y, classes, i_tr, i_va, i_te, args.out, args.seed,
                                   args.epochs, args.batch, args.lr, args.patience)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved results to {args.out}/")


if __name__ == "__main__":
    main()
