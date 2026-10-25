"""Print paper-ready findings tables from the saved evaluation JSON files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnn-dir", default="models")
    ap.add_argument("--svm-dir", default="models_svm")
    ap.add_argument("--cnn-checkpoint", default=None,
                    help="recompute held-out glyph metrics for this CNN checkpoint")
    ap.add_argument("--symbols-npz", default="out/symbols/symbols_dataset.npz")
    ap.add_argument("--word-eval", default="models/word_eval.json")
    ap.add_argument("--selftest", default="models/selftest.json")
    ap.add_argument("--sample", default="models/sample_word.json")
    ap.add_argument("--sentence-pairs", default="models/sentence_pair_results.json")
    ap.add_argument("--out", default=None, help="also save the tables to a Markdown file")
    args = ap.parse_args()

    cnn_metrics = read_json(os.path.join(args.cnn_dir, "metrics.json"))
    cnn_split = np.load(os.path.join(args.cnn_dir, "split.npz"))
    test_n = len(cnn_split["test"])
    train_n = len(cnn_split["train"])
    cnn = cnn_metrics.get("cnn", {})
    if args.cnn_checkpoint:
        from sklearn.metrics import accuracy_score, f1_score
        from train_models import load_cnn, load_split, predict_glyphs
        X, y, _classes, _i_train, _i_val, i_test = load_split(args.symbols_npz, 42)
        model, classes = load_cnn(args.cnn_checkpoint)
        predicted, _prob = predict_glyphs(model, classes, X[i_test])
        class_index = {name: i for i, name in enumerate(classes)}
        y_pred = np.asarray([class_index[name] for name in predicted])
        y_true = np.asarray(y[i_test])
        cnn = {"accuracy": accuracy_score(y_true, y_pred),
               "macro_f1": f1_score(y_true, y_pred, average="macro")}
        test_n = len(i_test)
    svm_path = os.path.join(args.svm_dir, "metrics.json")
    svm_metrics = read_json(svm_path).get("svm", {}) if os.path.exists(svm_path) else {}
    word = read_json(args.word_eval)["summary"]
    oracle = read_json(args.selftest)
    sample_json = read_json(args.sample)
    sentence_pairs = read_json(args.sentence_pairs)
    scored_sentence_pairs = [p for p in sentence_pairs if p.get("tagalog_reference")]
    if scored_sentence_pairs:
        sentence_note = (
            f"\n**Sentence-image evaluation:** Tagalog guide transcriptions were checked "
            f"for {len(scored_sentence_pairs)} numbered pairs. Mean Tagalog CER/WER: "
            f"{np.mean([p['tagalog_cer'] for p in scored_sentence_pairs]):.3f}/"
            f"{np.mean([p['tagalog_wer'] for p in scored_sentence_pairs]):.3f}. "
            f"English sentence accuracy is unscored for "
            f"{sum(not p.get('english_reference') for p in sentence_pairs)} pairs because "
            "checked English references are unavailable."
        )
    else:
        sentence_note = (
            "\n**Sentence-image evaluation:** no checked Tagalog guide transcription is "
            "available, so sentence accuracy is not scored."
        )
    line = (sample_json.get("lines") or [{}])[0]
    english = line.get("stage4_english", {})

    sections = [
        "# Findings",
        "\n**Table 1. CNN and SVM Character Classification Performance**\n",
        markdown_table(
            ["Evaluation metric", "CNN", "SVM"],
            [["Classification accuracy", pct(cnn["accuracy"]),
              pct(svm_metrics["accuracy"]) if svm_metrics else "Not run"],
             ["Macro-averaged F1-score", f"{cnn['macro_f1']:.4f}",
              f"{svm_metrics['macro_f1']:.4f}" if svm_metrics else "Not run"],
             ["Number of evaluation samples", str(test_n),
              str(svm_metrics.get("test_samples", test_n)) if svm_metrics else "Not run"]]),
        (f"\nCNN training used {train_n} glyphs; SVM training used "
         f"{svm_metrics.get('train_images_used', '—')} glyphs. Both used the same "
         "source-group-disjoint test split."),
        "\n**Table 2. Baybayin Word Identification Performance**\n",
        markdown_table(
            ["Evaluation metric", "Result"],
            [["Number of images tested", word["images"]],
             ["Failed images", word["failed_images"]],
             ["Top-1 exact-match accuracy", pct(word["exact_match_top1"])],
             ["Top-5 word identification accuracy", pct(word["top5_word_identification"])],
             ["Mean CER against filename references", f"{word['mean_cer_vs_filename']:.4f}"]]),
        "\n**Table 3. Sample Baybayin Word Recognition and English Translation**\n",
        markdown_table(
            ["Processing stage", "System output"],
            [["Input image", sample_json.get("image", "")],
             ["Baybayin reading (approx.)", line.get("baybayin_reading", "")],
             ["Recognized Tagalog word", line.get("tagalog", "")],
             ["English translation", english.get("english", "")],
             ["Translation method", english.get("method", "")],
             ["Unverified words", ", ".join(line.get("unverified_words", [])) or "None"],
             ["Processing time", f"{line.get('seconds', 0):.3f} s"]]),
        "\n**Table 4. Tagalog Word Candidate Generation Results (oracle test)**\n",
        markdown_table(
            ["Evaluation metric", "Result"],
            [["Words tested", oracle["words_tested"]],
             ["Candidate recall", pct(oracle["candidate_recall"])],
             ["Top-1 accuracy", pct(oracle["top1"])],
             ["Top-5 accuracy", pct(oracle["top5"])],
             ["Mean number of candidates", f"{oracle['mean_candidates']:.2f}"],
             ["Median number of candidates", f"{oracle['median_candidates']:.1f}"],
             ["Processing time", f"{oracle['seconds']:.1f} s"],
             ["Tagalog words in the word list", oracle["lexicon_words"]],
             ["Number of prefixes", oracle["prefixes"]],
             ["Language-model tokens", oracle["lm_tokens"]],
             ["Bigrams", oracle["bigrams"]]]),
        "\n**Table 5. Sample Errors in Tagalog Word Candidate Generation**\n",
        markdown_table(
            ["Reference word", "First predicted candidate", "Second predicted candidate",
             "Third predicted candidate"],
            [[miss["reference"], *(miss["predicted"] + ["—"] * 3)[:3]]
             for miss in oracle.get("sample_misses", [])]),
        sentence_note,
    ]
    result = "\n".join(sections) + "\n"
    print(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
