# Baybayin OCR & Transliteration Translator — Preprocessing Pipeline

This repository contains the preprocessing, binarization, and connected-component segmentation pipeline for the Baybayin OCR Transliteration Translator.

---

## 1. Installation Process

### Prerequisites

- **Python:** 3.10, 3.11, or 3.14
- **Core Libraries:** `opencv-python`, `numpy`

### Steps

1. **Navigate to your project root directory:**

   ```powershell
   cd "C:\Users\Julianna\SLU\4th Year\Special Topics\Project\Baybayin-Tagalog-English-Translation-through-OCR"
   ```

2. **Install required dependencies:**

   ```powershell
   python -m pip install opencv-python numpy
   ```

3. **Verify installation:**

   ```powershell
   python -c "import cv2, numpy; print('OpenCV version:', cv2.__version__)"
   ```

---

## 2. Dataset Folder Structure

Ensure your dataset directory is structured as follows before running batch processing:

```text
dataset/
├── Baybayin Symbol Dataset/
│   ├── a/
│   ├── ba/
│   ├── dara/
│   └── ...
├── Baybayin Word Images/
│   ├── 1 Gwapo.PNG
│   ├── 10 Ako.PNG
│   └── ...
└── Baybayin Sentence References Image/
    └── ...
```

**Folder descriptions:**

- `Baybayin Symbol Dataset/` — Contains isolated Baybayin character images organized into character-class folders.
- `Baybayin Word Images/` — Contains images of individual Baybayin words.
- `Baybayin Sentence References Image/` — Contains full-page or sentence-level reference images.

---

## 3. Running Preprocessing Execution Commands

The script `baybayin_preprocess.py` supports different execution modes depending on whether you want to process the entire dataset or individual images.

### Option A: Process the Full Dataset (Recommended)

Run end-to-end preprocessing across all symbols, words, and sentences:

```powershell
python baybayin_preprocess.py dataset dataset --out out/
```

This command processes the dataset and saves the generated artifacts in the `out/` directory.

### Option B: Single Image Processing Modes

#### 1. Isolated Character Symbol Image

Process an individual Baybayin character image:

```powershell
python baybayin_preprocess.py char "dataset/Baybayin Symbol Dataset/a/sample.png" --out out/
```

#### 2. Single Word Image (`clean` mode)

Process and segment a single word image:

```powershell
python baybayin_preprocess.py word "dataset/Baybayin Word Images/1 Gwapo.PNG" --out out/
```

#### 3. Full Page / Sentence Photo (`photo` mode)

Process a full-page or sentence image using photo preprocessing:

```powershell
python baybayin_preprocess.py sentence "dataset/Baybayin Sentence References Image/sample.jpg" --mode photo --out out/
```

**Note:** Replace `sample.png` and `sample.jpg` with the actual filenames in your dataset.

---

## 4. Output Structure

When executing batch mode, all processed artifacts are organized in the designated output directory, `out/`.

```text
out/
├── symbols/
│   ├── symbols_dataset.npz
│   └── images/
│       ├── a/
│       ├── ba/
│       └── ...
├── words/
│   ├── 1 Gwapo/
│   │   ├── 1 Gwapo_binary.png
│   │   ├── 1 Gwapo_overlay.png
│   │   ├── 1 Gwapo_segments.json
│   │   ├── 1 Gwapo_L0_W0_G0.png
│   │   └── 1 Gwapo_L0_W0_G0_mark0_above_dot.png
│   └── ...
└── sentences/
    └── ...
```

### Output Descriptions

| Output | Description |
|---|---|
| `symbols_dataset.npz` | Compressed NumPy file containing normalized 28×28 arrays and their corresponding labels. |
| `symbols/images/` | Stores normalized binary glyph images grouped by character class. |
| `*_binary.png` | Binarized image with ink represented as 255 and background as 0. |
| `*_overlay.png` | Debug visualization showing detected bounding boxes. |
| `*_segments.json` | Contains bounding-box information and structural metadata for segmented components. |
| `*_L0_W0_G0.png` | Example of a segmented 28×28 base-character glyph. |
| `*_mark0_above_dot.png` | Example of an extracted Kudlit or Virama mark. |
| `sentences/` | Contains sentence-level segmented outputs, including illumination-normalized results when supported by the pipeline. |

**Note:** Actual filenames and output contents may vary depending on the input image, detected components, and preprocessing configuration.

---

## 5. Python API Usage

The preprocessing functions can also be imported directly into model training or inference scripts without executing the command-line interface.

### Example

```python
from baybayin_preprocess import (
    process_image,
    preprocess_character_image,
)

# 1. Preprocess an isolated character image.
# Returns a normalized 28x28 NumPy array.
glyph_28x28 = preprocess_character_image("path/to/char.png")

print("Character image shape:", glyph_28x28.shape)

# 2. Process a full word image.
# Returns a Result object containing segmented glyphs and marks.
result = process_image(
    "path/to/word.png",
    mode="clean",
    single_word=True,
)

# 3. Access lines, words, glyphs, and marks.
for line in result.lines:
    for word in line.words:
        for glyph in word.glyphs:
            print(f"Base glyph shape: {glyph.image.shape}")

            for mark in glyph.marks:
                print(
                    f"  Mark position: {mark.position}, "
                    f"shape hint: {mark.shape_hint}"
                )
```

### API Function Descriptions

| Function | Description |
|---|---|
| `preprocess_character_image(image_path)` | Preprocesses an isolated character image into a normalized 28×28 NumPy array. |
| `process_image(image_path, mode="clean", single_word=True)` | Processes a word image and returns a result containing segmented glyphs and associated marks. |

### Accessing the Processing Results

The `process_image()` result provides access to the detected image structure:

- `result.lines` — Detected lines in the processed image.
- `line.words` — Words detected within each line.
- `word.glyphs` — Base-character glyphs detected within each word.
- `glyph.image` — Image array for an individual base glyph.
- `glyph.marks` — Associated marks, such as Kudlit or Virama.
- `mark.position` — Position information for a detected mark.
- `mark.shape_hint` — Shape classification or hint associated with the mark.

**Important:** The available fields and return types depend on the implementation of `baybayin_preprocess.py`. Confirm the function signatures and result-object definitions in the source code if they differ from this example.

---

## 6. Preprocessing Pipeline Summary

The preprocessing pipeline prepares Baybayin images for OCR and transliteration by organizing the input data, normalizing character images, binarizing word or sentence images, and segmenting connected components into individual glyphs and associated marks.

The resulting artifacts can be used for:

- Baybayin character recognition.
- Glyph classification and model training.
- Word-level character segmentation.
- Kudlit and Virama mark analysis.
- OCR inference and downstream transliteration.

## 7. Train, run, and evaluate the OCR model

Install the dependencies used by preprocessing, training, and evaluation:

```powershell
python -m pip install numpy opencv-python pandas scikit-learn matplotlib torch
```

From the project root, preprocess the images and train the character classifier:

```powershell
python baybayin_preprocess.py dataset dataset --out out/
python train_models.py --npz out/symbols/symbols_dataset.npz --out models/ --seed 42 --skip-svm
```

This reproduces the saved CNN: up to 40 epochs with early stopping after 6 epochs
without validation improvement (the saved run stopped at epoch 29, best at epoch 23).
For a quick five-epoch check, add `--epochs 5 --patience 3`. Omit `--skip-svm` to also
train the SVM baseline in the same folder.

To reproduce the separate SVM baseline used in the findings tables, train it into its
own output folder so the CNN metrics remain intact:

```powershell
python train_models.py --npz out/symbols/symbols_dataset.npz --out models_svm/ --seed 42 --skip-cnn --svm-max-train 2500
```

Training saves the CNN weights, a stratified and source-group-disjoint train/validation/test
split, class metrics, and confusion matrices in `models/`. Source groups come from the
symbol filenames, preventing crops from one source image/page from appearing on both
sides of the evaluation. The split is approximately 64/16/20 because groups stay intact.
The reported character accuracy and macro-F1 are measured on the held-out test split.
They measure isolated glyph classification, not word segmentation or translation accuracy.

Run deterministic checks of transliteration candidate generation and the language
model using ideal (perfect) glyph observations:

```powershell
python translate_pipeline.py selftest --data dataset --n 2000 --seed 0
```

This oracle check isolates candidate search and ranking. It does not measure the CNN or
image segmentation, and its generated spellings are only approximations of traditional
Baybayin conventions. The pipeline defaults to `--beta 0.25` for the language-model
weight; pass `--beta 1` to compare with the older stronger language-model weighting.

Evaluate end-to-end word-image recognition on the labeled filenames in
`dataset/Baybayin Word Images/`:

```powershell
python translate_pipeline.py words --data dataset --cnn models/cnn.pt --no-nllb --out models/word_eval.json
```

The report gives top-1 exact match,
top-5 identification, character error rate (CER), and failed-image count. This is the
full-dataset diagnostic on isolated word images. Labels of the evaluated images are
excluded from the filename-derived lexicon words, but they may still appear in the
Tagalog dictionary files, and the language-model weight was tuned on this same set, so
it is not a fully independent accuracy estimate. It does not measure full-sentence or
English translation quality. The current run covers 1,001 images and reports 66.63%
top-1, 71.13% top-5, and mean CER 0.1717. Use a separate,
reference-grouped holdout with held-out labels excluded from the lexicon for a fair
generalization estimate. `top1_by_beta_diagnostic` compares language-model weights on
the same dataset and is for tuning only.

To print another deterministic 50-image sample with the top Tagalog candidate and its
dictionary English gloss on each line, use a different seed:

```powershell
python translate_pipeline.py words --data dataset --cnn models/cnn.pt --no-nllb --english --limit 50 --seed 1 --out models/word_eval_english_50.json
```

The English gloss may contain multiple dictionary senses or `[word]` when no gloss is
available. This output is for review; it is not an English-translation accuracy score.

The main model used by the run commands is `models/cnn.pt`. Experimental adaptation
checkpoints and scratch evaluation outputs are not part of the main model workflow.

The English dictionary retains distinct senses for words with multiple parts of speech.
For example, `laban` now returns `fight, match, contest; against, versus` instead of
discarding the noun sense. An isolated word has no sentence context to choose a single
meaning.

Translate one word image or sentence photo:

```powershell
python translate_pipeline.py image "dataset/Baybayin Word Images/1 Gwapo.PNG" --data dataset --cnn models/cnn.pt --word --no-nllb
python translate_pipeline.py image path\to\sentence.jpg --data dataset --cnn models/cnn.pt --mode photo
```

For a compact terminal view, add `--pretty` to `image`. To run every numbered
Baybayin/Tagalog guide pair (for example, `2.jpg` with `2.1.jpg`) and print the
approximate Baybayin reading, model Tagalog output, and English output together:

```powershell
python translate_pipeline.py pairs --data dataset --cnn models/cnn.pt --no-nllb --out models/sentence_pair_results.json
```

The paired files are image guides, so they are not automatically scored as text. Fill
`sentence_pair_references.csv` with checked transcriptions in its `tagalog_reference`
column and human English translations in `english_reference`; then pass
`--references sentence_pair_references.csv` to `pairs` to print Tagalog and English CER/WER.
Keep `image` values as the Baybayin-side filenames (`2.jpg`, `8.jpg`).

Print the saved CNN, SVM, word-image, oracle, sample translation, and error tables in
the terminal (and save a Markdown copy) with:

```powershell
python report_findings.py --cnn-dir models --svm-dir models_svm --word-eval models/word_eval.json --selftest models/selftest.json --sample models/sample_word.json --out models/findings.md
```

The report is only as current as those saved evaluation files. Re-run the matching
training/evaluation command after changing the model or dataset.

For fair accuracy comparisons, keep the model, seed, preprocessing settings, and
evaluation image set fixed. The word-image filenames serve as references, so check that
each filename matches the Baybayin content before treating the aggregate score as
ground truth. NLLB sentence output is a separate, generative translation stage; evaluate
it against human-reviewed English references rather than the OCR word metrics.

---
