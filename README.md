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

---