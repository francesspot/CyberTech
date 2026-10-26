[Polski](README.md) | **English**

# Waste Classification with AI

A deep-learning model that sorts waste into **6 categories** based on a photo: `cardboard`, `glass`, `metal`, `paper`, `plastic` and `trash`. Built with TensorFlow/Keras using **transfer learning** on **EfficientNetB0** (pre-trained on ImageNet).

## Results

| | Accuracy |
|---|---|
| Validation, after phase 1 | ~0.89 |
| Validation, after fine-tuning | ~0.91 |
| **Test set** | **0.90** |

The notebook also plots the training curves and shows the classification report and confusion matrix for the test set. The weakest class is `trash`, which has the fewest examples.

## How it works

1. **Loading data** — images (224 × 224) from `dataset/train`, `dataset/valid` and `dataset/test`, in batches of 32.
2. **Faster loading** — `cache()` + `prefetch()`, so every epoch after the first runs roughly twice as fast.
3. **Augmentation** — random flips, rotation, zoom, contrast, brightness and shifts, so the model sees more varied images.
4. **Phase 1** — EfficientNetB0 with frozen weights as the feature extractor, plus a new classification head (pooling, a 128-unit dense layer, dropout); trained for up to 15 epochs with early stopping.
5. **Phase 2 — fine-tuning** — the last 30 layers of EfficientNetB0 are unfrozen and trained with a 10× lower learning rate (up to 20 epochs). The best model is saved to `best_model.keras`.
6. **Evaluation** on the test set.

## Dataset

The images are **not** included in the repository (`dataset/` is in `.gitignore`). Put them in this layout — one subfolder per class:

```
dataset/
├── train/   (1767 images)
├── valid/   (504 images)
└── test/    (253 images)
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Setting up the environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Then open `trash_sorting_model.ipynb` in Jupyter or VS Code and run all cells. Training is much faster on a GPU.
