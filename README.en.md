🇵🇱 [Polski](README.md) | 🇬🇧 **English**

# ♻️ Waste Classification from Photos

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)

> 🎓 Recruitment task for the **CyberTech** student research group (Wrocław University of Science and Technology)

A model that classifies waste from a photo into one of 6 classes: cardboard, glass, metal, paper, plastic and general trash.

## 🎯 Result

**90% accuracy on the test set**

## ✨ What I did

- **Transfer learning** on **EfficientNetB0** (pre-trained on ImageNet) with my own classification head
- **Data augmentation** — random flips, rotation, zoom, contrast, brightness and shifts
- **Two-phase fine-tuning**: first only the new head is trained, then the last 30 layers of the network are unfrozen and trained with a 10× lower learning rate
- Faster data loading with `cache()` and `prefetch()` — later epochs run about twice as fast
- Training curves, classification report and confusion matrix

## 🛠️ Tech

Python · TensorFlow / Keras · scikit-learn · Matplotlib · seaborn · Jupyter

## 🚀 Run it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The images aren't in the repo — put them in `dataset/train`, `dataset/valid` and `dataset/test` (one subfolder per class) and run `trash_sorting_model.ipynb`.
