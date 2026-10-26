🇵🇱 **Polski** | 🇬🇧 [English](README.en.md)

# ♻️ Klasyfikacja śmieci na podstawie zdjęć

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)

> 🎓 Zadanie rekrutacyjne do koła naukowego **CyberTech** (Politechnika Wrocławska)

Model klasyfikujący odpady na podstawie zdjęcia do jednej z 6 klas: karton, szkło, metal, papier, plastik i odpady zmieszane.

## 🎯 Wynik

**90% accuracy na zbiorze testowym**

## ✨ Co zrobiłem

- **Transfer learning** na **EfficientNetB0** (wytrenowanym na ImageNet) z własną głową klasyfikującą
- **Augmentacja danych** — losowe odbicia, obroty, zoom, kontrast, jasność i przesunięcia
- **Fine-tuning** w dwóch fazach: najpierw trenuję tylko nową głowę, potem odmrażam 30 ostatnich warstw sieci i douczam z 10× mniejszym learning rate
- Szybsze wczytywanie danych dzięki `cache()` i `prefetch()` — kolejne epoki trwają ok. dwa razy krócej
- Wykresy uczenia, raport klasyfikacji i macierz pomyłek

## 🛠️ Technologie

Python · TensorFlow / Keras · scikit-learn · Matplotlib · seaborn · Jupyter

## 🚀 Uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Zdjęcia nie są dołączone do repozytorium — umieść je w `dataset/train`, `dataset/valid` i `dataset/test` (po jednym podfolderze na klasę) i uruchom `trash_sorting_model.ipynb`.
