**Polski** | [English](README.en.md)

# Klasyfikacja śmieci z użyciem AI

Model głębokiego uczenia, który na podstawie zdjęcia przypisuje odpad do jednej z **6 kategorii**: `cardboard` (karton), `glass` (szkło), `metal`, `paper` (papier), `plastic` (plastik) i `trash` (zmieszane). Zbudowany w TensorFlow/Keras z użyciem **transfer learningu** na modelu **EfficientNetB0** (wytrenowanym wcześniej na ImageNet).

## Wyniki

| | Accuracy |
|---|---|
| Walidacja, po fazie 1 | ~0,89 |
| Walidacja, po fine-tuningu | ~0,91 |
| **Zbiór testowy** | **0,90** |

Notatnik pokazuje też wykresy przebiegu uczenia oraz raport klasyfikacji i macierz pomyłek dla zbioru testowego. Najsłabiej rozpoznawana jest klasa `trash`, która ma najmniej przykładów.

## Jak to działa

1. **Wczytanie danych** — obrazy (224 × 224) z `dataset/train`, `dataset/valid` i `dataset/test`, w paczkach po 32.
2. **Szybsze wczytywanie** — `cache()` + `prefetch()`, dzięki czemu każda epoka po pierwszej trwa mniej więcej o połowę krócej.
3. **Augmentacja** — losowe odbicia, obroty, przybliżenia, zmiany kontrastu i jasności oraz przesunięcia, żeby model uczył się na bardziej zróżnicowanych obrazach.
4. **Faza 1** — EfficientNetB0 z zamrożonymi wagami jako ekstraktor cech oraz nowa głowa klasyfikująca (pooling, warstwa gęsta 128 neuronów, dropout); trening do 15 epok z wczesnym zatrzymaniem.
5. **Faza 2 — fine-tuning** — odmrożenie 30 ostatnich warstw EfficientNetB0 i dalszy trening z 10× mniejszym learning rate (do 20 epok). Najlepszy model zapisywany jest do `best_model.keras`.
6. **Ocena** na zbiorze testowym.

## Zbiór danych

Zdjęcia **nie są** dołączone do repozytorium (`dataset/` jest w `.gitignore`). Umieść je w takiej strukturze — jeden podfolder na klasę:

```
dataset/
├── train/   (1767 zdjęć)
├── valid/   (504 zdjęcia)
└── test/    (253 zdjęcia)
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Odtworzenie środowiska

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Następnie otwórz `trash_sorting_model.ipynb` w Jupyterze lub VS Code i uruchom wszystkie komórki. Trening jest znacznie szybszy na GPU.
