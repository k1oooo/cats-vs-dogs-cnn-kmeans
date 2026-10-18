# Cats vs Dogs: Supervised vs Unsupervised Image Classification

**SAIA 2113 (Machine Learning), Assignment 1.** This project classifies cat and dog images with three approaches and compares them:

1. a **CNN built from scratch**
2. **transfer learning** with a pretrained MobileNetV2
3. **K-Means clustering** (unsupervised) on PCA-reduced pixels

The goal is to show how much supervision and learned features matter for image classification.

## Results at a Glance

| Approach | Type | Validation accuracy |
|---|---|---|
| CNN from scratch | Supervised | 65.71% |
| **MobileNetV2 transfer learning** | Supervised | **95.71%** |
| K-Means (k = 2, PCA 50) | Unsupervised | 59.71% (cluster-aligned, all 345 images) |

Always-predict-dog baseline: 65.7% on the validation set and 65.5% on the full dataset.

## Dataset

Cats vs Dogs images from the Oxford-IIIT Pet Dataset (Roboflow version via Kaggle).

| Split | Cats (0) | Dogs (1) | Total |
|---|---|---|---|
| Train | 95 | 180 | 275 |
| Validation | 24 | 46 | 70 |
| **Total** | **119** | **226** | **345** |

- Roughly an 80/20 train/validation split, with a class imbalance of about 1:2 (cats : dogs).
- `train.csv` and `val.csv` list each image path and label (`0` = cat, `1` = dog).
- Image folders follow the layout `train/cat/`, `train/dog/`, `val/cat/`, `val/dog/`.

## Methods

### Part 1: Supervised learning (`supervised_learning.ipynb`)

**Preprocessing:** images resized to 128×128 and rescaled to [0, 1]. The training set is augmented (rotation ±20°, zoom 0.2, horizontal flip, width/height shift 0.1). The validation set is only rescaled.

**Model A: CNN from scratch**
- 4 convolutional blocks (32 → 64 → 128 → 256 filters), each with two Conv layers, batch norm, max pooling and dropout (0.25)
- Dense(512) → Dense(256) → sigmoid output, with dropout 0.5
- About 9.7M parameters, Adam optimiser, binary cross-entropy, early stopping on `val_loss` (patience 3)

**Model B: Transfer learning**
- MobileNetV2 pretrained on ImageNet, with the base frozen
- GlobalAveragePooling → Dense(128) → Dropout(0.3) → sigmoid output
- 2.4M parameters in total, of which only 164K are trainable. Early stopping on `val_loss` (patience 5)

### Part 3: Unsupervised learning (`SAIA2113_Part3_Part4__1_.ipynb`)

1. Resize images to 64×64, flatten to 12,288 features and normalise to [0, 1]
2. Reduce to 50 dimensions with **PCA** (83.7% variance retained)
3. Choose k with the elbow method, then run **K-Means** (k = 2, `k-means++`, `n_init = 20`)
4. Align cluster IDs to the true labels, then evaluate with a confusion matrix and visualise with a 2D PCA projection

K-Means never sees the labels. They are used only for evaluation.

### Part 4: Reflection

The same notebook compares supervised and unsupervised learning, discusses data collection and preprocessing challenges, analyses the K-Means output, and proposes future improvements.

## Detailed Results

**CNN from scratch.** Accuracy was 65.71%, but the model predicted *every* image as a dog (cat recall = 0, dog recall = 1.00). This equals the majority-class baseline, so the network did not learn a useful decision rule in the few epochs it ran. With 275 training images and a 9.7M-parameter model, this is expected.

**MobileNetV2 transfer learning.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cat | 0.96 | 0.92 | 0.94 | 24 |
| Dog | 0.96 | 0.98 | 0.97 | 46 |
| **Overall accuracy** | | | **0.9571** | 70 |

**K-Means.**

| | Predicted cat | Predicted dog |
|---|---|---|
| **True cat** | 51 | 68 |
| **True dog** | 71 | 155 |

Both clusters contain a mix of cats and dogs, and the PCA projection shows heavy overlap between the classes. Raw pixels capture colour and background rather than animal type.

## Key Takeaways

- **Pretrained features matter most on small datasets.** Reusing ImageNet features lifted accuracy from 65.7% to 95.7% with the same data.
- **Supervision matters.** K-Means on raw pixels performed worse than always guessing "dog".
- **Accuracy alone can mislead on imbalanced data.** The from-scratch CNN's 65.7% looks reasonable but hides a model that never predicts "cat".

## Repository Structure

```
.
├── supervised_learning.ipynb              # Parts 1-2: CNN from scratch + MobileNetV2
├── SAIA2113_Part3_Part4__1_.ipynb         # Parts 3-4: K-Means clustering + reflection
├── train.csv                              # Training image paths and labels
├── val.csv                                # Validation image paths and labels
├── train/ and val/                        # Image folders (cat/ and dog/ subfolders)
└── README.md
```

## Getting Started

The notebooks were written and run on **Google Colab**, so paths point to `/content/`.

```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn pillow
```

1. Download the images and arrange them as `train/cat`, `train/dog`, `val/cat` and `val/dog`.
2. **`supervised_learning.ipynb`** reads from `/content/dataset/train` and `/content/dataset/val`. Update these paths to match your setup.
3. **`SAIA2113_Part3_Part4__1_.ipynb`** reads `train.csv`, `val.csv` and the images relative to `BASE = '/content'`. Change `BASE` to your project folder.
4. Run the notebooks top to bottom.

## Limitations and Future Work

- **Small dataset** (345 images). A 70-image validation set means each image is worth about 1.4 percentage points, so the results are noisy.
- **No separate test set.** The validation set is used for early stopping and for reporting final metrics, so the reported accuracy may be slightly optimistic.
- **Fine-tuning** the top layers of MobileNetV2 and trying other backbones (ResNet, EfficientNet) could improve results.
- **Class weights** or oversampling could address the 1:2 imbalance, which the from-scratch CNN did not handle.
- **Better clustering:** run K-Means on deep embeddings from a pretrained network instead of raw pixels, and try GMM or DBSCAN. Silhouette score would help validate k.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · NumPy · Pillow · matplotlib · seaborn · Google Colab

## Author

**Faiz**, AI student. SAIA 2113 Machine Learning, Assignment 1.
