# Deep-Learning
# WikiArt Artist Classification (Deep Learning)

Deep Learning project, MSc Data Science & Advanced Analytics, NOVA IMS (2025/2026).

Identifies the painter of an artwork from its image alone, on a 23-artist subset of WikiArt.
This is difficult because style is spread across texture, colour, light and composition, many artists look
alike (e.g. the Impressionists), and there are only a few hundred paintings per artist.

## Data
- 13,172 paintings across 23 artists after perceptual-hash duplicate removal
- Imbalanced classes (336 to 1,322 images per artist), handled with balanced class weights and macro-F1
- 70% train, 15% validation, 15% test (1,984 images, evaluated once)
- Images resized to 224 x 224

## Approach
- **EDA:** class balance, colour and grayscale analysis per artist, duplicate and corruption checks
- **Painting-specific augmentation:** compared identity, geometric and photometric strategies.
  Geometric changes (rotation, zoom) hurt, because composition is part of an artist's style, so only mild
  brightness and contrast changes were kept
- **Baseline:** 4-block CNN trained from scratch
- **Transfer learning:** EfficientNetV2S and MobileNetV3Large fine-tuned progressively in 3 phases
  (frozen backbone, deepest block unfrozen, mid-level block unfrozen) with decreasing learning rates
- **Regularisation study:** frozen BatchNorm, higher dropout and L2, and its trade-off with accuracy
- **Ensemble:** averaged softmax outputs of both models (50/50 outperformed 60/40)

## Results
| Model | Validation macro-F1 |
|---|---|
| CNN from scratch | 0.633 |
| MobileNetV3Large | 0.803 |
| EfficientNetV2S | 0.851 |
| **Ensemble (50/50)** | **0.862** |

**Test set (ensemble):** macro-F1 **0.860**, accuracy 0.874, Cohen's Kappa 0.867, MCC 0.867,
macro ROC AUC 0.991. The close match between validation and test scores shows model selection did not
overfit the validation set.

Errors mostly come from stylistically close artists (e.g. Pissarro confused with Monet,
Kustodiev with Repin), not from one systematic failure.
