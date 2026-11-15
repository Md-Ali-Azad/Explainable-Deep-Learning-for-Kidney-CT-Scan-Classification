# Explainable Deep Learning for Kidney CT Scan Classification

This project implements a deep-learning pipeline for classifying kidney CT images and examining whether post-hoc explainability methods provide useful evidence about the model's predictions.

The implementation is mainly based on the pipeline described in:

> Vasu, A., Srivastava, S. S., & Olivia, D. (2025, December). *Explainable Deep Learning for Kidney CT Scan Classification*. In **2025 International Conference on Modeling, Simulation & Intelligent Computing (MoSICom)** (pp. 141–146). IEEE.

I used this paper as a guide to understand explainable AI for kidney CT images and implemented the full workflow, including data inspection, model training and evaluation, calibration, and explainability analysis. I also added several evaluation steps to test the explanations rather than assuming that a heat-map is useful simply because it looks plausible.

## Project workflow

The notebook is organised into three main parts.

### 1. Dataset and ground-truth inspection

The dataset contains four image-level classes:

- Normal
- Cyst
- Tumor
- Stone

Each image has a class label. The dataset does **not** provide lesion masks, bounding boxes, patient identifiers, or scan-phase labels. Therefore, the labels tell the model which class an image belongs to, but they do not identify the exact pixels containing a lesion.

The notebook inspects the class distribution and sample images. It also checks for possible differences in image properties that could become shortcuts for the model.

### 2. Model training and evaluation

The classification model is based on **VGG16 with ImageNet weights**. The last two convolutional blocks are fine-tuned, followed by a Dense(256) layer, Dropout(0.5), and a four-class softmax output.

The workflow includes:

- Training, validation, and test splits.
- Early stopping based on validation loss.
- Classification metrics and a confusion matrix.
- Temperature scaling fitted on the validation set.
- Expected Calibration Error (ECE), before and after calibration.
- A near-duplicate image check to investigate possible leakage between the training and test sets.

The notebook supports both a stratified random image split and a heuristic grouped split. Since the dataset does not include patient IDs, the grouped split uses blocks of neighbouring file indices as a proxy. It is not equivalent to a true patient-level split.

### 3. Explainability analysis

Three post-hoc explanation methods are implemented:

- **Grad-CAM**: uses gradients flowing into the final convolutional feature maps to highlight image regions that influence a selected class score.
- **Score-CAM**: uses activation maps as input masks and weights them according to their effect on the class score.
- **LIME**: perturbs image superpixels and fits a local surrogate model to estimate which regions support the prediction.

The notebook evaluates the explanations in several ways:

1. **Faithfulness:** deletion and insertion tests compare the model's output as salient pixels are removed or revealed. A random heat-map is used as a reference.
2. **Localisation and shortcut inspection:** the proportion of Grad-CAM attention outside an estimated body region is measured. Optional lesion boxes can be supplied for more direct localisation evaluation.
3. **Selective prediction:** explanation-related signals are compared with calibrated confidence to see whether they help identify predictions that are more likely to be correct.
4. **XAI-guided refinement:** the model is run on a Grad-CAM-selected crop, and the crop prediction is fused with the original prediction.
5. **Case studies:** individual correct and incorrect predictions are inspected alongside their explanation maps.

---

## Results and interpretation

The following results are from the evaluation run represented in the notebook output. The XAI evaluation uses a small, class-balanced subset. These numbers should be treated as exploratory evidence about this model and dataset, not as clinical performance estimates.

### 3a. Faithfulness: do the heat-maps highlight pixels the model uses?

The deletion test progressively replaces the most salient pixels with blurred pixels. A faithful explanation should cause the target-class score to fall quickly, producing a **lower deletion AUC**.

The insertion test starts from a blurred image and progressively restores the most salient pixels. A faithful explanation should restore the target-class score quickly, producing a **higher insertion AUC**.

| Method | Deletion AUC (lower is better) | Insertion AUC (higher is better) | Images |
|---|---:|---:|---:|
| Grad-CAM | 0.344 | 0.958 | 200 |
| Score-CAM | 0.361 | 0.957 | 200 |
| LIME | 0.647 | 0.965 | 20 |
| Random map | 0.477 | 0.477 | 200 |

**What these results suggest**

- Grad-CAM and Score-CAM have lower deletion AUCs than the random reference. On average, removing the regions they highlight reduces the model's target-class score more than removing pixels selected randomly.
- Grad-CAM beats the random map on the deletion test for 66% of evaluated images; Score-CAM does so for 67%. This is encouraging, but it also means the result is not consistent across every image.
- LIME has a high deletion AUC (0.647), worse than the random reference (0.477). In this run, its highlighted regions did not behave as reliably under deletion. Its insertion AUC is high, but insertion alone does not establish that an explanation is faithful.
- The high insertion values for Grad-CAM, Score-CAM, and LIME should be interpreted alongside deletion, not in isolation. Insertion and deletion can be affected by the image baseline, perturbation method, and model behaviour.

**What needs improvement**

The evaluation should be repeated on a larger and more representative test set. It would also be useful to test different perturbation baselines and step sizes, report variation across images (for example, confidence intervals), and examine results separately for each class. LIME was evaluated on only 20 images, so its result is especially preliminary.

### 3b. Localisation and shortcut check

The notebook estimates a body region using image thresholding and measures how much Grad-CAM activation falls outside that region.

| Prediction group | Grad-CAM mass outside the estimated body |
|---|---:|
| Correct predictions | 0.251 |
| Incorrect predictions | 0.228 |

Around one quarter of the Grad-CAM mass lies outside the estimated body region for both correct and incorrect predictions. This deserves inspection. It may indicate that some model attention falls on the background, image borders, labels, or other non-anatomical regions.

However, this measurement is only a **rough shortcut check**. The body mask is generated automatically using thresholding and morphological operations. It is not a manually verified anatomical segmentation. Also, the fact that incorrect predictions have a slightly lower off-body value does not mean their explanations are better.

The dataset does not contain lesion bounding boxes or pixel-level masks. As a result, this experiment cannot establish whether the heat-map points to the actual cyst, tumour, or stone. A stronger localisation evaluation requires expert-annotated lesion regions or bounding boxes. The notebook can accept a CSV with `filename,x1,y1,x2,y2` coordinates for an optional box-based evaluation.

### 3c. Can explanation behaviour help identify unreliable predictions?

The notebook compares selective accuracy using three explanation-related signals and calibrated confidence. Selective prediction keeps only the highest-ranked predictions at each coverage level.

| Signal | 100% kept | 90% kept | 80% kept | 70% kept | 50% kept |
|---|---:|---:|---:|---:|---:|
| Calibrated confidence only | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| Grad-CAM / Score-CAM agreement | 0.990 | 0.994 | 1.000 | 1.000 | 1.000 |
| Deletion drop | 0.990 | 0.989 | 0.988 | 0.986 | 0.990 |
| Combined mean rank | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |

**What these results suggest**

The combined signal reaches 1.000 accuracy at 90% coverage, while confidence alone also reaches 1.000 at 90% coverage. Therefore, this run does not show a clear additional benefit from combining explanation behaviour with confidence.

Method agreement gives a small improvement at 90% coverage (0.994 compared with 1.000 for confidence in this run, so it is actually lower), and deletion drop does not improve selective accuracy. At lower coverage levels, several signals reach 1.000, but this is likely influenced by the very high starting accuracy and the small evaluation subset.

The important point is that **these results do not demonstrate that XAI can reliably filter incorrect predictions**. Here, explanation maps are more useful as a way to inspect model behaviour and individual cases than as an automatic confidence or triage system.

**What needs improvement**

A larger test set with more errors is needed to evaluate risk-coverage behaviour meaningfully. The comparison should include risk-coverage curves and summary measures such as area under the risk-coverage curve, with uncertainty estimates. The ranking signals and any combination of them should be defined on validation data and assessed on an independent test set to avoid tuning to the evaluation subset.

### 3d. XAI-guided refinement

The notebook crops the image around the high-saliency Grad-CAM region, classifies that crop, and averages the original and crop logits.

| Evaluation | Accuracy |
|---|---:|
| Original prediction | 0.990 |
| Refined prediction | 0.990 |

There were **0 wrong-to-right changes** and **0 right-to-wrong changes** in this run. The refinement step therefore made no measurable difference to accuracy.

This is still a useful negative result. It shows that applying a saliency-guided crop does not automatically improve classification. The crop may remove useful context, retain the same evidence as the original image, or fail to isolate the relevant anatomy. The current fusion rule is also simple and has not been optimised.

Future work could compare different crop thresholds and margins, test multi-scale inputs, and investigate whether the crop prediction adds information beyond the original image. Any changes should be assessed on a separate validation set before final testing.

### 3e. Case studies

The notebook saves visual case studies. These examples place the original image and explanation maps alongside the ground-truth class and model prediction.

The selected examples illustrate several situations:

- **Correct prediction with potentially weak evidence:** the model predicts the correct class with high confidence, but the explanation signals may disagree or indicate limited evidence use. A correct label does not guarantee that the model relied on appropriate anatomical features.
- **Confident error:** an image labelled Tumor is predicted as Normal with confidence 0.95. The contrast explanation can help inspect where the model's evidence for the true class differs from the evidence supporting its prediction.
- **Incorrect prediction with moderate confidence:** an example predicted as Normal instead of Tumor, with confidence 0.68, could be flagged for closer review.
- **Correct prediction with substantial off-body attention:** one correct Tumor prediction has 0.55 of its Grad-CAM mass outside the estimated body region. This is a warning that prediction correctness and explanation quality are different questions.

These are illustrative cases, not proof that the highlighted regions correspond to pathology. They are useful for identifying examples that deserve further investigation.

---

## Overall findings

This implementation goes beyond generating Grad-CAM, Score-CAM, and LIME visualisations. It tests how the maps behave when image regions are perturbed, checks whether attention falls outside an estimated body region, compares explanation signals with confidence, and tests a simple saliency-guided refinement step.

The current run suggests:

1. **Grad-CAM and Score-CAM show some faithfulness:** both outperform the random reference on average in the deletion test, but neither does so for every image.
2. **LIME needs further investigation:** its deletion result is worse than the random reference, and it was tested on a much smaller subset.
3. **Potential shortcut behaviour remains:** approximately 25% of Grad-CAM mass falls outside the estimated body region. This requires better anatomical masks and expert review.
4. **XAI did not clearly improve automatic prediction filtering:** confidence already performed very strongly on this small subset, and the combined signal did not show a clear additional benefit.
5. **The crop-based refinement did not change accuracy:** the current approach needs further investigation rather than being treated as an improvement.

### Limitations and next steps

The current results have several important limitations:

- **Image-level labels only:** there are no lesion masks or bounding boxes, so the experiments cannot verify whether the explanation identifies the actual pathology.
- **No patient identifiers:** a random image-level split may place neighbouring slices from the same scan in both training and test sets. The notebook's grouped split is only a heuristic proxy based on file ordering. A patient-level split would be preferable.
- **Small XAI evaluation subset:** the faithfulness and selective-prediction results are exploratory. LIME was evaluated on only 20 images.
- **High baseline accuracy in the evaluated subset:** with accuracy already at 0.990, selective-prediction comparisons have little room to show improvement. A larger test set with a meaningful number of errors is needed.
- **Post-hoc explanation limits:** heat-maps describe patterns associated with the model's output. They do not establish clinical causation or prove that a highlighted region is medically relevant.
- **Approximate body segmentation:** the threshold-based body mask can itself be inaccurate and should be replaced or checked using a more reliable anatomical segmentation.
- **No clinical validation:** the explanations have not been evaluated by radiologists or other medical professionals.

The next steps are to use a patient-level split if patient identifiers become available, obtain expert lesion annotations, increase the evaluation sample, quantify uncertainty in the metrics, and ask medical experts to assess whether the explanations are relevant and interpretable.


## Paper citation

This paper helped me understand explainable AI for kidney CT image classification and guided my implementation of the complete pipeline, including model training, evaluation, and explainability analysis.

Vasu, A., Srivastava, S. S., & Olivia, D. (2025, December). Explainable Deep Learning for Kidney CT Scan Classification. In *2025 International Conference on Modeling, Simulation & Intelligent Computing (MoSICom)* (pp. 141–146). IEEE.

## Dataset citation

The dataset used in this implementation is available from Kaggle:

[CT Kidney Dataset: Normal, Cyst, Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
