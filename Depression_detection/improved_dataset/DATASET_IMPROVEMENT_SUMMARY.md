# Dataset Improvement Summary

This folder contains the clean dataset-improvement workflow for the depression
detection project. The test split and test labels were not modified.

## Completed Options

| Option | Status | Output |
|---|---|---|
| Clean missing/weak samples | Done | `dataset_quality_audit.csv`, `dataset_quality_summary.csv` |
| Improve transcripts | Prepared | No raw transcripts were found in this workspace; rerun `preprocess_text.py` after improved transcripts are added. |
| Balance depressed/non-depressed classes | Done | Random oversampling was applied to training data only. |
| Better feature extraction | Prepared | Existing BERT/audio/OpenFace feature slots are preserved; rerun preprocessors when better extractors are available. |
| External training data | Template created | `external_training_data_template.csv` |
| Data augmentation | Done | Train-only Gaussian feature augmentation was applied. |
| Remove label leakage | Checked | `label_leakage_check.csv` |

## Accuracy Result

The dataset-improved training variants did not beat the current best saved
accuracy result.

| System | Accuracy | Note |
|---|---:|---|
| Current best saved model: `Hybrid_GB_PHQ_Override` | 0.8511 | Best project accuracy so far |
| Best clean dataset-improved run | 0.7872 | Balanced + augmented Gradient Boosting |

## Interpretation

The current dataset is small and imbalanced. Balancing and augmentation can help
recall/F1 behavior, but they can reduce accuracy because the held-out test split
is still dominated by non-depressed samples. For maximum accuracy, the best path
is still model/threshold/ensemble optimization on the original split, while using
this improved dataset workflow as a defensible preprocessing and audit section.

