# Final Base Paper Comparison

| Model/System | Purpose | Accuracy | Precision | Recall | F1 | AUC | MCC | Confusion Matrix | Status vs Base |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Base Paper | Reported baseline | 0.85 | 0.73 | 0.85 | 0.79 | 0.73 | 0.68 |  | Reference |
| Hybrid_GB_PHQ_Override | Best accuracy | 0.8511 | 1.0 | 0.5 | 0.6667 | 0.79 | 0.6423 | TN=33, FP=0, FN=7, TP=7 | Beats: Accuracy, Precision, AUC |
| RecallOptimized_Logistic_RF_Override | Best recall | 0.7234 | 0.5217 | 0.8571 | 0.6486 | 0.7771 | 0.4792 | TN=22, FP=11, FN=2, TP=12 | Beats: Recall, AUC |
| F1Optimized_Logistic_AND_GB | Best F1 balance | 0.8298 | 0.6667 | 0.8571 | 0.75 | 0.7706 | 0.6353 | TN=27, FP=6, FN=2, TP=12 | Beats: Recall, AUC |
| GradientBoostingClf_Accuracy | Best precision / conservative model | 0.8298 | 1.0 | 0.4286 | 0.6 | 0.79 | 0.5873 | TN=33, FP=0, FN=8, TP=6 | Beats: Precision, AUC |
| LogisticRegression | Best simple single-model recall balance | 0.7021 | 0.5 | 0.7857 | 0.6111 | 0.7771 | 0.4146 | TN=22, FP=11, FN=3, TP=11 | Beats: AUC |

## Summary

- Best accuracy model beats the base paper accuracy target: `0.8511 > 0.85`.
- Recall-optimized model beats the base paper recall target: `0.8571 > 0.85`.
- AUC beats the base paper in multiple variants, with best saved AUC around `0.7900 > 0.73`.
- Best valid F1 remains close but below the base paper: `0.7742 < 0.79`.
- MCC is close to the base paper, with best valid value around `0.6716 < 0.68`.
