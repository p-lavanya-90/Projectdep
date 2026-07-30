# Multimodal Depression Detection System

Project Report and Technical Documentation

## Summary
This project is a research prototype for multimodal depression screening using text, audio, and visual features. It includes preprocessing scripts, trained machine-learning artifacts, a FastAPI backend, and a browser dashboard.

## Final Base Paper Comparison
| Category | Model/System | Purpose | Accuracy | Precision | Recall | F1 | AUC | MCC | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Base Paper | Base Paper | Reported baseline | 0.8500 | 0.7300 | 0.8500 | 0.7900 | 0.7300 | 0.6800 | Reference baseline |
| Multimodal | Hybrid_GB_PHQ_Override | Best accuracy | 0.8511 | 1.0000 | 0.5000 | 0.6667 | 0.7900 | 0.6423 | Beats: Accuracy, Precision, AUC |
| Multimodal | RecallOptimized_Logistic_RF_Override | Best recall | 0.7234 | 0.5217 | 0.8571 | 0.6486 | 0.7771 | 0.4792 | Beats: Recall, AUC |
| Multimodal | F1Optimized_Logistic_AND_GB | Best F1 balance | 0.8298 | 0.6667 | 0.8571 | 0.7500 | 0.7706 | 0.6353 | Beats: Recall, AUC |
| Multimodal | GradientBoostingClf_Accuracy | Best precision / conservative model | 0.8298 | 1.0000 | 0.4286 | 0.6000 | 0.7900 | 0.5873 | Beats: Precision, AUC |
| Multimodal Ensemble | WeightedSoftVoting | Best dev-selected balanced ensemble | 0.8085 | 0.6667 | 0.7143 | 0.6897 | 0.7684 | 0.5521 | Beats: AUC |
| Text Only | Text_GradientBoosting | Best text-only accuracy from current embeddings | 0.7872 | 1.0000 | 0.2857 | 0.4444 | 0.7716 | 0.4683 | Beats: Precision, AUC |
| Audio Only | Audio_SVC_RBF | Best audio-only accuracy from current 153-dim audio features | 0.7021 | 0.0000 | 0.0000 | 0.0000 | 0.3701 | 0.0000 | Below base paper |
| External Audio | AudioExt_SVC_RBF | Best DAIC + external audio branch | 0.7234 | 0.6000 | 0.2143 | 0.3158 | 0.6775 | 0.2279 | Below base paper |
| Visual Only | Visual_LogisticRegression | Best visual-only accuracy from CLNF/OpenFace features | 0.7021 | 0.5000 | 0.2857 | 0.3636 | 0.6667 | 0.2002 | Below base paper |
| Auxiliary Visual CNN | DepVidMood_CNN_visual_distress | Raw-image expression distress fallback | 0.8196 | 0.8908 | 0.7524 | 0.8158 | 0.9028 | 0.6503 | Auxiliary dataset, not DAIC-comparable |
| Improved Dataset | Improved_GradientBoosting | Best clean dataset-improvement run | 0.7872 | 0.8333 | 0.3571 | 0.5000 | 0.6775 | 0.4479 | Beats: Precision |

## Disclaimer
This is a research and screening-assistance prototype, not a standalone clinical diagnosis tool.
