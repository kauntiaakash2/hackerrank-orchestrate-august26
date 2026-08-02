examples=30
nearest_action_accuracy=0.900
hybrid_action_accuracy=0.900
hybrid_type_accuracy=0.600
calibration_methods=class-specific-bands
brier_score=0.0728
log_loss=0.2701
expected_calibration_error=0.1305
reliability_bins:
  [0.6,0.7] count=1 accuracy=0.000 confidence=0.662
  [0.7,0.8] count=7 accuracy=0.714 confidence=0.770
  [0.8,0.9] count=15 accuracy=1.000 confidence=0.848
  [0.9,1.0] count=7 accuracy=1.000 confidence=0.917

Calibration predictions use user-grouped held-out folds. These folds have too
few errors to fit a stable sigmoid or isotonic mapping, so this run correctly
uses the documented class-specific confidence bands.
# Evaluation

Run `python code/evaluate.py` to rebuild `evaluation/weak_labels.csv` from the
one-to-one history/event join and write complete machine-readable results to
`evaluation/results.json`.

The evaluator creates stable leakage groups from the normalized message
template, user, and sender/business/group identity, then selects up to five
`StratifiedGroupKFold` folds according to the number of available groups. Every
model is fit inside each training fold: deterministic rules, fold-local
retrieval, calibrated TF-IDF logistic regression, metadata logistic regression,
scikit-learn histogram gradient boosting, model-only, and the final hybrid.

The JSON contains action accuracy, macro F1, multiclass log loss, Brier score,
and fixed-label per-class confusion matrices. It also contains all requested
component ablations, solved-example action and message-type metrics, and
explicit `unavailable` records for embedding and LLM variants. Message-type
precedence was tuned from the persisted solved-example confusion matrix rather
than a qualitative assessment; media-only types remain limited when ASR/OCR is
unavailable.
