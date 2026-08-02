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
