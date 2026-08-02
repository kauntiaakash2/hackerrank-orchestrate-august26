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
