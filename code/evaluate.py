"""Evaluate ablations on solved examples without using them for training."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from src.data_loader import DatasetBundle,normalize_text
from src.retrieval import Retriever
from src.router import Router
from src.confidence import ConfidenceCalibrator, calibration_metrics
from sklearn.model_selection import GroupKFold

def main()->None:
    data=DatasetBundle.load(Path("dataset")); sample=data.tables["sample_messages"].copy(); sample["normalized_text"]=sample.message_text.map(normalize_text).str.lower()
    router=Router(data,Path("cache/evaluation_media.json")); retriever=Retriever(data.history_events)
    detailed=[router.route_detailed(r) for _,r in sample.iterrows()]
    hybrid=pd.DataFrame([item[0] for item in detailed]); signals=[item[1] for item in detailed]
    nearest=[retriever.top(r,1).iloc[0].weak_action for _,r in sample.iterrows()]
    print(f"examples={len(sample)}")
    print(f"nearest_action_accuracy={(sample.action.reset_index(drop=True)==pd.Series(nearest)).mean():.3f}")
    print(f"hybrid_action_accuracy={(sample.action.reset_index(drop=True)==hybrid.action).mean():.3f}")
    print(f"hybrid_type_accuracy={(sample.message_type.reset_index(drop=True)==hybrid.message_type).mean():.3f}")
    correct=(sample.action.reset_index(drop=True)==hybrid.action).to_numpy()
    # User-grouped folds ensure a user's solved outcomes never calibrate that
    # same user's held-out confidence. Router predictions themselves never train
    # on sample_messages.csv.
    groups=sample.user_id.to_numpy(); calibrated=[0.0]*len(sample); methods=set()
    folds=min(5,len(set(groups)))
    for train,test in GroupKFold(n_splits=folds).split(sample,correct,groups):
        model=ConfidenceCalibrator().fit([signals[i] for i in train],correct[train])
        methods.add(model.method)
        for i in test: calibrated[i]=model.predict(hybrid.iloc[i].action,signals[i])
    metrics=calibration_metrics(correct,calibrated)
    print(f"calibration_methods={','.join(sorted(methods))}")
    print(f"brier_score={metrics['brier_score']:.4f}")
    print(f"log_loss={metrics['log_loss']:.4f}")
    print(f"expected_calibration_error={metrics['expected_calibration_error']:.4f}")
    print("reliability_bins:")
    for item in metrics["reliability_bins"]:
        print(f"  [{item['lower']:.1f},{item['upper']:.1f}] count={item['count']} accuracy={item['accuracy']:.3f} confidence={item['confidence']:.3f}")

if __name__=="__main__": main()
