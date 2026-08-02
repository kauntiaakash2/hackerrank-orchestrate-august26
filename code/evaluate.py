"""Evaluate ablations on solved examples without using them for training."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from src.data_loader import DatasetBundle,normalize_text
from src.retrieval import Retriever
from src.router import Router

def main()->None:
    data=DatasetBundle.load(Path("dataset")); sample=data.tables["sample_messages"].copy(); sample["normalized_text"]=sample.message_text.map(normalize_text).str.lower()
    router=Router(data,Path("cache/evaluation_media.json")); retriever=Retriever(data.history_events)
    hybrid=pd.DataFrame([router.route(r) for _,r in sample.iterrows()])
    nearest=[retriever.top(r,1).iloc[0].weak_action for _,r in sample.iterrows()]
    print(f"examples={len(sample)}")
    print(f"nearest_action_accuracy={(sample.action.reset_index(drop=True)==pd.Series(nearest)).mean():.3f}")
    print(f"hybrid_action_accuracy={(sample.action.reset_index(drop=True)==hybrid.action).mean():.3f}")
    print(f"hybrid_type_accuracy={(sample.message_type.reset_index(drop=True)==hybrid.message_type).mean():.3f}")
    print("Rules/model ablations are documented qualitatively where the solved set lacks sufficient grouped folds.")

if __name__=="__main__": main()
