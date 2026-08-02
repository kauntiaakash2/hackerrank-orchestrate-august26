"""Strict submission validator."""
from pathlib import Path
import pandas as pd
from .config import ACTIONS,MESSAGE_TYPES,OUTPUT_COLUMNS
from .data_loader import DatasetBundle

def validate_output(path:Path,data:DatasetBundle)->None:
    out=pd.read_csv(path,dtype={"message_id":str,"evidence_message_ids":str}); incoming=data.tables["messages"]; historical=set(data.tables["message_history"].message_id)
    assert list(out)==OUTPUT_COLUMNS
    assert out.message_id.tolist()==incoming.message_id.tolist()
    assert out.message_id.is_unique and not out.isna().any().any()
    assert set(out.action)<=ACTIONS and set(out.message_type)<=MESSAGE_TYPES
    assert out.confidence.between(0,1).all() and out.reason.str.strip().ne("").all() and out.reason.str.len().le(180).all()
    for value in out.evidence_message_ids:
        assert value=="none" or all(x in historical for x in value.split(";"))
