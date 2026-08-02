"""Strict submission validator."""
from pathlib import Path
import pandas as pd
from .config import ACTIONS,MESSAGE_TYPES,OUTPUT_COLUMNS
from .data_loader import DatasetBundle

def validate_output(path:Path,data:DatasetBundle,semantic_evidence:bool=False)->None:
    out=pd.read_csv(path,dtype={"message_id":str,"evidence_message_ids":str}); incoming=data.tables["messages"]; historical=set(data.tables["message_history"].message_id)
    assert list(out)==OUTPUT_COLUMNS
    assert out.message_id.tolist()==incoming.message_id.tolist()
    assert out.message_id.is_unique and not out.isna().any().any()
    assert set(out.action)<=ACTIONS and set(out.message_type)<=MESSAGE_TYPES
    assert out.confidence.between(0,1).all() and out.reason.str.strip().ne("").all() and out.reason.str.len().le(180).all()
    for value in out.evidence_message_ids:
        assert value=="none" or all(x in historical for x in value.split(";"))
    if semantic_evidence:
        from .retrieval import Retriever
        retriever=Retriever(data.history_events); incoming_by_id=incoming.set_index("message_id")
        for result in out.itertuples(index=False):
            expected=retriever.evidence(incoming_by_id.loc[result.message_id],result.action,result.message_type,"validation")
            supplied=[] if result.evidence_message_ids=="none" else result.evidence_message_ids.split(";")
            supported=set() if expected=="none" else set(expected.split(";"))
            assert set(supplied)<=supported, f"contradictory or irrelevant evidence for {result.message_id}: {supplied}"
