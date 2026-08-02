"""Personalized historical retrieval and weak supervision."""
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def weak_action(row: pd.Series) -> str:
    if row.message_reported=="1" or row.muted_after_message=="1" or row.notification_dismissed=="1": return "mute"
    if row.message_replied=="1" and float(row.reaction_time_minutes or 999)<=9: return "notify"
    return "digest"

class Retriever:
    def __init__(self, history: pd.DataFrame):
        self.history=history.copy(); self.history["weak_action"]=self.history.apply(weak_action,axis=1)
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,strip_accents="unicode")
        self.matrix=self.vectorizer.fit_transform(self.history.normalized_text.fillna(""))

    def top(self,row: pd.Series,n:int=3) -> pd.DataFrame:
        q=self.vectorizer.transform([row.normalized_text]); sim=cosine_similarity(q,self.matrix)[0]
        h=self.history.copy(); h["similarity"]=sim
        h["score"]=h.similarity + .42*(h.user_id==row.user_id) + .22*((h.sender_user_id==row.sender_user_id)&(row.sender_user_id!="")) + .22*((h.group_id==row.group_id)&(row.group_id!="")) + .22*((h.business_id==row.business_id)&(row.business_id!=""))
        return h.sort_values(["score","created_at"],ascending=False).head(n)
