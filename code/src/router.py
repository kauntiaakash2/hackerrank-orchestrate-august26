"""Hierarchical safety, personalization, retrieval, and content ensemble."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .config import OUTPUT_COLUMNS
from .data_loader import DatasetBundle
from .media_processing import MediaProcessor
from .retrieval import Retriever
from .reasons import ReasonContext, generate_reason
from .security import assess

class Router:
    def __init__(self,data:DatasetBundle,cache:Path):
        self.data=data; t=data.tables; self.retriever=Retriever(data.history_events); self.media=MediaProcessor(data.root,cache)
        self.users=t["users"].set_index("user_id").to_dict("index"); self.groups=t["groups"].set_index("group_id").to_dict("index")
        self.members=t["group_members"].set_index(["group_id","user_id"]).to_dict("index"); self.businesses=t["business_accounts"].set_index("business_id").to_dict("index")
        self.relations=t["user_business_history"].set_index(["user_id","business_id"]).to_dict("index")
        self.media_paths={**t["images"].set_index("image_id").file_path.to_dict(),**t["voice_notes"].set_index("voice_note_id").file_path.to_dict()}

    def route_all(self)->pd.DataFrame: return pd.DataFrame([self.route(r) for _,r in self.data.tables["messages"].iterrows()],columns=OUTPUT_COLUMNS)

    def route(self,r:pd.Series)->dict[str,object]:
        text=r.normalized_text; biz=self.businesses.get(r.business_id); rel=self.relations.get((r.user_id,r.business_id),{}); member=self.members.get((r.group_id,r.user_id),{}); group=self.groups.get(r.group_id,{})
        media=self.media.extract(r.media_type,r.media_id,self.media_paths.get(r.media_id,""))
        risk=assess(text,biz); top=self.retriever.top(r); best=top.iloc[0]; evidence=self._evidence(top,r)
        typ=self._type(text,r,biz); action="digest"; rule="default"; conf=.72; historical_action=None
        if risk.score:
            action,typ,rule,conf="mute",risk.kind,"safety",risk.score
        elif typ=="promotion" and rel.get("promotions_opted_out_at"):
            action,rule,conf="mute","promotion_opt_out",.88
        elif typ=="promotion" and rel.get("allows_promotions")=="0" and int(rel.get("messages_dismissed_30d","0") or 0)>=4:
            action,rule,conf="mute","promotion_dismissed",.88
        elif typ in {"greeting","forward"} and (int(r.forwarded_count or 0)>=5 or member.get("group_muted_by_user")=="1"):
            action,rule,conf="mute","forward_or_muted_group",.86
        elif self._urgent(text,r,group):
            action,typ,rule,conf="notify",("event" if typ=="event" else "urgent"),("urgent_direct" if f"@{r.user_id}" in text else "urgent_operational"),.88
        elif typ=="business_update" and rel and re.search(r"delivery|pickup|ride|booking|appointment|order|transaction",text):
            action,rule,conf="notify",("active_transaction_verified" if biz and biz.get("verified")=="1" else "active_transaction"),.88
        elif best.score>=.78 and best.user_id==r.user_id:
            action=best.weak_action; historical_action=action; conf=min(.89,.68+.16*float(best.similarity)); rule=f"history_{action}"
        elif typ=="promotion":
            action="digest" if rel.get("allows_promotions")=="1" or not rel else "mute"; rule="promotion_digest" if action=="digest" else "promotion_mute"; conf=.79
        elif typ in {"spam","scam"}: action="mute"; rule="spam"; conf=.84
        elif r.conversation_type=="personal" and re.search(r"\?|can you|could you|please",text) and not re.search(r"no urgency|not urgent|tomorrow",text): action="notify"; typ="personal"; rule="personal_request"; conf=.78
        if r.media_type and not text and media["confidence"]<.3: conf=min(conf,.69); rule="uncertain_media"
        relationship={"promotions_opted_out":bool(rel.get("promotions_opted_out_at")),"business_verified":bool(biz and biz.get("verified")=="1"),"active_transaction":rule.startswith("active_transaction"),"messages_dismissed_30d":int(rel.get("messages_dismissed_30d","0") or 0)}
        reason=generate_reason(ReasonContext(rule,media["status"],relationship,action,typ,evidence,f"@{r.user_id}" in text,historical_action))
        return {"message_id":r.message_id,"action":action,"message_type":typ,"reason":reason,"confidence":round(min(.98,max(.5,conf)),2),"evidence_message_ids":evidence}

    def _evidence(self,top:pd.DataFrame,r:pd.Series)->str:
        good=top[(top.score>=.65)&((top.user_id==r.user_id)|(top.similarity>=.48))].head(3).message_id.tolist(); return ";".join(good) if good else "none"

    @staticmethod
    def _urgent(t:str,r:pd.Series,g:dict[str,str])->bool:
        deadline=bool(re.search(r"\b(now|today|tonight|immediately|urgent|in \d+ minutes|by \d|before (standup|morning)|closes? (this|at)|leaving .*early|fire|missing person)\b|अभी|तुरंत|आज",t))
        direct=(f"@{r.user_id}" in t or bool(re.search(r"can you|need (your|quick)|please (send|come|reply|check|submit|sign)",t)))
        neg=bool(re.search(r"no urgency|not urgent|no need to respond|whenever convenient|tomorrow morning",t))
        operational=g.get("group_type") in {"school_group","coworker","society","caregiving","safety","college_faculty"}
        return not neg and (direct and deadline or deadline and operational)

    @staticmethod
    def _type(t:str,r:pd.Series,b:dict[str,str]|None)->str:
        if re.search(r"good morning|good night|bless|wishes|happy birthday",t): return "greeting"
        if int(r.forwarded_count or 0)>=4 or re.search(r"forward(ed)? as received|share with (ten|all)|send this to",t): return "forward"
        if re.search(r"sale|offer|discount|cashback|coupon|promo|% off|selling|for sale|deal|token today",t): return "promotion"
        if re.search(r"fee|invoice|bill|payment (due|reminder)|maintenance.*pay|premium due",t): return "payment"
        if re.search(r"meeting|class|school|trip|appointment|booking|invite|rsvp|timing|schedule|deadline|portal|bus|event|match",t): return "event"
        if r.conversation_type=="business" or re.search(r"delivery|pickup|refund|statement|ride update|account update|order",t): return "business_update"
        if r.conversation_type=="personal": return "personal"
        return "unknown"
