"""Hierarchical safety, personalization, retrieval, and content ensemble."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .config import OUTPUT_COLUMNS
from .context_features import ContextFeatureLayer, MessageContext
from .data_loader import DatasetBundle
from .data_loader import normalize_text
from .media_processing import MediaProcessor
from .retrieval import Retriever
from .security import assess

class Router:
    def __init__(self,data:DatasetBundle,cache:Path):
        self.data=data; t=data.tables; self.retriever=Retriever(data.history_events); self.media=MediaProcessor(data.root,cache)
        self.users=t["users"].set_index("user_id").to_dict("index"); self.groups=t["groups"].set_index("group_id").to_dict("index")
        self.members=t["group_members"].set_index(["group_id","user_id"]).to_dict("index"); self.businesses=t["business_accounts"].set_index("business_id").to_dict("index")
        self.context=ContextFeatureLayer(t["users"],t["daily_notification_summary"],t["group_members"])
        self.relations=t["user_business_history"].set_index(["user_id","business_id"]).to_dict("index")
        self.media_paths={**t["images"].set_index("image_id").file_path.to_dict(),**t["voice_notes"].set_index("voice_note_id").file_path.to_dict()}

    def route_all(self)->pd.DataFrame: return pd.DataFrame([self.route(r) for _,r in self.data.tables["messages"].iterrows()],columns=OUTPUT_COLUMNS)

    def route(self,r:pd.Series)->dict[str,object]:
        text=r.normalized_text; biz=self.businesses.get(r.business_id); rel=self.relations.get((r.user_id,r.business_id),{}); context=self.context.for_message(r); member=context.recipient_group; group=self.groups.get(r.group_id,{})
        caption=r.normalized_text; biz=self.businesses.get(r.business_id); rel=self.relations.get((r.user_id,r.business_id),{}); member=self.members.get((r.group_id,r.user_id),{}); group=self.groups.get(r.group_id,{})
        media=self.media.extract(r.media_type,r.media_id,self.media_paths.get(r.media_id,""))
        risk=assess(text,biz); top=self.retriever.top(r); best=top.iloc[0]
        typ=self._type(text,r,biz); action="digest"; reason="Safe content with no immediate action can be reviewed later."; conf=.72; decisive_rule="fallback"
        text=normalize_text(" ".join(part for part in (caption, media.get("text", "")) if part)).lower()
        enriched=r.copy(); enriched["normalized_text"]=text
        risk=assess(text,biz); top=self.retriever.top(enriched); best=top.iloc[0]; evidence=self._evidence(top,r)
        typ=self._type(text,r,biz); action="digest"; reason="Safe content with no immediate action can be reviewed later."; conf=.72
        if risk.score:
            action,typ,reason,conf,decisive_rule="mute",risk.kind,risk.reason,risk.score,"security"
        elif typ=="promotion" and (rel.get("promotions_opted_out_at") or rel.get("allows_promotions")=="0" and int(rel.get("messages_dismissed_30d","0") or 0)>=4):
            action,reason,conf,decisive_rule="mute","The user opted out of or repeatedly dismissed similar marketing.",.88,"promotion_opt_out"
        elif typ in {"greeting","forward"} and (int(r.forwarded_count or 0)>=5 or member.get("group_muted_by_user")=="1"):
            action,reason,conf,decisive_rule="mute","Repeated forwarded social content is low-value for this user.",.86,"low_value_forward"
        elif self._urgent(text,r,group):
            action,typ,reason,conf,decisive_rule="notify",("event" if typ=="event" else "urgent"),"A direct, time-sensitive dependency requires prompt attention.",.88,"urgent_dependency"
        elif typ=="business_update" and rel and re.search(r"delivery|pickup|ride|booking|appointment|order|transaction",text):
            action,reason,conf,decisive_rule="notify","A legitimate update matches the user's active transaction or booking.",.88,"active_transaction"
            action,reason,conf="notify","A legitimate update matches the user's active transaction or booking.",.88
        elif r.conversation_type=="group" and context.sender_is_admin and re.search(r"\b(announcement|notice|reminder|schedule|meeting|class|maintenance|closure|update)\b",text):
            action,reason,conf="notify","An operational announcement comes from a group administrator.",.76
        elif r.conversation_type=="group" and self._direct_mention(text,r):
            action,reason,conf="notify","The recipient is directly mentioned and asked to respond.",.76
        elif best.score>=.78 and best.user_id==r.user_id:
            action=best.weak_action; conf=min(.89,.68+.16*float(best.similarity)); decisive_rule="behavioral_analogue"; reason={"notify":"Similar messages previously received quick engagement from this user.","digest":"Similar useful messages were opened later without urgent engagement.","mute":"Similar messages were previously dismissed or suppressed by this user."}[action]
        elif typ=="promotion":
            action="digest" if rel.get("allows_promotions")=="1" or not rel else "mute"; reason="The legitimate offer can be reviewed later." if action=="digest" else "Low-priority marketing does not match the user's preferences."; conf=.79; decisive_rule="promotion_preference"
        elif typ in {"spam","scam"}: action="mute"; reason="Suspicious or unwanted content should be suppressed."; conf=.84; decisive_rule="security"
        elif r.conversation_type=="personal" and re.search(r"\?|can you|could you|please",text) and not re.search(r"no urgency|not urgent|tomorrow",text): action="notify"; typ="personal"; reason="A personal sender asks for a direct response."; conf=.78; decisive_rule="direct_question"
        if r.media_type and not text and media["confidence"]<.3: conf=min(conf,.69); reason="Media extraction is uncertain; contextual history supports this conservative route."
        evidence=self.retriever.evidence(r,action,typ,decisive_rule)
        return {"message_id":r.message_id,"action":action,"message_type":typ,"reason":reason,"confidence":round(min(.98,max(.5,conf)),2),"evidence_message_ids":evidence}

            action="digest" if rel.get("allows_promotions")=="1" or not rel else "mute"; reason="The legitimate offer can be reviewed later." if action=="digest" else "Low-priority marketing does not match the user's preferences."; conf=.79
        elif typ in {"spam","scam"}: action="mute"; reason="Suspicious or unwanted content should be suppressed."; conf=.84
        elif r.conversation_type=="personal" and re.search(r"\?|can you|could you|please",text) and not re.search(r"no urgency|not urgent|tomorrow",text): action="notify"; typ="personal"; reason="A personal sender asks for a direct response."; conf=.78
        action,reason,conf=self._apply_interruption_context(action,reason,conf,context,self._urgent(text,r,group),self._direct_mention(text,r))
        if r.media_type and not text and media["confidence"]<.3: conf=min(conf,.69); reason="Media extraction is uncertain; contextual history supports this conservative route."
        if r.media_type and not caption and media["confidence"]<.3: conf=min(conf,.69); reason="Media extraction is uncertain; contextual history supports this conservative route."
        return {"message_id":r.message_id,"action":action,"message_type":typ,"reason":reason,"confidence":round(min(.98,max(.5,conf)),2),"evidence_message_ids":evidence}

    @staticmethod
    def _apply_interruption_context(action:str,reason:str,probability:float,context:MessageContext,genuine_urgency:bool,direct_mention:bool)->tuple[str,str,float]:
        """Calibrate interruption probability without turning context into mute rules."""
        if action!="notify" or genuine_urgency:
            return action,reason,probability
        reduction=(.16 if context.in_dnd_window else 0)+(.24*context.fatigue_score)
        if context.recipient_group_muted and not direct_mention:
            reduction+=.12
        adjusted=max(.5,probability-reduction)
        if adjusted<.67:
            detail="quiet hours" if context.in_dnd_window else "recent notification fatigue"
            return "digest",f"Useful but low-priority content is deferred because of {detail}.",1-adjusted
        return action,reason,adjusted

    @staticmethod
    def _direct_mention(t:str,r:pd.Series)->bool:
        return f"@{r.user_id}" in t and bool(re.search(r"\?|can you|could you|please|need (your|you)",t))

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
