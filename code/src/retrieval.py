"""Personalized historical retrieval, weak supervision, and evidence selection."""
from __future__ import annotations

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _number(value: object, default: float = 999.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def weak_action(row: pd.Series) -> str:
    """Derive a deliberately conservative action from an observed reaction."""
    if row.message_reported == "1" or row.muted_after_message == "1" or row.notification_dismissed == "1":
        return "mute"
    if row.message_replied == "1" and _number(row.reaction_time_minutes) <= 9:
        return "notify"
    return "digest"


def _security_features(text: str) -> set[str]:
    patterns = {
        "credential": r"\b(otp|pin|cvv|password|login code|verification code|card details|bank details)\b|ओटीपी|पासवर्ड",
        "pressure": r"\b(block|suspend|expire|immediately|within \d+|urgent)\b|अभी|तुरंत|बंद",
        "payment": r"\b(qr|pay|payment|refund|wallet|prize|reward|token)\b",
        "injection": r"ignore (all |the )?(previous|prior)|set action|confidence\s*=|system note",
        "medical": r"\b(cure|cancer|diabetes|stop medicine|doctors? hide)\b|इलाज",
    }
    value = (text or "").lower()
    return {name for name, pattern in patterns.items() if re.search(pattern, value)}


class Retriever:
    def __init__(self, history: pd.DataFrame):
        self.history = history.copy()
        self.history["weak_action"] = self.history.apply(weak_action, axis=1)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform(self.history.normalized_text.fillna(""))

    def _similarities(self, row: pd.Series) -> pd.DataFrame:
        query = self.vectorizer.transform([row.normalized_text])
        result = self.history.copy()
        result["similarity"] = cosine_similarity(query, self.matrix)[0]
        return result

    def top(self, row: pd.Series, n: int = 3) -> pd.DataFrame:
        """Return action analogues; retained separately from final evidence ranking."""
        history = self._similarities(row)
        history["score"] = (
            history.similarity
            + .42 * (history.user_id == row.user_id)
            + .22 * ((history.sender_user_id == row.sender_user_id) & (row.sender_user_id != ""))
            + .22 * ((history.group_id == row.group_id) & (row.group_id != ""))
            + .22 * ((history.business_id == row.business_id) & (row.business_id != ""))
        )
        return history.sort_values(["score", "created_at"], ascending=False).head(n)

    def evidence(
        self,
        row: pd.Series,
        final_action: str,
        final_message_type: str,
        decisive_rule: str,
    ) -> str:
        """Select up to three records that support, rather than merely resemble, a decision."""
        history = self._similarities(row)
        same_user = history.user_id == row.user_id
        same_sender = (row.sender_user_id != "") & (history.sender_user_id == row.sender_user_id)
        same_group = (row.group_id != "") & (history.group_id == row.group_id)
        same_business = (row.business_id != "") & (history.business_id == row.business_id)
        context = same_sender | same_group | same_business
        compatible = history.weak_action == final_action

        # High textual similarity is not evidence when the observed reaction implies
        # the opposite action. Same-user provenance is mandatory unless both semantic
        # and operational context are unusually strong.
        eligible = compatible & (same_user | (context & (history.similarity >= .72)))
        incoming_security = _security_features(row.normalized_text)
        security_override = final_action == "mute" and (final_message_type in {"scam", "spam"} or decisive_rule == "security")
        if security_override:
            feature_overlap = history.normalized_text.map(lambda value: len(incoming_security & _security_features(value)))
            negative_reaction = (
                (history.message_reported == "1")
                | (history.muted_after_message == "1")
                | (history.notification_dismissed == "1")
            )
            eligible &= negative_reaction & (feature_overlap > 0)
        else:
            feature_overlap = pd.Series(0, index=history.index)

        reaction_time = history.reaction_time_minutes.map(_number)
        if final_action == "notify":
            positive_reaction = (history.message_replied == "1") | ((history.message_opened == "1") & (reaction_time <= 9))
            eligible &= positive_reaction & context
        elif final_action == "digest":
            delayed_clean_open = (
                (history.message_opened == "1")
                & (reaction_time > 9)
                & (history.notification_dismissed != "1")
                & (history.muted_after_message != "1")
                & (history.message_reported != "1")
            )
            eligible &= delayed_clean_open

        candidates = history.loc[eligible].copy()
        if candidates.empty:
            return "none"
        candidate_context = context.loc[candidates.index]
        candidates["evidence_score"] = (
            2.0 * same_user.loc[candidates.index]
            + .55 * candidate_context
            + 1.15 * candidates.similarity
            + .35 * (candidates.normalized_text.str.replace(r"\d+", "#", regex=True) == re.sub(r"\d+", "#", row.normalized_text))
            + .25 * feature_overlap.loc[candidates.index]
        )
        candidates["created_at_sort"] = pd.to_datetime(candidates.created_at, errors="coerce")
        # A genuinely supportive record needs personalization plus some semantic or
        # contextual connection; generic same-user history is not enough.
        candidates = candidates[(candidates.similarity >= .28) | candidate_context]
        candidates = candidates.sort_values(["evidence_score", "created_at_sort"], ascending=False).head(3)
        return ";".join(candidates.message_id) if not candidates.empty else "none"
