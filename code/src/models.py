"""Reusable weak-label models and feature switches for offline evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ACTIONS = np.array(["digest", "mute", "notify"])
SAFETY_RE = re.compile(r"\b(otp|cvv|pin|password|verification code|scan.*qr|send money|account blocked)\b|ignore (all |the )?(previous|above) instructions", re.I)
URGENT_RE = re.compile(r"\b(urgent|now|today|tonight|immediately|in \d+ minutes|deadline|fire|missing person)\b|अभी|तुरंत|आज", re.I)
PROMO_RE = re.compile(r"\b(sale|offer|discount|cashback|coupon|promo|% off|deal)\b", re.I)

@dataclass(frozen=True)
class FeatureSwitches:
    personalization: bool = True
    retrieval: bool = True
    safety: bool = True
    media: bool = True
    user_business_history: bool = True
    group_context: bool = True
    notification_load: bool = True


def deterministic_action(row: pd.Series, switches: FeatureSwitches = FeatureSwitches()) -> str:
    """A deliberately small, deterministic action baseline."""
    text = str(row.get("normalized_text", ""))
    if switches.safety and SAFETY_RE.search(text):
        return "mute"
    if switches.user_business_history and (str(row.get("promotions_opted_out_at", "")) or str(row.get("allows_promotions", "")) == "0") and PROMO_RE.search(text):
        return "mute"
    if switches.group_context and str(row.get("group_muted_by_user", "")) == "1" and int(float(row.get("forwarded_count", 0) or 0)) >= 4:
        return "mute"
    if URGENT_RE.search(text) or (str(row.get("conversation_type", "")) == "personal" and "?" in text):
        return "notify"
    return "digest"


def rule_probabilities(frame: pd.DataFrame, switches: FeatureSwitches = FeatureSwitches()) -> np.ndarray:
    out = np.full((len(frame), len(ACTIONS)), .075)
    for i, (_, row) in enumerate(frame.iterrows()):
        out[i, np.where(ACTIONS == deterministic_action(row, switches))[0][0]] = .85
    return out


def text_model(calibrate: bool = True) -> Pipeline:
    base = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=26)
    classifier = CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrate else base
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", max_features=12000)), ("classifier", classifier)])


def metadata_columns(switches: FeatureSwitches = FeatureSwitches()) -> tuple[list[str], list[str]]:
    categorical = ["conversation_type", "media_type", "group_type", "business_category", "business_verified"]
    numeric = ["forwarded_count", "business_reports_30d", "business_account_age_days"]
    if not switches.media: categorical.remove("media_type")
    if not switches.group_context: categorical.remove("group_type")
    if switches.personalization: numeric += ["user_opened_30d", "user_replied_30d", "user_dismissed_30d", "user_reported_30d"]
    if switches.user_business_history: categorical += ["allows_promotions", "has_business_relationship"]; numeric += ["business_activity_count", "business_messages_dismissed"]
    if switches.group_context: categorical += ["group_muted_by_user"]; numeric += ["group_messages_read", "group_replies_sent"]
    if switches.notification_load: numeric += ["load_notifications_sent", "load_notifications_dismissed"]
    return categorical, numeric


def metadata_model(frame: pd.DataFrame, switches: FeatureSwitches = FeatureSwitches(), boosted: bool = False) -> tuple[Pipeline, list[str]]:
    categorical, numeric = metadata_columns(switches)
    categorical = [c for c in categorical if c in frame]
    numeric = [c for c in numeric if c in frame]
    pre = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not boosted))]), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    classifier = HistGradientBoostingClassifier(max_iter=150, learning_rate=.07, max_leaf_nodes=15, l2_regularization=1.0, random_state=26) if boosted else LogisticRegression(max_iter=1200, class_weight="balanced", random_state=26)
    return Pipeline([("features", pre), ("classifier", classifier)]), categorical + numeric


def align_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(ACTIONS)))
    for source, label in enumerate(classes):
        aligned[:, np.where(ACTIONS == label)[0][0]] = probabilities[:, source]
    return aligned
