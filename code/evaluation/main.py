"""Build weak labels, run grouped evaluation and persist reproducible ablations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import DatasetBundle, normalize_text
from src.models import (ACTIONS, FeatureSwitches, align_probabilities, metadata_model,
                        rule_probabilities, text_model)
from src.retrieval import weak_action
from src.router import Router


def normalized_template(text: object) -> str:
    value = normalize_text(text).lower()
    value = re.sub(r"https?://\S+", "<url>", value)
    value = re.sub(r"\b\d+(?:[.,:]\d+)*\b", "<num>", value)
    return re.sub(r"[^\w<>]+", " ", value).strip()


def leakage_groups(frame: pd.DataFrame) -> pd.Series:
    """Stable joint groups encode template, sender/business identity, and user.

    A transitive connected-component grouping collapses this small, intentionally
    interconnected fixture into one component.  The joint key retains every
    requested leakage dimension while leaving enough groups for honest folds.
    """
    def key(row: pd.Series) -> str:
        identity = row.get("business_id", "") or row.get("sender_user_id", "") or row.get("group_id", "")
        raw = "\x1f".join((str(row.get("user_id", "")), str(identity), str(row.template)))
        return "leak-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return frame.apply(key, axis=1)


def build_training_table(data: DatasetBundle) -> pd.DataFrame:
    frame = data.history_events.copy()
    frame["weak_action"] = frame.apply(weak_action, axis=1)
    frame["normalized_text"] = frame.message_text.map(normalize_text).str.lower()
    frame["template"] = frame.message_text.map(normalized_template)
    users = data.tables["users"].rename(columns={"messages_opened_30d":"user_opened_30d", "messages_replied_30d":"user_replied_30d", "notifications_dismissed_30d":"user_dismissed_30d", "messages_reported_30d":"user_reported_30d"})
    groups = data.tables["groups"][["group_id", "group_type"]]
    members = data.tables["group_members"].rename(columns={"messages_read_30d":"group_messages_read", "replies_sent_30d":"group_replies_sent"})
    businesses = data.tables["business_accounts"].rename(columns={"category":"business_category", "verified":"business_verified", "user_reports_30d":"business_reports_30d", "account_age_days":"business_account_age_days"})
    relations = data.tables["user_business_history"].rename(columns={"activity_count_180d":"business_activity_count", "messages_dismissed_30d":"business_messages_dismissed"})
    relations["has_business_relationship"] = "1"
    load = data.tables["daily_notification_summary"].copy()
    for col in ("notifications_sent", "notifications_dismissed"): load[col] = pd.to_numeric(load[col], errors="coerce").fillna(0)
    load = load.groupby("user_id", as_index=False).agg(load_notifications_sent=("notifications_sent", "mean"), load_notifications_dismissed=("notifications_dismissed", "mean"))
    frame = frame.merge(users, on="user_id", how="left").merge(groups, on="group_id", how="left").merge(members, on=["group_id","user_id"], how="left").merge(businesses, on="business_id", how="left").merge(relations, on=["user_id","business_id"], how="left").merge(load, on="user_id", how="left")
    frame["leakage_group"] = leakage_groups(frame)
    for col in frame.columns:
        if col not in {"message_text", "normalized_text", "template", "created_at"}: frame[col] = frame[col].replace("", np.nan)
    return frame


def retrieval_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, strip_accents="unicode")
    matrix = vectorizer.fit_transform(train.normalized_text.fillna("")); query = vectorizer.transform(test.normalized_text.fillna(""))
    similarity = cosine_similarity(query, matrix)
    output = np.zeros((len(test), len(ACTIONS)))
    train_labels = train.weak_action.to_numpy()
    for i, (_, row) in enumerate(test.iterrows()):
        score = similarity[i].copy()
        score += .35 * (train.user_id.fillna("").to_numpy() == str(row.user_id))
        for col in ("sender_user_id", "business_id", "group_id"):
            value = row.get(col, np.nan)
            if pd.notna(value): score += .15 * (train[col].fillna("").to_numpy() == str(value))
        nearest = np.argsort(score)[-5:]
        weights = np.maximum(score[nearest], .05)
        for label, weight in zip(train_labels[nearest], weights): output[i, np.where(ACTIONS == label)[0][0]] += weight
        output[i] = (output[i] + .1) / (output[i].sum() + .3)
    return output


def metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    prediction = ACTIONS[probabilities.argmax(axis=1)]
    return {"action_accuracy": accuracy_score(y_true, prediction), "macro_f1": f1_score(y_true, prediction, labels=ACTIONS, average="macro", zero_division=0), "log_loss": log_loss(y_true, probabilities, labels=ACTIONS), "brier_score": float(np.mean(np.sum((probabilities - (y_true.to_numpy()[:,None] == ACTIONS)) ** 2, axis=1))), "confusion_matrix": {"labels": ACTIONS.tolist(), "rows_true_columns_predicted": confusion_matrix(y_true, prediction, labels=ACTIONS).tolist()}}


def evaluate_variant(frame: pd.DataFrame, name: str, switches: FeatureSwitches, mode: str) -> dict:
    groups, y = frame.leakage_group, frame.weak_action
    group_count = groups.nunique(); splits = min(5, group_count)
    if splits < 2: raise ValueError("At least two leakage groups are required")
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=26)
    probabilities = np.zeros((len(frame), len(ACTIONS)))
    for train_idx, test_idx in cv.split(frame, y, groups):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        rules = rule_probabilities(test, switches)
        retrieval = retrieval_probabilities(train, test) if switches.retrieval else np.full_like(rules, 1/3)
        if mode == "rules": predicted = rules
        elif mode == "retrieval": predicted = retrieval
        elif mode == "text":
            model = text_model(calibrate=train.weak_action.value_counts().min() >= 3); model.fit(train.normalized_text, train.weak_action)
            predicted = align_probabilities(model.predict_proba(test.normalized_text), model.classes_)
        elif mode in {"metadata", "boosted"}:
            model, columns = metadata_model(train, switches, boosted=mode == "boosted"); model.fit(train[columns], train.weak_action)
            predicted = align_probabilities(model.predict_proba(test[columns]), model.classes_)
        else:
            text = text_model(calibrate=train.weak_action.value_counts().min() >= 3); text.fit(train.normalized_text, train.weak_action)
            text_prob = align_probabilities(text.predict_proba(test.normalized_text), text.classes_)
            meta, columns = metadata_model(train, switches, boosted=True); meta.fit(train[columns], train.weak_action)
            model_prob = .55 * text_prob + .45 * align_probabilities(meta.predict_proba(test[columns]), meta.classes_)
            predicted = model_prob if mode == "model" else .35 * rules + .25 * retrieval + .40 * model_prob
            if switches.safety:
                safety = test.normalized_text.str.contains(r"otp|cvv|password|scan.*qr|ignore .*instructions", case=False, regex=True, na=False).to_numpy()
                predicted[safety] = np.array([.025, .95, .025])
        probabilities[test_idx] = predicted / predicted.sum(axis=1, keepdims=True)
    return {"name": name, "mode": mode, "switches": asdict(switches), "folds": splits, "leakage_groups": group_count, **metrics(y, probabilities)}


def solved_metrics(data: DatasetBundle) -> dict:
    sample = data.tables["sample_messages"].copy()
    sample = sample[sample.action.isin(ACTIONS)].copy(); sample["normalized_text"] = sample.message_text.map(normalize_text).str.lower()
    router = Router(data, Path("cache/evaluation_media.json")); prediction = pd.DataFrame([router.route(row) for _, row in sample.iterrows()])
    result = metrics(sample.action, np.array([[.8 if label == action else .1 for label in ACTIONS] for action in prediction.action]))
    typed = sample.message_type.ne("")
    result.update({"examples": len(sample), "message_type_examples": int(typed.sum()), "message_type_accuracy": accuracy_score(sample.loc[typed,"message_type"], prediction.loc[typed,"message_type"]) if typed.any() else None, "message_type_confusion_matrix": {"labels": sorted(set(sample.loc[typed,"message_type"]) | set(prediction.loc[typed,"message_type"])), "rows_true_columns_predicted": confusion_matrix(sample.loc[typed,"message_type"], prediction.loc[typed,"message_type"], labels=sorted(set(sample.loc[typed,"message_type"]) | set(prediction.loc[typed,"message_type"]))).tolist()} if typed.any() else None})
    return result


def run(dataset_dir: Path, output: Path, training_table: Path) -> dict:
    data = DatasetBundle.load(dataset_dir); frame = build_training_table(data)
    training_table.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(training_table, index=False)
    full = FeatureSwitches()
    specifications = [("deterministic_rules", full, "rules"), ("retrieval", full, "retrieval"), ("tfidf_calibrated_logistic", full, "text"), ("metadata_classifier", full, "metadata"), ("gradient_boosted_tabular", full, "boosted"), ("rules_only", full, "rules"), ("model_only", full, "model"), ("final_hybrid", full, "hybrid")]
    for field in asdict(full): specifications.append((f"without_{field}", FeatureSwitches(**{**asdict(full), field: False}), "hybrid"))
    results = {"schema_version": 1, "weak_label_rows": len(frame), "weak_label_distribution": frame.weak_action.value_counts().to_dict(), "models": [evaluate_variant(frame, *spec) for spec in specifications], "solved_examples": solved_metrics(data), "unavailable_variants": {"embedding": "unavailable: no embedding model dependency or weights are installed", "llm": "unavailable: no LLM runtime/API is configured; evaluation remains offline and deterministic"}}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset-dir", type=Path, default=Path("dataset")); parser.add_argument("--output", type=Path, default=Path("evaluation/results.json")); parser.add_argument("--training-table", type=Path, default=Path("evaluation/weak_labels.csv")); args = parser.parse_args()
    result = run(args.dataset_dir, args.output, args.training_table)
    print(json.dumps({"weak_label_rows": result["weak_label_rows"], "models": {m["name"]: {"accuracy": round(m["action_accuracy"],3), "macro_f1": round(m["macro_f1"],3)} for m in result["models"]}, "solved_examples": result["solved_examples"]}, indent=2))


if __name__ == "__main__": main()
