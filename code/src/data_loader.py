"""Schema-aware loading, normalization, joins, and integrity reporting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata
import pandas as pd

TABLES = ("messages", "sample_messages", "users", "groups", "group_members", "business_accounts", "user_business_history", "message_history", "message_events", "images", "voice_notes", "daily_notification_summary")


def normalize_text(value: object) -> str:
    if pd.isna(value): return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


@dataclass
class DatasetBundle:
    root: Path
    tables: dict[str, pd.DataFrame]

    @classmethod
    def load(cls, root: Path) -> "DatasetBundle":
        tables = {name: pd.read_csv(root / f"{name}.csv", dtype=str, keep_default_na=False) for name in TABLES}
        required = {"messages": {"message_id", "user_id", "conversation_type", "message_text", "media_type", "media_id"}, "message_history": {"message_id", "user_id"}, "message_events": {"message_id", "user_id"}}
        for name, cols in required.items():
            missing = cols - set(tables[name]);
            if missing: raise ValueError(f"{name}.csv missing columns: {sorted(missing)}")
        for name in ("messages", "message_history"):
            tables[name]["normalized_text"] = tables[name]["message_text"].map(normalize_text).str.lower()
        return cls(root, tables)

    def audit(self) -> str:
        lines = ["# Data Audit", ""]
        for name, df in self.tables.items():
            lines += [f"## {name}.csv", f"Shape: {df.shape}", f"Columns: {', '.join(df.columns)}", "Null/empty: " + ", ".join(f"{c}={int((df[c]=='').sum())}" for c in df), ""]
        events = self.tables["message_events"]
        lines += ["## Behavioral join", f"History/event joined rows: {len(self.history_events)}", "Event signatures:", events.groupby(list(events.columns[2:]), dropna=False).size().to_string(), "", "## Foreign keys"]
        checks = [("messages","user_id","users","user_id"),("messages","group_id","groups","group_id"),("messages","business_id","business_accounts","business_id"),("message_events","message_id","message_history","message_id")]
        for s, col, t, key in checks:
            vals=set(self.tables[s][col])-{""}; missing=vals-set(self.tables[t][key]); lines.append(f"{s}.{col} -> {t}.{key}: {len(missing)} missing")
        lines += ["", "## Findings", "The event data has deliberately discrete reaction patterns: quick replies support notify; delayed opens support digest; dismiss/mute/report support mute.", "Incoming content includes exact and near-duplicate templates, Hindi/Hinglish, prompt injection, suspicious domains, sensitive-code requests, image messages, and voice notes.", "Same content is intentionally routed differently across users, so user-business opt-in, group engagement/mute state, and same-user historical reactions are first-class signals."]
        return "\n".join(lines) + "\n"

    @property
    def history_events(self) -> pd.DataFrame:
        return self.tables["message_history"].merge(self.tables["message_events"], on=["message_id", "user_id"], validate="one_to_one")
