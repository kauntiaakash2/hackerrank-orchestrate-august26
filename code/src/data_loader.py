"""Schema-aware loading, normalization, joins, and integrity reporting."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import unicodedata

import pandas as pd

TABLES = (
    "messages", "sample_messages", "users", "groups", "group_members",
    "business_accounts", "user_business_history", "message_history",
    "message_events", "images", "voice_notes", "daily_notification_summary",
)

MESSAGE_COLUMNS = (
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
)


@dataclass(frozen=True)
class TableSchema:
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    composite_keys: tuple[tuple[str, ...], ...] = ()
    categories: dict[str, frozenset[str]] = field(default_factory=dict)
    timestamps: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    nonnegative_integers: tuple[str, ...] = ()


SCHEMAS: dict[str, TableSchema] = {
    "messages": TableSchema(MESSAGE_COLUMNS, ("message_id",), categories={"conversation_type": frozenset({"personal", "group", "business"}), "media_type": frozenset({"", "image", "voice"})}, timestamps=("created_at",), nonnegative_integers=("forwarded_count",)),
    "sample_messages": TableSchema(MESSAGE_COLUMNS + ("action", "message_type", "reason", "confidence", "evidence_message_ids"), ("message_id",), categories={"conversation_type": frozenset({"personal", "group", "business"}), "media_type": frozenset({"", "image", "voice"}), "action": frozenset({"notify", "digest", "mute"})}, timestamps=("created_at",), nonnegative_integers=("forwarded_count",)),
    "users": TableSchema(("user_id", "do_not_disturb_window", "messages_opened_30d", "messages_replied_30d", "notifications_dismissed_30d", "messages_reported_30d"), ("user_id",), nonnegative_integers=("messages_opened_30d", "messages_replied_30d", "notifications_dismissed_30d", "messages_reported_30d")),
    "groups": TableSchema(("group_id", "group_name", "group_type", "member_count", "admin_count", "created_at", "messages_30d"), ("group_id",), timestamps=("created_at",), nonnegative_integers=("member_count", "admin_count", "messages_30d")),
    "group_members": TableSchema(("group_id", "user_id", "role", "joined_at", "messages_sent_30d", "messages_read_30d", "replies_sent_30d", "notifications_dismissed_30d", "group_muted_by_user"), ("group_id", "user_id"), composite_keys=(("group_id", "user_id"),), categories={"role": frozenset({"admin", "member"})}, timestamps=("joined_at",), booleans=("group_muted_by_user",), nonnegative_integers=("messages_sent_30d", "messages_read_30d", "replies_sent_30d", "notifications_dismissed_30d")),
    "business_accounts": TableSchema(("business_id", "display_name", "brand_name", "category", "verified", "official_domain", "domain_used_by_sender", "account_age_days", "messages_sent_30d", "user_reports_30d", "domain_used_by_sender_age_days"), ("business_id",), booleans=("verified",), nonnegative_integers=("account_age_days", "messages_sent_30d", "user_reports_30d", "domain_used_by_sender_age_days")),
    "user_business_history": TableSchema(("user_id", "business_id", "why_user_knows_account", "last_activity_at", "allows_promotions", "promotions_opted_out_at", "activity_count_180d", "messages_opened_30d", "messages_dismissed_30d", "messages_replied_30d", "last_reply_at"), ("user_id", "business_id"), composite_keys=(("user_id", "business_id"),), timestamps=("last_activity_at", "promotions_opted_out_at", "last_reply_at"), booleans=("allows_promotions",), nonnegative_integers=("activity_count_180d", "messages_opened_30d", "messages_dismissed_30d", "messages_replied_30d")),
    "message_history": TableSchema(MESSAGE_COLUMNS, ("message_id",), categories={"conversation_type": frozenset({"personal", "group", "business"}), "media_type": frozenset({"", "image", "voice"})}, timestamps=("created_at",), nonnegative_integers=("forwarded_count",)),
    "message_events": TableSchema(("user_id", "message_id", "message_opened", "message_replied", "reaction_time_minutes", "notification_dismissed", "muted_after_message", "message_reported"), ("user_id", "message_id"), composite_keys=(("user_id", "message_id"),), booleans=("message_opened", "message_replied", "notification_dismissed", "muted_after_message", "message_reported"), nonnegative_integers=("reaction_time_minutes",)),
    "images": TableSchema(("image_id", "file_path"), ("image_id",)),
    "voice_notes": TableSchema(("voice_note_id", "file_path"), ("voice_note_id",)),
    "daily_notification_summary": TableSchema(("user_id", "date", "notifications_sent", "notifications_dismissed"), ("user_id", "date"), composite_keys=(("user_id", "date"),), timestamps=("date",), nonnegative_integers=("notifications_sent", "notifications_dismissed")),
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    table: str
    message: str
    rows: tuple[int, ...] = ()


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, severity: Literal["error", "warning"], code: str, table: str, message: str, rows=()) -> None:
        self.issues.append(ValidationIssue(severity, code, table, message, tuple(int(row) + 2 for row in rows)))

    def as_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "error_count": len(self.errors), "warning_count": len(self.warnings), "issues": [issue.__dict__ for issue in self.issues]}

    def __str__(self) -> str:
        return "\n".join(f"{i.severity.upper()} [{i.code}] {i.table}.csv: {i.message}" + (f" (CSV rows {list(i.rows)})" if i.rows else "") for i in self.issues) or "Dataset validation passed."


class DatasetValidationError(ValueError):
    """Fatal dataset contract violations, with a machine-readable report."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(str(report))


def _bad_rows(series: pd.Series, predicate) -> list[int]:
    return series.index[predicate(series)].tolist()


def _validate_schema(name: str, frame: pd.DataFrame, report: ValidationReport) -> bool:
    schema = SCHEMAS[name]
    missing = [column for column in schema.columns if column not in frame.columns]
    if missing:
        report.add("error", "missing_columns", name, f"Add required column(s): {', '.join(missing)}.")
        return False
    for key in dict.fromkeys((schema.primary_key,) + schema.composite_keys):
        blank = frame[list(key)].eq("").any(axis=1)
        if blank.any():
            report.add("error", "blank_key", name, f"Key {key} cannot contain empty values.", frame.index[blank])
        duplicate = frame.duplicated(list(key), keep=False)
        if duplicate.any():
            report.add("error", "duplicate_key", name, f"Key {key} must be unique; remove or merge duplicate records.", frame.index[duplicate])
    for column, allowed in schema.categories.items():
        bad = ~frame[column].isin(allowed)
        if bad.any():
            values = sorted(frame.loc[bad, column].unique())
            report.add("error", "invalid_category", name, f"{column} contains {values}; allowed values are {sorted(allowed)}.", frame.index[bad])
    for column in schema.booleans:
        bad = ~frame[column].isin({"0", "1"})
        if bad.any():
            report.add("error", "invalid_boolean", name, f"{column} must contain only 0 or 1.", frame.index[bad])
    for column in schema.timestamps:
        present = frame[column].ne("")
        parsed = pd.to_datetime(frame.loc[present, column], errors="coerce", format="mixed")
        bad_rows = parsed.index[parsed.isna()]
        if len(bad_rows):
            report.add("error", "invalid_timestamp", name, f"{column} contains timestamps that cannot be parsed.", bad_rows)
    for column in schema.nonnegative_integers:
        present = frame[column].ne("")
        numeric = pd.to_numeric(frame.loc[present, column], errors="coerce")
        bad_rows = numeric.index[numeric.isna() | numeric.lt(0) | numeric.mod(1).ne(0)]
        if len(bad_rows):
            report.add("error", "invalid_nonnegative_integer", name, f"{column} must be a nonnegative integer when present.", bad_rows)
    return True


def _foreign_key(report: ValidationReport, tables: dict[str, pd.DataFrame], source: str, column: str, target: str, key: str, *, warning: bool = False) -> None:
    if column not in tables[source] or key not in tables[target]:
        return
    values = tables[source][column]
    bad = values.ne("") & ~values.isin(set(tables[target][key]))
    if bad.any():
        severity = "warning" if warning else "error"
        action = "Optional personalization context is unavailable" if warning else f"Add the referenced {target} record or correct {column}"
        report.add(severity, "missing_optional_relationship" if warning else "foreign_key", source, f"{column} has no match in {target}.{key}. {action}.", tables[source].index[bad])


def _validate_conversations(name: str, frame: pd.DataFrame, report: ValidationReport) -> None:
    needed = set(MESSAGE_COLUMNS)
    if not needed.issubset(frame.columns):
        return
    rules = {
        "personal": ({"sender_user_id"}, {"group_id", "business_id"}),
        "group": ({"group_id", "sender_user_id"}, {"business_id"}),
        "business": ({"business_id"}, {"group_id", "sender_user_id"}),
    }
    for kind, (required, empty) in rules.items():
        selected = frame["conversation_type"].eq(kind)
        bad = selected & (frame[list(required)].eq("").any(axis=1) | frame[list(empty)].ne("").any(axis=1))
        if bad.any():
            report.add("error", "conversation_nullability", name, f"{kind} rows require {sorted(required)} and require {sorted(empty)} to be empty.", frame.index[bad])
    media_missing = frame["media_type"].eq("") != frame["media_id"].eq("")
    if media_missing.any():
        report.add("error", "media_nullability", name, "media_type and media_id must either both be empty or both be populated.", frame.index[media_missing])


def _validate_media(root: Path, tables: dict[str, pd.DataFrame], report: ValidationReport) -> None:
    root = root.resolve()
    mappings = {"image": ("images", "image_id"), "voice": ("voice_notes", "voice_note_id")}
    for table, key in (("images", "image_id"), ("voice_notes", "voice_note_id")):
        if not {key, "file_path"}.issubset(tables[table]):
            continue
        for row, path_text in tables[table]["file_path"].items():
            candidate = (root / path_text).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                report.add("error", "media_path_traversal", table, f"file_path {path_text!r} escapes the dataset directory; use a relative in-dataset path.", [row])
                continue
            if not candidate.is_file():
                report.add("error", "missing_media_file", table, f"Mapped file {path_text!r} does not exist or is not a regular file.", [row])
    for source in ("messages", "sample_messages", "message_history"):
        if not {"media_type", "media_id"}.issubset(tables[source]):
            continue
        for media_type, (target, key) in mappings.items():
            selected = tables[source]["media_type"].eq(media_type)
            bad = selected & ~tables[source]["media_id"].isin(set(tables[target].get(key, [])))
            if bad.any():
                report.add("error", "media_type_mismatch", source, f"{media_type} media_id must reference {target}.{key}; correct the type or mapping.", tables[source].index[bad])


def validate_dataset(root: Path, tables: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    complete = {name: _validate_schema(name, tables[name], report) for name in TABLES}
    for name in ("messages", "sample_messages"):
        _validate_conversations(name, tables[name], report)

    for source in ("messages", "sample_messages", "message_history"):
        _foreign_key(report, tables, source, "user_id", "users", "user_id")
        _foreign_key(report, tables, source, "group_id", "groups", "group_id")
        _foreign_key(report, tables, source, "business_id", "business_accounts", "business_id")
        _foreign_key(report, tables, source, "sender_user_id", "users", "user_id")
    for source, column, target, key in (
        ("group_members", "group_id", "groups", "group_id"), ("group_members", "user_id", "users", "user_id"),
        ("user_business_history", "user_id", "users", "user_id"), ("user_business_history", "business_id", "business_accounts", "business_id"),
        ("message_events", "user_id", "users", "user_id"), ("message_events", "message_id", "message_history", "message_id"),
        ("daily_notification_summary", "user_id", "users", "user_id"),
    ):
        _foreign_key(report, tables, source, column, target, key)

    if complete["message_events"] and complete["message_history"]:
        history_pairs = set(map(tuple, tables["message_history"][["user_id", "message_id"]].to_numpy()))
        bad = ~tables["message_events"][["user_id", "message_id"]].apply(tuple, axis=1).isin(history_pairs)
        if bad.any():
            report.add("error", "event_history_pair", "message_events", "The (user_id, message_id) pair must match the message_history owner.", tables["message_events"].index[bad])

    memberships = set(map(tuple, tables["group_members"][["group_id", "user_id"]].to_numpy())) if complete["group_members"] else set()
    relationships = set(map(tuple, tables["user_business_history"][["user_id", "business_id"]].to_numpy())) if complete["user_business_history"] else set()
    for source in ("messages", "sample_messages", "message_history"):
        if not complete[source]:
            continue
        group_rows = tables[source]["conversation_type"].eq("group")
        for column in ("user_id", "sender_user_id"):
            bad = group_rows & ~tables[source][["group_id", column]].apply(tuple, axis=1).isin(memberships)
            if bad.any():
                report.add("warning", "missing_optional_relationship", source, f"No group_members row exists for ({'group_id'}, {column}); group personalization will be limited.", tables[source].index[bad])
        business_rows = tables[source]["conversation_type"].eq("business")
        bad = business_rows & ~tables[source][["user_id", "business_id"]].apply(tuple, axis=1).isin(relationships)
        if bad.any():
            report.add("warning", "missing_optional_relationship", source, "No user_business_history row exists for this user/business pair; business personalization will be limited.", tables[source].index[bad])

    _validate_media(root, tables, report)
    return report


@dataclass
class DatasetBundle:
    root: Path
    tables: dict[str, pd.DataFrame]
    validation_report: ValidationReport = field(default_factory=ValidationReport)

    @classmethod
    def load(cls, root: Path) -> "DatasetBundle":
        root = Path(root)
        tables: dict[str, pd.DataFrame] = {}
        report = ValidationReport()
        for name in TABLES:
            path = root / f"{name}.csv"
            if not path.is_file():
                report.add("error", "missing_table", name, f"Required file {path.name} does not exist.")
                tables[name] = pd.DataFrame()
                continue
            tables[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
        if any(frame.empty and not len(frame.columns) for frame in tables.values()):
            raise DatasetValidationError(report)
        report.issues.extend(validate_dataset(root, tables).issues)
        if not report.valid:
            raise DatasetValidationError(report)
        for name in ("messages", "message_history"):
            tables[name]["normalized_text"] = tables[name]["message_text"].map(normalize_text).str.lower()
        return cls(root.resolve(), tables, report)

    def audit(self) -> str:
        lines = ["# Data Audit", "", f"Validation: {len(self.validation_report.errors)} errors, {len(self.validation_report.warnings)} warnings", ""]
        lines.extend(f"- {issue}" for issue in self.validation_report.issues)
        for name, frame in self.tables.items():
            lines += [f"## {name}.csv", f"Shape: {frame.shape}", f"Columns: {', '.join(frame.columns)}", "Null/empty: " + ", ".join(f"{column}={int((frame[column] == '').sum())}" for column in frame), ""]
        return "\n".join(lines) + "\n"

    @property
    def history_events(self) -> pd.DataFrame:
        return self.tables["message_history"].merge(self.tables["message_events"], on=["message_id", "user_id"], validate="one_to_one")
