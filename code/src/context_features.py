"""Deterministic recipient, notification-load, and group context features."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Mapping

import pandas as pd


def _number(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _clock(value: str) -> time | None:
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MessageContext:
    """Features available at the time an incoming message was received."""

    user_open_rate: float = 0.0
    user_reply_rate: float = 0.0
    user_dismiss_rate: float = 0.0
    rolling_notification_load: int = 0
    recent_dismissal_ratio: float = 0.0
    fatigue_score: float = 0.0
    in_dnd_window: bool = False
    daily_summary_available: bool = False
    recipient_group: Mapping[str, str] = field(default_factory=dict)
    sender_group: Mapping[str, str] = field(default_factory=dict)

    @property
    def recipient_group_muted(self) -> bool:
        return self.recipient_group.get("group_muted_by_user") == "1"

    @property
    def recipient_group_engagement_rate(self) -> float:
        read = _number(self.recipient_group.get("messages_read_30d"))
        dismissed = _number(self.recipient_group.get("notifications_dismissed_30d"))
        return _ratio(read, read + dismissed)

    @property
    def sender_is_admin(self) -> bool:
        return self.sender_group.get("role", "").lower() in {"admin", "owner"}

    @property
    def sender_role(self) -> str:
        return self.sender_group.get("role", "")


class ContextFeatureLayer:
    """Build leakage-safe features from participant-facing context tables.

    Daily summaries are indexed once by ``(user_id, date)``. A seven-day
    calendar window ending on the message date is used; absent rows contribute
    no invented load and leave ``daily_summary_available`` false when the exact
    message date is missing.
    """

    def __init__(self, users: pd.DataFrame, daily: pd.DataFrame, members: pd.DataFrame):
        self.users = users.set_index("user_id").to_dict("index")
        self.members = members.set_index(["group_id", "user_id"]).to_dict("index")
        self.daily_by_user_date: dict[tuple[str, date], dict[str, str]] = {}
        for row in daily.to_dict("records"):
            try:
                day = date.fromisoformat(str(row.get("date", "")))
            except ValueError:
                continue
            self.daily_by_user_date[(str(row.get("user_id", "")), day)] = row

    def for_message(self, message: pd.Series | Mapping[str, object]) -> MessageContext:
        user_id = str(message.get("user_id", ""))
        group_id = str(message.get("group_id", ""))
        sender_id = str(message.get("sender_user_id", ""))
        user = self.users.get(user_id, {})
        timestamp = pd.to_datetime(message.get("created_at", ""), errors="coerce")

        opened = _number(user.get("messages_opened_30d"))
        replied = _number(user.get("messages_replied_30d"))
        dismissed = _number(user.get("notifications_dismissed_30d"))
        handled = opened + dismissed
        load = sent = recent_dismissed = 0.0
        exact_summary = False
        if not pd.isna(timestamp):
            message_day = timestamp.date()
            exact_summary = (user_id, message_day) in self.daily_by_user_date
            for offset in range(7):
                summary = self.daily_by_user_date.get((user_id, message_day - timedelta(days=offset)))
                if summary:
                    day_sent = _number(summary.get("notifications_sent"))
                    load += day_sent
                    sent += day_sent
                    recent_dismissed += _number(summary.get("notifications_dismissed"))

        dismissal_ratio = _ratio(recent_dismissed, sent)
        # Seven notifications/day reaches full load pressure. Dismissal behavior
        # supplies the other half, keeping the score bounded and interpretable.
        load_pressure = min(1.0, load / 49.0)
        fatigue = min(1.0, 0.5 * load_pressure + 0.5 * dismissal_ratio)
        return MessageContext(
            user_open_rate=round(_ratio(opened, handled), 4),
            user_reply_rate=round(_ratio(replied, opened), 4),
            user_dismiss_rate=round(_ratio(dismissed, handled), 4),
            rolling_notification_load=int(load),
            recent_dismissal_ratio=round(dismissal_ratio, 4),
            fatigue_score=round(fatigue, 4),
            in_dnd_window=self._in_dnd(timestamp, str(user.get("do_not_disturb_window", ""))),
            daily_summary_available=exact_summary,
            recipient_group=self.members.get((group_id, user_id), {}),
            sender_group=self.members.get((group_id, sender_id), {}),
        )

    @staticmethod
    def _in_dnd(timestamp: pd.Timestamp, window: str) -> bool:
        if pd.isna(timestamp) or "-" not in window:
            return False
        start, end = (_clock(part) for part in window.split("-", 1))
        if start is None or end is None or start == end:
            return False
        current = timestamp.time()
        return start <= current < end if start < end else current >= start or current < end
