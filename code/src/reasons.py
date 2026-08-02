"""Controlled, context-aware explanations for routing decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping


@dataclass(frozen=True)
class ReasonContext:
    """Structured facts that a reason is allowed to claim."""

    decisive_rule: str
    media_status: str
    relationship: Mapping[str, object] = field(default_factory=dict)
    action: str = "digest"
    message_type: str = "unknown"
    evidence: str = "none"
    direct_mention: bool = False
    historical_action: str | None = None


_TEMPLATES = {
    "safety": "Safety checks identified suspicious or unsafe content, so it was suppressed.",
    "promotion_opt_out": "The user opted out of promotions, so this marketing message was suppressed.",
    "promotion_dismissed": "The user repeatedly dismissed marketing messages, so this promotion was suppressed.",
    "forward_or_muted_group": "Forwarded or muted-group social content is low-value for this user.",
    "urgent_direct": "A direct mention with a time-sensitive request requires prompt attention.",
    "urgent_operational": "A time-sensitive operational update requires prompt attention.",
    "active_transaction_verified": "This verified business update matches the user's active transaction or booking.",
    "active_transaction": "This business update matches the user's active transaction or booking.",
    "history_notify": "Historical support shows similar messages received quick engagement from this user.",
    "history_digest": "Historical support shows similar useful messages were opened later.",
    "history_mute": "Historical dismissal of similar messages supports suppressing this one.",
    "promotion_digest": "This legitimate offer can be reviewed later.",
    "promotion_mute": "This low-priority promotion does not match the user's preferences.",
    "spam": "Safety checks identified suspicious or unwanted content, so it was suppressed.",
    "personal_request": "A personal sender asks for a response without an explicit delay cue.",
    "default": "Safe content with no immediate action can be reviewed later.",
    "uncertain_media": "Media extraction is uncertain, so the message was routed conservatively.",
}


def validate_reason(reason: str, context: ReasonContext) -> None:
    """Reject claims that are unsupported by the supplied structured facts."""
    text = reason.lower()
    relationship = context.relationship
    checks = (
        (r"opt(?:ed)? out", bool(relationship.get("promotions_opted_out")), "opt-out"),
        (r"\bverified\b", bool(relationship.get("business_verified")), "verification"),
        (r"active transaction|active booking", bool(relationship.get("active_transaction")), "active transaction"),
        (r"direct mention", context.direct_mention, "direct mention"),
        (r"historical dismissal|previously dismissed", context.historical_action == "mute" and context.evidence != "none", "historical dismissal"),
        (r"historical support|history supports", context.evidence != "none", "historical support"),
    )
    for pattern, supported, claim in checks:
        if re.search(pattern, text) and not supported:
            raise ValueError(f"reason claims {claim} without structured support")


def generate_reason(context: ReasonContext) -> str:
    """Render and validate a controlled reason for one decisive rule."""
    rule = context.decisive_rule
    if rule.startswith("history_") and context.evidence == "none":
        rule = "uncertain_media" if context.media_status not in {"not_applicable", "image_inspected"} else "default"
    reason = _TEMPLATES[rule]
    validate_reason(reason, context)
    return reason
