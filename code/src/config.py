"""Shared constants."""
ACTIONS = {"notify", "digest", "mute"}
MESSAGE_TYPES = {"personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"}
OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
SEED = 2026
