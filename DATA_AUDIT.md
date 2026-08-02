# Data Audit

## messages.csv
Shape: (110, 12)
Columns: message_id, user_id, conversation_type, group_id, business_id, sender_user_id, created_at, message_text, media_type, media_id, forwarded_count, normalized_text
Null/empty: message_id=0, user_id=0, conversation_type=0, group_id=47, business_id=80, sender_user_id=30, created_at=0, message_text=8, media_type=87, media_id=87, forwarded_count=0, normalized_text=8

## sample_messages.csv
Shape: (30, 16)
Columns: message_id, user_id, conversation_type, group_id, business_id, sender_user_id, created_at, message_text, media_type, media_id, forwarded_count, action, message_type, reason, confidence, evidence_message_ids
Null/empty: message_id=0, user_id=0, conversation_type=0, group_id=13, business_id=22, sender_user_id=8, created_at=0, message_text=3, media_type=22, media_id=22, forwarded_count=0, action=0, message_type=0, reason=0, confidence=0, evidence_message_ids=0

## users.csv
Shape: (54, 6)
Columns: user_id, do_not_disturb_window, messages_opened_30d, messages_replied_30d, notifications_dismissed_30d, messages_reported_30d
Null/empty: user_id=0, do_not_disturb_window=0, messages_opened_30d=0, messages_replied_30d=0, notifications_dismissed_30d=0, messages_reported_30d=0

## groups.csv
Shape: (23, 7)
Columns: group_id, group_name, group_type, member_count, admin_count, created_at, messages_30d
Null/empty: group_id=0, group_name=0, group_type=0, member_count=0, admin_count=0, created_at=0, messages_30d=0

## group_members.csv
Shape: (401, 9)
Columns: group_id, user_id, role, joined_at, messages_sent_30d, messages_read_30d, replies_sent_30d, notifications_dismissed_30d, group_muted_by_user
Null/empty: group_id=0, user_id=0, role=0, joined_at=0, messages_sent_30d=0, messages_read_30d=0, replies_sent_30d=0, notifications_dismissed_30d=0, group_muted_by_user=0

## business_accounts.csv
Shape: (110, 11)
Columns: business_id, display_name, brand_name, category, verified, official_domain, domain_used_by_sender, account_age_days, messages_sent_30d, user_reports_30d, domain_used_by_sender_age_days
Null/empty: business_id=0, display_name=0, brand_name=0, category=0, verified=0, official_domain=5, domain_used_by_sender=1, account_age_days=0, messages_sent_30d=0, user_reports_30d=0, domain_used_by_sender_age_days=0

## user_business_history.csv
Shape: (106, 11)
Columns: user_id, business_id, why_user_knows_account, last_activity_at, allows_promotions, promotions_opted_out_at, activity_count_180d, messages_opened_30d, messages_dismissed_30d, messages_replied_30d, last_reply_at
Null/empty: user_id=0, business_id=0, why_user_knows_account=0, last_activity_at=0, allows_promotions=0, promotions_opted_out_at=92, activity_count_180d=0, messages_opened_30d=0, messages_dismissed_30d=0, messages_replied_30d=0, last_reply_at=42

## message_history.csv
Shape: (412, 12)
Columns: message_id, user_id, conversation_type, group_id, business_id, sender_user_id, created_at, message_text, media_type, media_id, forwarded_count, normalized_text
Null/empty: message_id=0, user_id=0, conversation_type=0, group_id=204, business_id=257, sender_user_id=155, created_at=0, message_text=4, media_type=389, media_id=389, forwarded_count=0, normalized_text=4

## message_events.csv
Shape: (412, 8)
Columns: user_id, message_id, message_opened, message_replied, reaction_time_minutes, notification_dismissed, muted_after_message, message_reported
Null/empty: user_id=0, message_id=0, message_opened=0, message_replied=0, reaction_time_minutes=134, notification_dismissed=0, muted_after_message=0, message_reported=0

## images.csv
Shape: (20, 2)
Columns: image_id, file_path
Null/empty: image_id=0, file_path=0

## voice_notes.csv
Shape: (13, 2)
Columns: voice_note_id, file_path
Null/empty: voice_note_id=0, file_path=0

## daily_notification_summary.csv
Shape: (756, 4)
Columns: user_id, date, notifications_sent, notifications_dismissed
Null/empty: user_id=0, date=0, notifications_sent=0, notifications_dismissed=0

## Behavioral join
History/event joined rows: 412
Event signatures:
message_opened  message_replied  reaction_time_minutes  notification_dismissed  muted_after_message  message_reported
0               0                                       1                       1                    0                    79
                                                                                                     1                    55
1               0                120                    0                       0                    0                   110
                                 9                      0                       0                    0                    15
                1                2                      0                       0                    0                   153

## Foreign keys
messages.user_id -> users.user_id: 0 missing
messages.group_id -> groups.group_id: 0 missing
messages.business_id -> business_accounts.business_id: 0 missing
message_events.message_id -> message_history.message_id: 0 missing

## Findings
The event data has deliberately discrete reaction patterns: quick replies support notify; delayed opens support digest; dismiss/mute/report support mute.
Incoming content includes exact and near-duplicate templates, Hindi/Hinglish, prompt injection, suspicious domains, sensitive-code requests, image messages, and voice notes.
Same content is intentionally routed differently across users, so user-business opt-in, group engagement/mute state, and same-user historical reactions are first-class signals.
