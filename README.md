# Personalized Multimodal Notification Router

A deterministic, CPU-friendly hybrid router for the HackerRank Orchestrate challenge. It combines a high-precision safety firewall, behavioral weak supervision, personalized TF–IDF retrieval, structured user/group/business context, explicit semantic precedence, and calibrated decision templates. It does **not** require an API key and never treats message content as instructions.

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py --dataset-dir dataset --output dataset/output.csv
python main.py --dataset-dir dataset --output dataset/output.csv --validate-only
pytest -q
python code/evaluate.py
```

Python 3.10+ is supported. Runs are idempotent and use no organizer-only data or incoming ID/order features.

## Architecture

1. **Validated loading:** all participant CSVs are loaded as strings; required schemas are checked; Unicode is NFKC-normalized and whitespace canonicalized. `--audit DATA_AUDIT.md` reports shapes, columns, empty values, event signatures, joins, and foreign-key integrity.
2. **Multimodal layer:** images are opened and inspected with Pillow and cached by content SHA-256. Optional extraction failure is non-fatal. Voice files are likewise hashed and cached; when no ASR is installed the explicit `asr_unavailable` state lowers confidence instead of pretending a transcript exists. Captions and historical same-media records remain usable context. This deliberately reliable fallback can be extended with local OCR/ASR without changing routing.
3. **Security firewall:** sensitive-code/account requests, fake verification, QR-payment pressure, prompt injection, domain mismatch, risky unverified businesses, and unsafe medical forwards override personalization. Message, OCR, and transcript text are always untrusted.
4. **Behavioral weak supervision:** `message_history.csv` joins one-to-one with `message_events.csv`. Reports, post-message mutes, and dismissals imply `mute`; replies within the observed fast reaction modes imply `notify`; delayed opens imply `digest`. This policy follows the dataset's discrete reaction-time distribution rather than an invented generic threshold.
5. **Personalized retrieval:** word/bigram TF–IDF cosine similarity is augmented by same user, sender, group, and business. One to three relevant same-user or high-similarity records are returned; IDs are validated against history.
6. **Hierarchical ensemble:** safety, opt-out/repeated unwanted content, genuine direct urgency, active transaction updates, personalized history, semantic category rules, and conservative fallback run in that order. Quiet hours and group mute state never suppress genuine urgent direct dependencies.
7. **Controlled outputs:** message type has explicit precedence separate from action. Reasons are short grounded templates. Confidence combines override precision, analogue quality, context completeness, and media certainty and never reaches `1.0`.

## Context and personalization

- User engagement and notification fatigue are available from user and daily summaries.
- Group type, membership, mute status, and direct mentions distinguish operational school/work/society messages from noisy forwards.
- Business verification, official/sender-domain agreement, report load, promotion permission/opt-out, dismissals, and active relationship determine whether a legitimate promotion is digested or muted and whether an active transaction notifies.
- Safety always wins: prior banking engagement cannot legitimize an OTP request.

## Models and ablations

`python code/evaluate.py` now builds a programmatic weak-label table from the
history/event join and executes leakage-aware grouped cross-validation. It runs
deterministic rules, retrieval, calibrated TF-IDF logistic regression, metadata
logistic regression, histogram gradient boosting, model-only, and final-hybrid
baselines. Personalization, retrieval, safety, media, user-business history,
group context, and notification-load removals are executable variants rather
than qualitative claims. Results, probability metrics, per-class confusion
matrices, and solved-example message-type accuracy are persisted in
`evaluation/results.json`; see `EVALUATION.md` for details. Embedding and LLM
variants are explicitly recorded as unavailable when their runtime is absent.

## Validation

The validator checks exact column order, exact input order and cardinality, unique IDs, allowed enums, numeric `[0,1]` confidence, concise nonempty reasons, no NaNs, and history-only evidence IDs. Tests cover multilingual scam text, trusted-sender OTP requests, prompt injection, QR pressure, unsafe medical forwards, muted-group direct urgency, explicit non-urgency, and missing media extraction. Output is reloaded with pandas as part of validation.

## Configuration and fallback behavior

See `.env.example`. No secrets are used. Media cache paths are configurable with `--cache`; cache keys include the media digest. Missing/corrupt optional media does not crash a batch. A media-only message without available ASR/OCR receives conservative routing and reduced confidence rather than fabricated interpretation.

## Files and packaging

`main.py` is the root entry point; implementation lives in `code/src/`; tests in `tests/`; the deterministic cache in `cache/`; predictions in `dataset/output.csv`; audit in `DATA_AUDIT.md`; and the conversation log is copied to `chat_transcript`. Create the submission archive locally with the following command. `code.zip` is intentionally ignored by Git because pull-request diff viewers do not support binary archives; the archive is a submission artifact, not source code.

```bash
python -m zipfile -c code.zip main.py code requirements.txt .env.example README.md prompts cache tests DATA_AUDIT.md
```

## Known limitations

The base dependency set does not ship a heavyweight multilingual speech model or system OCR binary. Images are inspected structurally and benefit from their captions/same-media analogues, while unavailable voice transcription is declared and confidence-discounted. Installing a local ASR/OCR adapter would improve novel, captionless media. Weak labels represent observed behavior rather than authoritative notification labels, so safety and operational urgency rules intentionally override them.
