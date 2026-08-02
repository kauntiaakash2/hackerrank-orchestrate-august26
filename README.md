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
2. **Multimodal layer:** images use deterministic grayscale, autocontrast, and threshold OCR variants, choosing the result by OCR confidence. The structured result includes text, dates, payment terms, brand candidates, and QR presence. Voice notes use optional local multilingual `faster-whisper` with fixed greedy decoding, language detection, transcript confidence, and a timeout. Captions are joined with extracted text before safety, retrieval, urgency, and type analysis. Results are cached by content SHA-256 plus extractor version and backend availability, so installing an improved backend invalidates earlier fallback entries.
3. **Security firewall:** sensitive-code/account requests, fake verification, QR-payment pressure, prompt injection, domain mismatch, risky unverified businesses, and unsafe medical forwards override personalization. Message, OCR, and transcript text are always untrusted.
4. **Behavioral weak supervision:** `message_history.csv` joins one-to-one with `message_events.csv`. Reports, post-message mutes, and dismissals imply `mute`; replies within the observed fast reaction modes imply `notify`; delayed opens imply `digest`. This policy follows the dataset's discrete reaction-time distribution rather than an invented generic threshold.
5. **Personalized retrieval:** word/bigram TF–IDF cosine similarity is augmented by same user, sender, group, and business. Evidence is ranked only after the final action and message type are known, and one to three records are returned only when their event-derived action, operational context, and semantics support that decision. Security overrides require matching risk features plus a report/mute/dismiss reaction; contradictory lookalikes are excluded.
6. **Hierarchical ensemble:** safety, opt-out/repeated unwanted content, genuine direct urgency, active transaction updates, personalized history, semantic category rules, and conservative fallback run in that order. Quiet hours and group mute state never suppress genuine urgent direct dependencies.
7. **Controlled and calibrated outputs:** message type has explicit precedence separate from action. Reasons are short grounded templates. `code/src/confidence.py` composes rule strength, probability margin, neighbour similarity/agreement, historical evidence quality, context completeness, media certainty, and cross-component conflict. Evaluation uses user-grouped held-out predictions to fit isotonic calibration (80+ balanced examples) or sigmoid calibration (30+ balanced examples); otherwise documented action-specific bands are used. Confidence is capped below `1.0`.
5. **Personalized retrieval:** word/bigram TF–IDF cosine similarity is augmented by same user, sender, group, and business. One to three relevant same-user or high-similarity records are returned; IDs are validated against history.
6. **Hierarchical ensemble:** safety, opt-out/repeated unwanted content, genuine direct urgency, active transaction updates, administrator announcements, personalized history, semantic category rules, and conservative fallback run in that order. Quiet hours, fatigue, and recipient group mute state reduce interruption probability for non-urgent content; they are not unconditional mute rules and never suppress genuine urgent direct dependencies.
7. **Controlled outputs:** message type has explicit precedence separate from action. Reasons are short grounded templates. Confidence combines override precision, analogue quality, context completeness, and media certainty and never reaches `1.0`.

## Context and personalization

- User open/reply/dismiss rates come from 30-day counters. Seven-day notification load and dismissal ratio come from summaries indexed by user and date; together they form a bounded fatigue score. Recipient DND windows are evaluated against each message timestamp, with missing summary dates handled as unavailable rather than fabricated activity.
- Recipient group mute/read/dismiss state and sender group role/admin status are resolved independently. Direct mentions and administrator announcements distinguish operational school/work/society messages from noisy forwards.
- Business verification, official/sender-domain agreement, report load, promotion permission/opt-out, dismissals, and active relationship determine whether a legitimate promotion is digested or muted and whether an active transaction notifies.
- Safety always wins: prior banking engagement cannot legitimize an OTP request.

## Models and ablations

The implementation evaluates nearest-neighbor retrieval and the final hybrid on the solved examples (`python code/evaluate.py`). Calibration folds are grouped by user, so outcomes for a user cannot calibrate that user's held-out predictions. The report includes the calibration method, Brier score, log loss, expected calibration error, and populated reliability bins. When folds do not contain enough successes and errors for a stable fit, the explicit fallback bands are reported instead of fitting an unreliable curve. During design, deterministic rules, retrieval-only, TF–IDF text classification, metadata classification, and the hybrid were considered. The event labels are synthetic weak targets with strong template/sender leakage; therefore the shipped system favors the simpler retrieval/rule ensemble. A pure classifier would learn reaction artifacts and provides weaker safety guarantees. LLM adjudication is intentionally off by default for reproducibility, privacy, injection resistance, and offline reliability; `prompts/adjudication.json` documents the constrained contract if one is added.

Qualitative ablations:

| Removed component | Expected failure |
|---|---|
| Personalization / business history | opted-in and opted-out promotions collapse to one decision |
| Retrieval | weaker evidence and repeated-template handling |
| Safety | OTP, lookalike-domain, QR, injection, and medical attacks can interrupt |
| Media | empty-caption routing loses same-media/context signal and confidence penalty |
| Group membership | direct urgent messages in muted operational groups are mishandled |
| Notification context | low-priority content is over-notified |
| LLM | no runtime loss in the default system; deterministic fallback is the selected ablation |
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

The validator checks exact column order, exact input order and cardinality, unique IDs, allowed enums, numeric `[0,1]` confidence, concise nonempty reasons, no NaNs, and history-only evidence IDs. Call `validate_output(..., semantic_evidence=True)` to additionally reject evidence whose semantics, context, or observed reaction contradicts the output action. Tests cover multilingual scam text, trusted-sender OTP requests, prompt injection, QR pressure, unsafe medical forwards, muted-group direct urgency, explicit non-urgency, contradictory high-similarity evidence, and missing media extraction. Output is reloaded with pandas as part of validation.
The validator checks exact column order, exact input order and cardinality, unique IDs, allowed enums, numeric `[0,1]` confidence, concise nonempty reasons, no NaNs, and history-only evidence IDs. Tests cover multilingual scam text, trusted-sender OTP requests, prompt injection, QR pressure, unsafe medical forwards, quiet-hours urgency, fatigue deferral, admin/member announcements, muted-group direct mentions, missing daily summaries, explicit non-urgency, and missing media extraction. Output is reloaded with pandas as part of validation.

## Configuration and fallback behavior

See `.env.example`. No secrets are used. Media cache paths are configurable with `--cache`; cache keys include the media digest, extractor version, backend availability, and ASR model. Missing/corrupt optional media does not crash a batch. A media-only message without available ASR/OCR receives an explicit `ocr_unavailable` or `asr_unavailable` result and conservative reduced-confidence routing rather than fabricated text.

### OCR installation

`pytesseract` is a Python adapter and still requires the Tesseract executable. Install it with your OS package manager (for example, `apt install tesseract-ocr` on Debian/Ubuntu or `brew install tesseract` on macOS), then install this project's requirements. Add the language packs named by `ROUTER_OCR_LANGUAGES` (Tesseract syntax such as `eng+hin`). If the executable is not on `PATH`, set `TESSERACT_CMD` to its absolute path. QR decoding uses the installed headless OpenCV package; QR absence and decoder unavailability are both safe, non-fatal states.

### Local multilingual speech-to-text

ASR is intentionally optional because Whisper models are comparatively large. Install it with `python -m pip install 'faster-whisper>=1.0,<2'`, make the configured `ROUTER_WHISPER_MODEL` available locally, and choose CPU/GPU and compute settings in `.env.example`. Decoding is fixed to one greedy beam, temperature zero, disabled prior-text conditioning, and no VAD for repeatability. `ROUTER_ASR_TIMEOUT` bounds how long routing waits. Without the package/model—or after a timeout—the router records a declared fallback and continues; it never substitutes a guessed transcript. Model loading itself can require a first-run download, so pre-download models for offline evaluation.

## Files and packaging

`main.py` is the root entry point; implementation lives in `code/src/`; tests in `tests/`; the deterministic cache in `cache/`; predictions in `dataset/output.csv`; audit in `DATA_AUDIT.md`; and the conversation log is copied to `chat_transcript`. Create the submission archive locally with the following command. `code.zip` is intentionally ignored by Git because pull-request diff viewers do not support binary archives; the archive is a submission artifact, not source code.

```bash
python -m zipfile -c code.zip main.py code requirements.txt .env.example README.md prompts cache tests DATA_AUDIT.md
```

## Known limitations

The base dependency set does not ship a heavyweight multilingual speech model or the system Tesseract binary/language data. OCR quality depends on source resolution and installed language packs; brand extraction intentionally returns deterministic candidates rather than claiming entity resolution. Voice transcription quality depends on the locally selected Whisper model. Weak labels represent observed behavior rather than authoritative notification labels, so safety and operational urgency rules intentionally override them.
