# Document Desk

A local end-to-end document extraction and human review app for Project 1.

## Install inside Project_1

Place this `document_app` folder inside `~/sixt-hackathon-2026/Project_1/`.
Use the existing Project_1 virtual environment:

```bash
cd ~/sixt-hackathon-2026/Project_1
source .venv/bin/activate
pip install -r document_app/requirements.txt
bash document_app/run.sh
```

Open http://127.0.0.1:8000. If the Ubuntu computer is remote, forward port 8000 using your editor's Ports panel or an SSH tunnel; your browser's localhost is otherwise a different computer.

The backend reads `Project_1/.env` automatically. It expects LLM_BASE_URL (including /v1 if required by your gateway), LLM_API_KEY and LLM_MODEL. No temperature parameter is sent. Restart after changing settings. Never commit .env. Use a model/gateway that accepts image_url for PDFs/images. Text-only gateways can be tested with TXT uploads. The browser never receives the key.

## Workflow

1. Try a clearly labeled fictional sample without an API call.
2. Upload PDF/image/TXT and choose Invoice, Contract, or a custom JSON schema.
3. Analyze; PDFs are rendered page by page to images to avoid dependence on a damaged text layer.
4. Review each value, evidence and provisional confidence. Edit values, approve, or return to review.
5. Export the review record as JSON. This product export is not the hackathon JSONL prediction format.

## Architecture

- `frontend/`: plain HTML/CSS/JS, no build step. It communicates only with the HTTP API.
- `backend/app.py`: upload validation, response normalization, API and static serving.
- `backend/engine.py`: replaceable extraction adapter. Replace `extract_document(data, filename, schema)` with your future OCR, retrieval, verifier, Microsoft Decision-1 or other pipeline. Keep its return tuple `(fields, model, usage, source_text)` unchanged or implement an equivalent adapter.
- `POST /api/extract`: multipart file, JSON schema string, threshold; returns fields with name, value, confidence, evidence, page, reason, status plus run metadata.
- `GET /api/health`: configuration status (no key).
- API docs: http://127.0.0.1:8000/docs.

## Boundaries of this version

This is a functioning local product foundation, not a calibrated autonomous booking system. Suggested status requires non-null value, evidence and threshold; it is not proof of correctness. Image evidence is model-provided and not independently verified. Text quotes receive a simple substring check. No accuracy/savings claims are computed without ground truth.

PDFs: 8 pages maximum, uploads: 15 MB maximum, TXT: 100,000 characters. No silent truncation. Live calls send document content to the configured provider and may incur charges. No document/history database; review state lives in the tab and is lost on refresh unless exported. No auth or multiuser isolation; the server binds localhost. Do not expose it publicly as-is. PDF parsing runs in-process; use trusted/public/mock files for this hackathon. Dependencies use bounded ranges; pin after verifying your environment.

No changes to the original scorer, starter, or dataset are required. A future benchmark adapter should emit the official doc_id/fields JSONL separately and strip annotations from inputs.

## Checks

```bash
python -m unittest discover -s document_app/tests -v
```

Tests cover the API upload/schema error paths, normalized confidence, candidate gating and a mocked successful extraction. Live provider requests require your local key and were not executed in the build environment.
