# Repair validation

Executed against the supplied archive on the repair host:

- `python -m unittest discover -s tests -q`: 21 passed, including provided fake-model end-to-end tests and new all-review/full-trace tests.
- `python -m unittest discover -s document_app/tests -q`: 7 passed, including the GUI-to-ProofGuard adapter, upload cleanup, baseline compatibility and error paths.
- `python test_score.py`: 19 passed. Official scorer unchanged.
- Python compilation, JavaScript syntax and shell syntax checks passed.
- Preflight rejected the absent live calibrator; hidden runner rejected missing arguments.

These are software checks, not measured extraction results. No live model call was made, and browser visual QA was not performed. Gateway vision support is still unverified. Backend tests emitted dependency deprecation warnings but passed.

Package is an overlay: replaces listed code files and submission draft; preserves user's .env, data, raw run outputs and existing slides. Old calibrators are rejected by the new signature and must be refitted. Old slides must be regenerated before submission. The included example configuration has no secrets.
