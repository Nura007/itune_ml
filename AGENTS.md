# Repository guidance

Implement the owner's Midterm protocol in docs/protocol.md and config/settings.json.

- Use exactly the seven whitelisted model inputs; never include user rating, rating counts, IDs or text.
- Keep test isolated from fitting, preprocessing, model selection and threshold tuning.
- Use the saved split manifest and its checksums. A new snapshot belongs to a separately reported experiment.
- Never invent empirical metrics, app records, instructor approval or human contributions.
- Tests may use explicitly labeled synthetic fixtures; public research outputs must come from the real API run.
- Raw API JSON, descriptions, app-level CSVs, split IDs, detailed predictions and model files stay private.
  Do not force-add ignored files. Use scripts/check_publication.py for public aggregate artifacts.
- API collection requires the recorded instructor approval and >=3.2 seconds between request starts.
- Run ruff check src tests scripts and pytest after substantive code changes.
- Keep neural networks, NLP and Streamlit for Final unless the owner changes scope.
- The DOCX/PDF source requirements remain unverified until their actual contents can be read.
- Do not merge or delete branches without the owner's explicit request.
