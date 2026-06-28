# Contributing

Keep changes small and easy to review. This project fronts real robot commands,
so behavior changes should include tests or a clear manual verification note.

Run before opening a pull request:

```bash
python -m pip install -e .[dev]
python -m ruff check .
python -m pytest lekiwi_tui/tests
```

Do not commit local robot config, datasets, checkpoints, caches, or generated
build artifacts. Publish changes against `lekiwi.example.yaml`; keep
`lekiwi.yaml` private.
