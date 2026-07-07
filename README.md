# smiles-fp-rs

```bash
python build_wheels.py 2024.9.6 2025.9.3 2025.9.6 2026.3.2
mv ./target/wheels ./target/smiles-fp-rs
python -m http.server --directory ./target/
uv pip install smiles-fp-rs --extra-index-url http://localhost:8000
```
