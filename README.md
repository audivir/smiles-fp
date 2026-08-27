# smiles-fp

Rust-accelerated Tanimoto similarity search over RDKit Morgan fingerprints.

## Prerequisites

- Python 3.10 to 3.14.
- RDKit, pinned to the version a given wheel was built against.
- To build from source: a Rust toolchain and [`uv`](https://docs.astral.sh/uv/) (RDKit/Boost headers
  are fetched automatically from PyPI, no conda required).

## Installation

A wheel is published to PyPI per supported RDKit release, pin the RDKit version during installation:

```bash
uv pip install smiles-fp~=0.2.1 rdkit~=2024.0 # installs smiles-fp==0.2.1.2024.9.6 and rdkit==2024.9.6
```

To build locally instead, `build_wheel.py` builds one wheel for the RDKit version you pass it,
targeting whichever Python interpreter runs the script (use `uv run --python` to pick one):

```bash
uv run --python 3.12 smiles-fp-pypi/build_wheel.py 2024.9.6
mv ./target/wheels ./target/smiles-fp
python -m http.server --directory ./target/
uv pip install smiles-fp --extra-index-url http://localhost:8000
```

## Usage

```python
from smiles_fp import get_mols, get_morgan_fps, save_fingerprints
from smiles_fp.search import similarity_search

query_ids = ["aspirin", "ethanol"]
mols = get_mols(["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"])
fps = get_morgan_fps(mols)
save_fingerprints(fps, "query.fp")

db_ids = ["ethylamine", "acetylsalicylic acid"]
db_mols = get_mols(["CCN", "CC(=O)OC1=CC=CC=C1C(=O)O"])
db_fps = get_morgan_fps(db_mols)
save_fingerprints(db_fps, "db.fp")

results = similarity_search(query_ids, "query.fp", db_ids, "db.fp", k=5)
```

## License

MIT, see `LICENSE`.
