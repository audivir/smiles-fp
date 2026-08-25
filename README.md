# smiles-fp

Rust-accelerated Tanimoto similarity search over RDKit Morgan fingerprints.

## Prerequisites

- Python 3.10 to 3.14.
- RDKit, pinned to the version a given wheel was built against.
- To build from source: a Rust toolchain and a conda package manager (`micromamba`, `mamba`, or `conda`).

## Installation

A wheel is published to PyPI per supported RDKit release, with the RDKit version appended to
the package version (e.g. `0.1.0.2026.3.2` for RDKit `2026.3.2`):

```bash
uv pip install smiles-fp==0.1.0.2026.3.2
```

To build locally instead, `build_wheels.py` builds wheels per RDKit release into a local index:

```bash
python build_wheels.py 2024.9.6 2025.9.3 2025.9.6 2026.3.2
mv ./target/wheels ./target/smiles-fp
python -m http.server --directory ./target/
uv pip install smiles-fp --extra-index-url http://localhost:8000
```

## Usage

```python
from smiles_fp import get_mols, get_morgan_fps
from smiles_fp.search import similarity_search

mols = get_mols(["CCO", "CCN"])
fps = get_morgan_fps(mols)
```

## License

MIT, see `LICENSE`.
