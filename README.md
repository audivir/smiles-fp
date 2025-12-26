# smiles-fp

## Installation

```bash
# on Linux only
sudo apt install libfreetype-dev

git clone https://github.com/audivir/smiles-fp
cd smiles-fp

# set the correct RDKit version
python build_smiles_fp.py <smiles_fp_version> <rdkit_version>

pip install ".[dev]" # uv pip install ".[dev]"
pytest tests
```