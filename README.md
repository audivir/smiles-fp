On Ubuntu: sudo apt install libfreetype-dev

RDKit must be >2025.0.0.

Boost 1.70 and smaller: Python must be < 3.11
Boost 1.81 and smaller: NumPy must be < 2.0
RDKit in use and during build must be the same
2025.3.1 (Python 3.10, Boost 1.70)
2025.3.6 (Python 3.11, Boost 1.81)

Installation takes long time as it ...
    ... downloads Boost and clones RDKit
    ... bootstraps and builds Boost
    ... builds RDKit

Takes a while:
clones RDKit, downloads Boost source code, builds Boost, builds RDKit, builds the extension

pip install -v ".[dev]"
# uv pip install -v ".[dev]"

pytest tests