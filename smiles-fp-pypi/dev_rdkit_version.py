"""Prints dependency-free the RDKit version pinned in pyproject.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT_TOML = Path(__file__).parent.parent / "pyproject.toml"


def get_dev_rdkit_version() -> str:
    """Extracts the pinned `rdkit==<version>` dependency from pyproject.toml.

    Raises:
        ValueError: If pyproject.toml has no pinned `rdkit==<version>` dependency.
    """
    match = re.search(r'"rdkit==([\d.]+)"', PYPROJECT_TOML.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Could not find a pinned rdkit==<version> dependency in {PYPROJECT_TOML}")
    return match.group(1)


def main() -> int:
    """Prints the pinned dev/CI RDKit version, or an error to stderr on failure."""
    try:
        print(get_dev_rdkit_version())  # noqa: T201
    except ValueError as e:
        print(e, file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
