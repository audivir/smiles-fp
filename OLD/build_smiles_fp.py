"""Configure and build the smiles-fp package."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import build
from pip._vendor.packaging.version import parse

DEFAULT_DIST_DIR = Path("dist")
PARENT = Path(__file__).parent
INIT_PATH = PARENT / "smiles_fp" / "__init__.py"
PYPROJECT_PATH = PARENT / "pyproject.toml"


def pip_download(name: str, version: str) -> None:
    """Try to download a version-specific package from PyPI."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.check_call(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    f"{name}=={version}",
                    "--no-deps",
                    "--dest",
                    temp_dir,
                ]
            )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to download {name}=={version}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set the correct requirements for smiles-fp and build it."
    )
    parser.add_argument(
        "version",
        help="Version of smiles-fp (e.g. 0.1)",
    )
    parser.add_argument(
        "rdkit_version",
        help="Version of RDKit (e.g. 2023.09.1)",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    version = parse(args.version)
    rdkit_version = parse(args.rdkit_version)
    if rdkit_version < parse("2025.0.0"):
        raise ValueError("RDKit version must be at least 2025.0.0")
    rdkit_str = f"{rdkit_version.major}.{rdkit_version.minor:02d}.{rdkit_version.micro}"
    # verify version exists
    pip_download("rdkit", rdkit_str)
    pip_download("rdkit_headers", rdkit_str)

    # read __init__.py
    init = INIT_PATH.read_text()
    # replace with current version
    init = re.sub(r"__version__ = \".*\"", f'__version__ = "{version}.{rdkit_str}"', init)
    # write __init__.py
    INIT_PATH.write_text(init)

    # read pyproject.toml
    pyproject = PYPROJECT_PATH.read_text()
    # replace with current version
    pyproject = re.sub(r"version = \".*\"", f'version = "{version}.{rdkit_str}"', pyproject)
    pyproject = re.sub(
        r'"rdkit_headers==\d+\.\d+\.\d+"', f'"rdkit_headers=={rdkit_str}"', pyproject
    )
    pyproject = re.sub(r'"rdkit==\d+\.\d+\.\d+"', f'"rdkit=={rdkit_str}"', pyproject)
    # write pyproject.toml
    PYPROJECT_PATH.write_text(pyproject)

    # build
    builder = build.ProjectBuilder(PARENT)
    builder.build("sdist", args.dist)
