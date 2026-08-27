"""Runs the test suite against every published (RDKit, Python) smiles-fp wheel via uv."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

from build_env import UV_EXE
from tqdm import tqdm
from update_rdkit_matrix import get_cargo_version, get_published_wheel_pairs

if TYPE_CHECKING:
    from packaging.version import Version

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def run_pytest(cargo_version: str, rdkit_ver: Version, py_ver: Version) -> CompletedProcess[str]:
    """Installs one published wheel into an ephemeral uv env and runs the test suite against it."""
    wheel_ver = f"{cargo_version}.{rdkit_ver}"
    return subprocess.run(  # noqa: S603
        [
            str(UV_EXE),
            "run",
            "--no-project",
            "--python",
            str(py_ver),
            "--with",
            f"smiles-fp=={wheel_ver}",
            "--with",
            f"rdkit=={rdkit_ver}",
            "--with",
            "pytest",
            "--with",
            "pytest-benchmark",
            "python",
            "-m",
            "pytest",
            str(TESTS_DIR),
            "--benchmark-skip",
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    """Runs the test suite against every published (RDKit, Python) wheel combination."""
    cargo_version = get_cargo_version()
    pairs = sorted(get_published_wheel_pairs(cargo_version))
    if not pairs:
        logger.warning("No published wheels found for smiles-fp==%s.*", cargo_version)
        return 1

    failures: list[tuple[Version, Version, str]] = []
    for rdkit_ver, py_ver in tqdm(pairs, desc="Testing published wheels"):
        result = run_pytest(cargo_version, rdkit_ver, py_ver)
        if result.returncode != 0:
            failures.append((rdkit_ver, py_ver, result.stdout + result.stderr))

    for rdkit_ver, py_ver, output in failures:
        logger.error("rdkit==%s python==%s", rdkit_ver, py_ver)
        logger.error(output)

    logger.info("%d/%d (RDKit, Python) pairs passed.", len(pairs) - len(failures), len(pairs))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
