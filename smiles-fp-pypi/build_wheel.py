"""Builds the pip wheel for one RDKit version, targeting the interpreter running this script."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import doctyper
import requests
import tomli
from build_env import find_required_exe
from packaging.version import Version

if TYPE_CHECKING:
    from _typeshed import StrPath

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
WHEEL_DIR = REPO_ROOT / "target" / "wheels"
WORKTREE_ROOT = REPO_ROOT / ".build_cache" / "worktrees"

AUDITWHEEL_EXCLUDES = ["libRDKit*", "libboost_python*"]


def git_cmd(*args: StrPath, use_cwd: bool = False) -> str:
    """Runs a git command from repo root if `use_cwd=False` and returns its stripped output."""
    return subprocess.check_output(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=None if use_cwd else REPO_ROOT,
        text=True,
    ).strip()


def get_cargo_version() -> Version:
    """Extracts the base version robustly using a TOML parser."""
    if not CARGO_TOML.exists():
        raise FileNotFoundError(f"Missing {CARGO_TOML}")

    cargo_data = tomli.loads(CARGO_TOML.read_text())

    try:
        return Version(cargo_data["package"]["version"])
    except KeyError as e:
        raise ValueError("Could not find [package] -> 'version' in Cargo.toml") from e


def get_supported_pythons(rdkit_ver: Version) -> list[Version]:
    """Queries PyPI using requests to find supported Python wheels."""
    url = f"https://pypi.org/pypi/rdkit/{rdkit_ver}/json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch RDKit {rdkit_ver} info from PyPI: {e}") from e

    data = response.json()
    py_versions: set[Version] = set()

    for release in data.get("urls", []):
        if release["packagetype"] == "bdist_wheel":
            py_tag = release["python_version"]
            # Extract standard CPython tags (e.g., 'cp310' -> '3.10')
            if py_tag.startswith("cp"):
                major, minor = py_tag[2], py_tag[3:]
                py_versions.add(Version(f"{major}.{minor}"))

    return sorted(py_versions)


def sync_worktree(rdkit_ver: Version) -> Path:
    """Checks out a worktree mirroring the current working tree, reused across runs for caching."""
    worktree_dir = WORKTREE_ROOT / f"rdkit-{rdkit_ver}"
    snapshot_sha = git_cmd("stash", "create")
    if not snapshot_sha:
        snapshot_sha = git_cmd("rev-parse", "HEAD")

    if worktree_dir.exists():
        git_cmd("-C", str(worktree_dir), "reset", "--hard", snapshot_sha, use_cwd=True)
    else:
        WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
        git_cmd("worktree", "add", "--detach", str(worktree_dir), snapshot_sha)

    return worktree_dir


def patch_pyproject(path: Path, version: str, py_ver_range: str, rdkit_ver: Version) -> None:
    """Pins the wheel version, Python range, and exact RDKit dependency for one build."""
    content = path.read_text()
    content = re.sub(r'(?m)^version = ".*"$', f'version = "{version}"', content, count=1)
    content = re.sub(
        r'(?m)^requires-python = ".*"$', f'requires-python = "{py_ver_range}"', content, count=1
    )
    content = re.sub(r'"rdkit(==[^"]*)?"', f'"rdkit=={rdkit_ver}"', content, count=1)
    path.write_text(content)


def build_wheel(rdkit_ver_str: str) -> None:
    """Builds the wheel for one RDKit version, targeting the Python interpreter running this script.

    Args:
        rdkit_ver_str: The RDKit version to build against.
    """
    rdkit_ver = Version(rdkit_ver_str)
    maturin_exe = find_required_exe("maturin")
    cargo_version = get_cargo_version()
    py_ver = Version(f"{sys.version_info.major}.{sys.version_info.minor}")

    safe_rdkit = str(rdkit_ver).replace("-", ".").replace("_", ".")
    wheel_ver = f"{cargo_version}.{safe_rdkit}"

    logger.info("Targeting Wheel Version: %s (Python %s)", wheel_ver, py_ver)

    py_versions = get_supported_pythons(rdkit_ver)
    py_versions = [v for v in py_versions if not v < Version("3.10")]
    if not py_versions:
        raise ValueError(f"No Python wheels found for RDKit {rdkit_ver} on PyPI.")
    if py_ver not in py_versions:
        raise ValueError(f"RDKit {rdkit_ver} has no PyPI wheel for the running Python {py_ver}.")
    min_py, max_py = max(Version("3.10"), min(py_versions)), max(py_versions)
    py_ver_range = f">={min_py},<={max_py}"

    worktree_dir = sync_worktree(rdkit_ver)
    patch_pyproject(worktree_dir / "pyproject.toml", wheel_ver, py_ver_range, rdkit_ver)

    with tempfile.TemporaryDirectory() as tmp:
        build_out = Path(tmp) if sys.platform == "linux" else WHEEL_DIR
        subprocess.check_call(  # noqa: S603
            [
                maturin_exe,
                "build",
                "--release",
                "--auditwheel=skip",
                "--interpreter",
                sys.executable,
                "--out",
                str(build_out),
            ],
            cwd=worktree_dir,
            env={**os.environ, "RDKIT_VERSION": str(rdkit_ver)},
        )

        if sys.platform == "linux":
            repair_linux_wheel(next(build_out.glob("*.whl")))


def repair_linux_wheel(wheel_path: Path) -> None:
    """Repairs a maturin --auditwheel=skip wheel into a valid manylinux wheel."""
    cmd = ["auditwheel", "repair", str(wheel_path), "--wheel-dir", str(WHEEL_DIR)]
    for lib in AUDITWHEEL_EXCLUDES:
        cmd += ["--exclude", lib]
    subprocess.check_call(cmd)  # noqa: S603


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    app = doctyper.DocTyper()
    app.command()(build_wheel)
    app()
