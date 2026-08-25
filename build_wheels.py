"""Build the build pip wheels for different RDKit versions."""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from packaging.version import Version

from build_env import get_conda_prog_exe, init_conda_env

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PARENT = Path(__file__).parent
CARGO_TOML = PARENT / "Cargo.toml"
WHEEL_DIR = PARENT / "target" / "wheels"
WORKTREE_ROOT = PARENT / ".build_cache" / "worktrees"

maturin = shutil.which("maturin")
if not maturin:
    raise FileNotFoundError("Could not find maturin binary")
MATURIN_EXE = maturin

AUDITWHEEL_EXCLUDES = ["libRDKit*", "libboost_python*"]


def get_cargo_version() -> Version:
    """Extracts the base version robustly using a TOML parser."""
    if not CARGO_TOML.exists():
        raise FileNotFoundError(f"Missing {CARGO_TOML}")

    cargo_data = tomllib.loads(CARGO_TOML.read_text())

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
    py_versions = set()

    for release in data.get("urls", []):
        if release["packagetype"] == "bdist_wheel":
            py_tag = release["python_version"]
            # Extract standard CPython tags (e.g., 'cp310' -> '3.10')
            if py_tag.startswith("cp"):
                major, minor = py_tag[2], py_tag[3:]
                py_versions.add(Version(f"{major}.{minor}"))

    return sorted(py_versions)


def sync_worktree(rdkit_ver: Version) -> Path:
    """Check out a worktree mirroring the current working tree, reused across runs for caching."""
    worktree_dir = WORKTREE_ROOT / f"rdkit-{rdkit_ver}"
    snapshot_sha = subprocess.check_output(
        ["git", "stash", "create"],  # noqa: S607
        cwd=PARENT,
        text=True,
    ).strip()
    if not snapshot_sha:
        snapshot_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=PARENT,
            text=True,
        ).strip()

    if worktree_dir.exists():
        subprocess.check_call(  # noqa: S603
            ["git", "-C", str(worktree_dir), "reset", "--hard", snapshot_sha]  # noqa: S607
        )
    else:
        WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(worktree_dir), snapshot_sha],  # noqa: S607
            cwd=PARENT,
        )

    return worktree_dir


def patch_pyproject(path: Path, version: str, py_ver_range: str, rdkit_ver: Version) -> None:
    """Pin the wheel version, Python range, and exact RDKit dependency for one build."""
    content = path.read_text(encoding="utf-8")
    content = re.sub(r'(?m)^version = ".*"$', f'version = "{version}"', content, count=1)
    content = re.sub(
        r'(?m)^requires-python = ".*"$', f'requires-python = "{py_ver_range}"', content, count=1
    )
    content = re.sub(r'"rdkit(==[^"]*)?"', f'"rdkit=={rdkit_ver}"', content, count=1)
    path.write_text(content, encoding="utf-8")


def build_wheels(rdkit_ver: Version) -> None:
    """Build all wheels for the given RDKit version."""
    cargo_version = get_cargo_version()

    safe_rdkit = str(rdkit_ver).replace("-", ".").replace("_", ".")
    wheel_ver = f"{cargo_version}.{safe_rdkit}"

    logger.info("Targeting Wheel Version: %s", wheel_ver)

    py_versions = get_supported_pythons(rdkit_ver)
    py_versions = [v for v in py_versions if not v < Version("3.10")]
    if not py_versions:
        raise ValueError(f"No Python wheels found for RDKit {rdkit_ver} on PyPI.")
    min_py, max_py = max(Version("3.10"), min(py_versions)), max(py_versions)
    py_ver_range = f">={min_py},<={max_py}"

    logger.info("Found Python targets for RDKit %s: %s", rdkit_ver, py_ver_range)

    worktree_dir = sync_worktree(rdkit_ver)
    patch_pyproject(worktree_dir / "pyproject.toml", wheel_ver, py_ver_range, rdkit_ver)

    for py_ver in py_versions:
        logger.info("Building for Python %s", py_ver)

        env_dir = (PARENT / f".conda_envs/{py_ver}").absolute()
        init_conda_env(env_dir, py_ver)
        py_exe = get_conda_prog_exe("python", env_dir)

        with tempfile.TemporaryDirectory() as tmp:
            build_out = Path(tmp) if sys.platform == "linux" else WHEEL_DIR
            try:
                subprocess.check_call(  # noqa: S603
                    [
                        MATURIN_EXE,
                        "build",
                        "--release",
                        "--auditwheel=skip",
                        "--interpreter",
                        py_exe,
                        "--out",
                        str(build_out),
                    ],
                    cwd=worktree_dir,
                    env={
                        **os.environ,
                        "PYTHON_VERSION": str(py_ver),
                        "RDKIT_VERSION": str(rdkit_ver),
                        "ENV_DIR": str(env_dir),
                    },
                )
            except subprocess.CalledProcessError as e:
                logger.error("Failed to create wheel for %s: %s", py_ver, e)  # noqa: TRY400
                continue

            if sys.platform == "linux":
                try:
                    repair_linux_wheel(next(build_out.glob("*.whl")))
                except subprocess.CalledProcessError as e:
                    logger.error("Failed to repair wheel for %s: %s", py_ver, e)  # noqa: TRY400


def repair_linux_wheel(wheel_path: Path) -> None:
    """Repair a maturin --auditwheel=skip wheel into a valid manylinux wheel."""
    cmd = ["auditwheel", "repair", str(wheel_path), "--wheel-dir", str(WHEEL_DIR)]
    for lib in AUDITWHEEL_EXCLUDES:
        cmd += ["--exclude", lib]
    subprocess.check_call(cmd)  # noqa: S603


class Namespace(argparse.Namespace):
    """Typed argparse namespace for our wheel builder."""

    rdkit_versions: list[Version]


def build_wheels_cli() -> int:
    """Build the wheels for the RDKit version provided as CLI argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rdkit_versions", nargs="*", type=Version)
    args = parser.parse_args(namespace=Namespace)

    for rdkit_ver in args.rdkit_versions:
        build_wheels(rdkit_ver)

    return 0


if __name__ == "__main__":
    raise SystemExit(build_wheels_cli())
