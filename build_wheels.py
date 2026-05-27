"""Build the build pip wheels for different RDKit versions."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jinja2
import requests

from build_env import Version, get_conda_prog_exe, init_conda_env

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PARENT = Path(__file__).parent
CARGO_TOML = PARENT / "Cargo.toml"
TEMPLATE_PATH = PARENT / "pyproject.toml.j2"
PYPROJECT_TOML = PARENT / "pyproject.toml"
WHEEL_DIR = PARENT / "target" / "wheels"

_maturin = shutil.which("maturin")
if not _maturin:
    raise FileNotFoundError("Could not find maturin binary")
MATURIN_EXE = _maturin


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


def build_wheels(rdkit_ver: Version) -> None:
    """Build all wheels for the given RDKit version."""
    cargo_version = get_cargo_version()

    # Create a PEP 440 compliant local version (e.g. 0.1.0+rdkit.2023.9.5)
    safe_rdkit = str(rdkit_ver).replace("-", ".").replace("_", ".")
    wheel_ver = f"{cargo_version}+rdkit.{safe_rdkit}"

    logger.info("Targeting Wheel Version: %s", wheel_ver)

    # 1. Get supported Python versions from PyPI
    py_versions = get_supported_pythons(rdkit_ver)
    py_versions = [v for v in py_versions if not v < Version("3.10")]
    if not py_versions:
        raise ValueError(f"No Python wheels found for RDKit {rdkit_ver} on PyPI.")
    min_py, max_py = max(Version("3.10"), min(py_versions)), max(py_versions)
    py_ver = f">={min_py},<={max_py}"

    logger.info("Found Python targets for RDKit %s: %s", rdkit_ver, py_ver)

    template = jinja2.Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    rendered_toml = template.render(version=wheel_ver, py_ver=py_ver, rdkit_ver=str(rdkit_ver))

    PYPROJECT_TOML.write_text(rendered_toml, encoding="utf-8")

    # 3. Build a wheel for each supported Python version
    for py_ver in py_versions:
        logger.info("Building for Python %s", py_ver)

        env_dir = Path(f".conda_envs/{py_ver}")
        init_conda_env(env_dir, py_ver)
        py_exe = get_conda_prog_exe("python", env_dir)

        try:
            subprocess.check_call(  # noqa: S603
                [MATURIN_EXE, "build", "--release", "--auditwheel=skip", "--interpreter", py_exe],
                env={
                    **os.environ,
                    "PYTHON_VERSION": str(py_ver),
                    "RDKIT_VERSION": str(rdkit_ver),
                    "ENV_DIR": env_dir,
                },
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to create wheel for %s: %s", py_ver, e)  # noqa: TRY400


class Namespace(argparse.Namespace):
    """Typed argparse namespace for our wheel builder."""

    rdkit_versions: list[Version]


def build_wheels_cli() -> None:
    """Build the wheels for the RDKit version provided as CLI argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rdkit_versions", nargs="*", type=Version)
    args = parser.parse_args(namespace=Namespace)

    for rdkit_ver in args.rdkit_versions:
        build_wheels(rdkit_ver)


if __name__ == "__main__":
    raise SystemExit(build_wheels_cli())
