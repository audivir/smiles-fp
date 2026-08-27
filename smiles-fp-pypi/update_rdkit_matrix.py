"""Computes which RDKit versions smiles-fp should build wheels for and test against.

Everything here is derived live from PyPI: `release.yml` builds wheels for the union of the
current latest-per-series RDKit releases and any RDKit versions that already have a published
smiles-fp wheel for the current Cargo.toml version (so a fresh smiles-fp release naturally
collapses to just latest-per-series, while repeat builds under the same Cargo.toml version keep
every already-built patch). Nothing gets committed back to release.yml.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import doctyper
import requests
import tomli
from build_wheels import get_supported_pythons
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
MIN_MAJOR = 2024

# The canonical dev/CI RDKit pin lives only in pyproject.toml; ci.yml and .pre-commit-config.yaml
# derive it at runtime via smiles-fp-pypi/dev_rdkit_version.py instead of duplicating it.
DEV_PIN_PATTERN = re.compile(r'"rdkit==[\d.]+"')
DEV_PIN_TEMPLATE = '"rdkit=={version}"'


def get_cargo_version() -> str:
    """Returns the `[package].version` string from Cargo.toml."""
    cargo_data = tomli.loads(CARGO_TOML.read_text(encoding="utf-8"))
    return str(cargo_data["package"]["version"])


def get_requires_python() -> SpecifierSet:
    """Returns the `[project].requires-python` specifier from pyproject.toml."""
    pyproject_data = tomli.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    return SpecifierSet(pyproject_data["project"]["requires-python"])


def fetch_latest_patches() -> list[Version]:
    """Returns the latest published patch version of each RDKit minor series since 2024."""
    response = requests.get("https://pypi.org/pypi/rdkit/json", timeout=30)
    response.raise_for_status()
    data = response.json()

    latest_by_series: dict[tuple[int, int], Version] = {}
    for ver_str, files in data["releases"].items():
        if not any(not f.get("yanked") for f in files):
            continue
        version = Version(ver_str)
        if version.major < MIN_MAJOR:
            continue
        series = (version.major, version.minor)
        if series not in latest_by_series or version > latest_by_series[series]:
            latest_by_series[series] = version

    return sorted(latest_by_series.values())


def get_published_rdkit_versions(cargo_version: str) -> set[Version]:
    """Returns the RDKit versions with an already-published smiles-fp wheel for `cargo_version`.

    smiles-fp's own wheel version is `f"{cargo_version}.{rdkit_version}"` (set by
    build_wheels.py), so the RDKit version is recovered by stripping that known prefix.
    """
    response = requests.get("https://pypi.org/pypi/smiles-fp/json", timeout=30)
    if response.status_code == requests.codes.not_found:
        return set()
    response.raise_for_status()
    data = response.json()

    prefix = f"{cargo_version}."
    return {
        Version(ver_str[len(prefix) :])
        for ver_str in data["releases"]
        if ver_str.startswith(prefix)
    }


def compute_release_matrix() -> list[Version]:
    """Computes the RDKit versions release.yml's build job should build wheels for."""
    cargo_version = get_cargo_version()
    latest = fetch_latest_patches()
    already_published = get_published_rdkit_versions(cargo_version)
    return sorted({*latest, *already_published})


def diff_new_versions() -> list[Version]:
    """Determines the RDKit versions that still need a wheel for the current Cargo.toml version."""
    cargo_version = get_cargo_version()
    latest = fetch_latest_patches()
    already_published = get_published_rdkit_versions(cargo_version)
    return sorted(v for v in latest if v not in already_published)


def compute_ci_matrix() -> list[dict[str, str]]:
    """Computes the valid (python-version, rdkit_version) pairs for ci.yml's tests job.

    Cross-references the release matrix against each RDKit release's own published Python
    wheel tags, so combinations with no matching RDKit wheel are never scheduled (e.g. RDKit
    2024.3.6 never published a Python 3.14 wheel).
    """
    requires_python = get_requires_python()

    pairs: list[dict[str, str]] = []
    for rdkit_ver in compute_release_matrix():
        pairs.extend(
            {"python-version": str(py_ver), "rdkit_version": str(rdkit_ver)}
            for py_ver in get_supported_pythons(rdkit_ver)
            if requires_python.contains(py_ver)
        )
    return pairs


def latest_overall() -> Version:
    """Returns the single newest published RDKit release across all minor series."""
    return max(fetch_latest_patches())


def bump_dev_pin(version: Version) -> bool:
    """Rewrites the canonical dev/CI RDKit pin in pyproject.toml to `version`.

    Returns:
        True if the file changed.
    """
    original_content = PYPROJECT_TOML.read_text(encoding="utf-8")
    new_content, count = DEV_PIN_PATTERN.subn(
        DEV_PIN_TEMPLATE.format(version=version), original_content
    )
    if count == 0:
        raise ValueError(f"Could not find the rdkit== pin in {PYPROJECT_TOML}")
    if new_content == original_content:
        return False
    PYPROJECT_TOML.write_text(new_content, encoding="utf-8")
    return True


def release_matrix() -> None:
    """Prints the RDKit versions release.yml's build job should build, as a JSON array."""
    print(json.dumps([str(v) for v in compute_release_matrix()]))  # noqa: T201


def diff() -> None:
    """Prints the RDKit versions that still need building, as a JSON array."""
    print(json.dumps([str(v) for v in diff_new_versions()]))  # noqa: T201


def ci_matrix() -> None:
    """Prints the valid (python-version, rdkit_version) pairs for ci.yml's tests job, as JSON."""
    print(json.dumps(compute_ci_matrix()))  # noqa: T201


def bump_pin() -> None:
    """Bumps the canonical dev/CI RDKit pin to the newest overall RDKit release."""
    changed = bump_dev_pin(latest_overall())
    print("changed" if changed else "unchanged")  # noqa: T201


def main() -> None:
    """Runs the RDKit-matrix CLI."""
    app = doctyper.DocTyper()
    app.command()(release_matrix)
    app.command()(diff)
    app.command()(ci_matrix)
    app.command()(bump_pin)
    app()


if __name__ == "__main__":
    main()
