"""Computes which RDKit versions smiles-fp should build wheels for and test against."""

from __future__ import annotations

import json
import re
from pathlib import Path

import doctyper
import requests
import tomli
from build_wheel import get_supported_pythons
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
MIN_MAJOR = 2024

DEV_PIN_PATTERN = re.compile(r'"rdkit==[\d.]+"')
DEV_PIN_TEMPLATE = '"rdkit=={version}"'

CROSS_PLATFORM_OS = ["macos-latest", "ubuntu-24.04-arm"]
ALL_OS = ["ubuntu-latest", "ubuntu-24.04-arm", "macos-latest"]

CP_TAG_PATTERN = re.compile(r"-cp(\d)(\d+)-")

# smiles_fp-0.2.1.2026.3.5-cp312-cp312-manylinux_2_34_aarch64.whl -> ubuntu-24.04-arm
OS_BY_PLATFORM_TAG = (
    ("macosx", "arm64", "macos-latest"),
    ("manylinux", "aarch64", "ubuntu-24.04-arm"),
    ("manylinux", "x86_64", "ubuntu-latest"),
)


def get_cargo_version() -> str:
    """Returns the `[package].version` string from Cargo.toml."""
    cargo_data = tomli.loads(CARGO_TOML.read_text())
    return str(cargo_data["package"]["version"])


def get_requires_python() -> SpecifierSet:
    """Returns the `[project].requires-python` specifier from pyproject.toml."""
    pyproject_data = tomli.loads(PYPROJECT_TOML.read_text())
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

    Strips smiles-fp `cargo_version` from the wheel to recover `rdkit_version`.
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
    """Computes the RDKit versions the release job should build wheels for."""
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


def compute_pairs(rdkit_versions: list[Version]) -> list[dict[str, str]]:
    """Cross-references RDKit versions against each release's own published Python wheel tags.

    Combinations with no matching RDKit wheel are dropped (e.g. RDKit 2024.3.6 never published
    a Python 3.14 wheel).
    """
    requires_python = get_requires_python()

    pairs: list[dict[str, str]] = []
    for rdkit_ver in rdkit_versions:
        pairs.extend(
            {"python-version": str(py_ver), "rdkit_version": str(rdkit_ver)}
            for py_ver in get_supported_pythons(rdkit_ver)
            if requires_python.contains(py_ver)
        )
    return pairs


def compute_ci_matrix() -> list[dict[str, str]]:
    """Computes the valid (python-version, rdkit_version, os) triples for ci.yml's tests job.

    Every valid pair runs on ubuntu-latest. The newest pair additionally runs once on
    macos-latest and once on ubuntu-24.04-arm, so platform-specific build/link regressions
    (e.g. rpath resolution, boost_python naming) get caught without tripling the whole matrix.
    """
    pairs = compute_pairs(compute_release_matrix())
    triples = [{**pair, "os": "ubuntu-latest"} for pair in pairs]
    if pairs:
        newest = pairs[-1]
        triples.extend({**newest, "os": os_name} for os_name in CROSS_PLATFORM_OS)
    return triples


def compute_build_matrix() -> list[dict[str, str]]:
    """Computes the (python-version, rdkit_version) pairs release.yml's build job builds."""
    return compute_pairs(compute_release_matrix())


def wheel_os(filename: str) -> str | None:
    """Maps a wheel filename's platform tag to the GitHub Actions runner os that built it."""
    for platform_tag, arch_tag, os_name in OS_BY_PLATFORM_TAG:
        if platform_tag in filename and arch_tag in filename:
            return os_name
    return None


def get_published_wheel_pairs(cargo_version: str) -> set[tuple[Version, Version]]:
    """Returns the (rdkit_version, python_version) pairs with an already-published wheel."""
    return {
        (rdkit_ver, py_ver) for rdkit_ver, py_ver, _os in get_published_wheel_triples(cargo_version)
    }


def get_published_wheel_triples(cargo_version: str) -> set[tuple[Version, Version, str]]:
    """Returns the (rdkit_version, python_version, os) triples with an already-published wheel."""
    response = requests.get("https://pypi.org/pypi/smiles-fp/json", timeout=30)
    if response.status_code == requests.codes.not_found:
        return set()
    response.raise_for_status()
    data = response.json()

    prefix = f"{cargo_version}."
    triples: set[tuple[Version, Version, str]] = set()
    for ver_str, files in data["releases"].items():
        if not ver_str.startswith(prefix):
            continue
        rdkit_ver = Version(ver_str[len(prefix) :])
        for file_info in files:
            filename = file_info["filename"]
            cp_match = CP_TAG_PATTERN.search(filename)
            os_name = wheel_os(filename)
            if cp_match and os_name:  # pragma: no branch
                py_ver = Version(f"{cp_match.group(1)}.{cp_match.group(2)}")
                triples.add((rdkit_ver, py_ver, os_name))
    return triples


def missing_build_matrix() -> list[dict[str, str]]:
    """Drops (rdkit, python, os) triples that already have a published wheel."""
    published = get_published_wheel_triples(get_cargo_version())
    return [
        {**pair, "os": os_name}
        for pair in compute_build_matrix()
        for os_name in ALL_OS
        if (Version(pair["rdkit_version"]), Version(pair["python-version"]), os_name)
        not in published
    ]


def latest_overall() -> Version:
    """Returns the single newest published RDKit release across all minor series."""
    return max(fetch_latest_patches())


def bump_dev_pin(version: Version) -> bool:
    """Rewrites the canonical dev/CI RDKit pin in pyproject.toml to `version`.

    Returns:
        True if the file changed.
    """
    original_content = PYPROJECT_TOML.read_text()
    new_content, count = DEV_PIN_PATTERN.subn(
        DEV_PIN_TEMPLATE.format(version=version), original_content
    )
    if count == 0:
        raise ValueError(f"Could not find the rdkit== pin in {PYPROJECT_TOML}")
    if new_content == original_content:
        return False
    PYPROJECT_TOML.write_text(new_content)
    return True


def release_matrix() -> None:
    """Prints the RDKit versions release.yml's build job should build, as a JSON array."""
    print(json.dumps([str(v) for v in compute_release_matrix()]))  # noqa: T201


def diff() -> None:
    """Prints the RDKit versions that still need building, as a JSON array."""
    print(json.dumps([str(v) for v in diff_new_versions()]))  # noqa: T201


def ci_matrix() -> None:
    """Prints the (python-version, rdkit_version, os) triples for ci.yml's tests job, as JSON."""
    print(json.dumps(compute_ci_matrix()))  # noqa: T201


def build_matrix() -> None:
    """Prints the (python-version, rdkit_version) pairs release.yml's build job builds, as JSON."""
    print(json.dumps(compute_build_matrix()))  # noqa: T201


def missing_matrix() -> None:
    """Prints the (python-version, rdkit_version) pairs still missing a published wheel, as JSON."""
    print(json.dumps(missing_build_matrix()))  # noqa: T201


def bump_pin() -> None:
    """Bumps the canonical dev/CI RDKit pin to the newest overall RDKit release."""
    changed = bump_dev_pin(latest_overall())
    print("changed" if changed else "unchanged")  # noqa: T201


if __name__ == "__main__":
    app = doctyper.DocTyper()
    app.command()(release_matrix)
    app.command()(diff)
    app.command()(ci_matrix)
    app.command()(build_matrix)
    app.command()(missing_matrix)
    app.command()(bump_pin)
    app()
