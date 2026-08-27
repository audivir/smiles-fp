"""Builds the build environment for maturin."""

from __future__ import annotations

import logging
import platform as platform_mod
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import doctyper
from packaging.version import Version

if TYPE_CHECKING:
    from _typeshed import StrPath

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LIB_EXTS = (".so", ".dylib", ".dll")


def find_required_exe(name: str) -> Path:
    """Locates `name` on PATH. Raises FileNotFoundError if it is missing."""
    exe = shutil.which(name)
    if not exe:
        raise FileNotFoundError(f"No {name} binary found in PATH")
    return Path(exe)


def find_install_name_tool(platform: str) -> Path:
    """Locates `install_name_tool` on macOS. Returns an empty path elsewhere (not needed)."""
    if platform != "darwin":
        return Path()
    return find_required_exe("install_name_tool")


INSTALL_NAME_TOOL_EXE = find_install_name_tool(sys.platform)
UV_EXE = find_required_exe("uv")

CACHE_DIR = Path(".build_cache").absolute()


def get_uv_platform(
    platform: Literal["darwin", "linux", "win32"],
    machine: Literal["arm64", "aarch64", "x86_64"],
) -> str:
    """Maps a host OS/CPU pair to the `uv pip install --python-platform` tag for RDKit's wheels."""
    if platform == "win32":
        return "x86_64-pc-windows-msvc"
    if platform == "darwin":
        return "aarch64-apple-darwin" if machine == "arm64" else "x86_64-apple-darwin"
    return "aarch64-manylinux_2_28" if machine == "aarch64" else "x86_64-manylinux_2_28"


def fetch_headers(rdkit_ver: Version, cache_dir: StrPath) -> tuple[Path, Path]:
    """Fetches prebuilt Boost/RDKit headers from PyPI (boost-headers / rdkit-headers)."""
    header_dir = Path(cache_dir) / "headers" / str(rdkit_ver)
    rdkit_include = header_dir / "rdkit_headers" / "include" / "rdkit"
    boost_include = header_dir / "boost_headers" / "include"

    if rdkit_include.exists() and boost_include.exists():
        return boost_include, rdkit_include

    header_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching rdkit-headers==%s.* ...", rdkit_ver)
    subprocess.check_call(  # noqa: S603
        [
            UV_EXE,
            "pip",
            "install",
            "--target",
            str(header_dir),
            "--no-deps",
            f"rdkit-headers=={rdkit_ver}.*",
        ]
    )

    metadata_files = list(header_dir.glob("rdkit_headers-*.dist-info/METADATA"))
    if not metadata_files:
        raise FileNotFoundError("rdkit-headers install did not produce dist-info metadata.")
    metadata = metadata_files[0].read_text(encoding="utf-8")
    match = re.search(r"(?m)^Requires-Dist:\s*boost-headers~=([\w.]+)\s*$", metadata)
    if not match:
        raise ValueError("Could not find a pinned boost-headers version in rdkit-headers metadata.")
    boost_ver = match.group(1)

    logger.info("Fetching boost-headers==%s.* ...", boost_ver)
    subprocess.check_call(  # noqa: S603
        [
            UV_EXE,
            "pip",
            "install",
            "--target",
            str(header_dir),
            "--no-deps",
            f"boost-headers=={boost_ver}.*",
        ]
    )

    return boost_include, rdkit_include


def get_lib_cache(
    pip_libs_dir: StrPath,
    py_ver: Version,
    rdkit_ver: Version,
    platform: str | None = None,
    machine: str | None = None,
) -> Path:
    """Returns the path to the shared-library cache for the given versions."""
    return (
        Path(pip_libs_dir) / f"{py_ver.major}.{py_ver.minor}_{rdkit_ver}"
        f"_{platform or sys.platform}_{machine or platform_mod.machine()}"
    )


def lib_ext(file: Path) -> str:
    """Returns the canonical shared-library extension for a (possibly versioned) filename."""
    if ".dylib" in file.suffixes:
        return ".dylib"
    if ".dll" in file.suffixes:
        return ".dll"
    return ".so"


def stage_lib(file: Path, lib_cache: Path, platform: str) -> str | None:
    """Copies one extracted library into the cache under both its original and clean names.

    Returns:
        The boost_python link name (e.g. "boost_python312") if `file` is that library.
    """
    # e.g. libRDKitDataStructs-4e1124.so.1.0 -> libRDKitDataStructs
    core_match = re.match(r"^([a-zA-Z0-9_]+)", file.stem)
    if not core_match:
        return None
    core_name = core_match.group(1)
    if not (core_name.startswith("lib") or platform == "win32"):
        return None

    clean_name = f"{core_name}{lib_ext(file)}"
    original_dest = lib_cache / file.name
    shutil.copy2(file, original_dest)
    clean_dest = lib_cache / clean_name
    if file.name != clean_name:
        shutil.copy2(file, clean_dest)

    if platform == "darwin":
        for dest in {original_dest, clean_dest}:
            subprocess.run(  # noqa: S603
                [INSTALL_NAME_TOOL_EXE, "-id", f"@rpath/{file.name}", str(dest)],
                check=False,
                stderr=subprocess.DEVNULL,
            )

    link_match = re.search(r"^(?:lib)?(boost_python\d+)", clean_name)
    return link_match.group(1) if link_match else None


def find_cached_boost_link_name(lib_cache: Path) -> str | None:
    """Looks for an already-staged boost_python library in the cache."""
    for file in lib_cache.iterdir():
        link_match = re.match(r"^(?:lib)?(boost_python\d+)", file.name)
        if link_match:
            return link_match.group(1)
    return None


def fetch_libs(
    pip_libs_dir: StrPath,
    py_ver: Version,
    rdkit_ver: Version,
    platform: Literal["darwin", "linux", "win32"],
    machine: Literal["arm64", "aarch64", "x86_64"],
) -> tuple[Path, str]:
    """Extracts the dynamic libraries bundled inside RDKit's own PyPI wheel.

    RDKit wheels already ship libRDKit*/libboost_python* built for a specific platform and
    Python ABI, so `uv pip install --target` (no execution, just an unpack) is enough to get at
    them without needing a matching local interpreter.
    """
    lib_cache = get_lib_cache(pip_libs_dir, py_ver, rdkit_ver, platform, machine)
    boost_link_name: str | None = None

    if lib_cache.exists():
        boost_link_name = find_cached_boost_link_name(lib_cache)
        if boost_link_name:
            return lib_cache, boost_link_name

    lib_cache.mkdir(parents=True, exist_ok=True)
    uv_platform = get_uv_platform(platform, machine)

    logger.info(
        "Fetching rdkit==%s wheel (%s, py%s) for linked libraries...",
        rdkit_ver,
        uv_platform,
        py_ver,
    )
    subprocess.check_call(  # noqa: S603
        [
            UV_EXE,
            "pip",
            "install",
            "--target",
            str(lib_cache / "_wheel"),
            "--python-platform",
            uv_platform,
            "--python-version",
            str(py_ver),
            "--only-binary=:all:",
            "--no-deps",
            f"rdkit=={rdkit_ver}",
        ]
    )

    extracted_path = lib_cache / "_wheel"

    for file in extracted_path.rglob("*"):
        if not any(ext in file.suffixes for ext in LIB_EXTS):
            continue
        link_name = stage_lib(file, lib_cache, platform)
        boost_link_name = link_name or boost_link_name

    shutil.rmtree(extracted_path, ignore_errors=True)

    if not boost_link_name:
        raise ValueError("Could not determine the boost_python library name from the RDKit wheel.")

    return lib_cache, boost_link_name


def build_env(rdkit_ver: Version, py_ver: Version) -> None:
    """Resolves headers and libraries needed to compile the C++ shim against RDKit."""
    CACHE_DIR.mkdir(exist_ok=True)
    pip_libs_dir = CACHE_DIR / "pip_libs"
    pip_libs_dir.mkdir(exist_ok=True)

    boost_include, rdkit_include = fetch_headers(rdkit_ver, CACHE_DIR)
    lib_cache, boost_link_name = fetch_libs(
        pip_libs_dir,
        py_ver,
        rdkit_ver,
        sys.platform,  # type: ignore[arg-type]
        platform_mod.machine(),  # type: ignore[arg-type]
    )

    # ruff: noqa: T201
    print(f"BOOST_INCLUDE_DIR={boost_include}")
    print(f"RDKIT_INCLUDE_DIR={rdkit_include}")
    print(f"PIP_LIB_DIR={lib_cache}")
    print(f"BOOST_LINK_NAME={boost_link_name}")


def cli(rdkit_version: str, python_version: str) -> None:
    """Resolves and prints the Boost/RDKit include dirs and libraries for one build.

    Args:
        rdkit_version: RDKit release to build against, e.g. 2026.3.2.
        python_version: Target Python version, e.g. 3.12.
    """
    build_env(Version(rdkit_version), Version(python_version))


def main() -> None:
    """Runs the build-environment CLI."""
    app = doctyper.DocTyper()
    app.command()(cli)
    app()


if __name__ == "__main__":
    main()
