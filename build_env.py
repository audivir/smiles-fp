"""Build the build environment for maturin."""

from __future__ import annotations

import argparse
import json
import logging
import platform as platform_mod
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import StrPath

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LIB_EXTS = (".so", ".dylib", ".dll")


class Version:
    """Shim clone of packaging's Version."""

    def __init__(self, ver_str: str) -> None:
        """Initialize a Version from a string."""
        if not ver_str:
            raise ValueError("Empty version string")
        self.parts = ver_str.split(".")

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.major, self.minor, self.micro) < (other.major, other.minor, other.micro)

    def __repr__(self) -> str:
        return f"<{self}>"

    def __str__(self) -> str:
        return ".".join(self.parts)

    @property
    def major(self) -> int:
        """Major version."""
        return int(self.parts[0])

    @property
    def minor(self) -> int:
        """Minor version."""
        if len(self.parts) > 1:
            return int(self.parts[1])
        return 0

    @property
    def micro(self) -> int:
        """Patch version."""
        if len(self.parts) > 2:  # noqa: PLR2004
            return int(self.parts[2])
        return 0


def get_conda_exe() -> Path:
    """Get the path any conda package manager."""
    for exe in ["micromamba", "mamba", "conda"]:
        path = shutil.which(exe)
        if path:
            logger.error("Using %s as conda package manager.", exe)
            return Path(path)
    raise FileNotFoundError("No conda package manager found in PATH.")


CONDA_EXE = get_conda_exe()

if sys.platform == "darwin":
    _install_name_tool = shutil.which("install_name_tool")
    if not _install_name_tool:
        raise FileNotFoundError("No install_name_tool binary found")
    INSTALL_NAME_TOOL_EXE = Path(_install_name_tool)
else:
    INSTALL_NAME_TOOL_EXE = Path()  # not needed


def get_conda_prog_exe(prog: str, env_dir: StrPath) -> Path:
    """Get the absolute path for an executable from the conda environment's bin directory."""
    env_dir = Path(env_dir).absolute()
    conda_prog_exe = env_dir / "bin" / prog
    if not conda_prog_exe.exists():  # Windows fallback
        conda_prog_exe = env_dir / f"{prog}.exe"
    if not conda_prog_exe.exists():
        raise FileNotFoundError(
            f"Could not find {prog} binary in conda environment's bin directory"
        )
    return conda_prog_exe


def download_file(env_dir: StrPath, url: str, dest_path: StrPath) -> None:
    """Download a file using curl from a conda environment."""
    env_dir = Path(env_dir)
    py_exe = get_conda_prog_exe("python", env_dir)
    curl_exe = get_conda_prog_exe("curl", env_dir)
    subprocess.check_call([py_exe, "-m", "pip", "install", "certifi"])  # noqa: S603
    ca_bundle = subprocess.check_output(  # noqa: S603
        [py_exe, "-c", "import certifi; print(certifi.where())"], text=True
    ).strip()
    subprocess.check_call(  # noqa: S603
        [curl_exe, "-L", "--fail-with-body", url, "-o", dest_path],
        env={"CURL_CA_BUNDLE": ca_bundle},
    )


def get_wheel_platform(
    platform: Literal["darwin", "linux", "win32"], machine: Literal["arm64", "amd64"]
) -> str:
    """Determine the correct PyPI wheel tag based on the host OS."""
    if platform == "win32":
        return "win_amd64"
    if platform == "darwin":
        return "macosx_11_0_arm64" if machine == "arm64" else "macosx_10_9_x86_64"
    return "manylinux_2_28_x86_64"  # Default Linux fallback


def init_conda_env(env_dir: StrPath, py_ver: Version) -> None:
    """Initialize the conda environment with python and required packages."""
    packages = [
        f"python={py_ver.major}.{py_ver.minor}",
        "ca-certificates",
        "cmake",
        "curl",
        "freetype",
    ]
    sync_conda_env(env_dir, packages)


def sync_conda_env(env_dir: StrPath, packages: Iterable[str]) -> None:
    """Install the required versions of boost, cmake, and curl in the conda environment."""
    env_dir = Path(env_dir).absolute()

    packages = list(packages)

    action = "install" if env_dir.exists() else "create"
    cmd = [CONDA_EXE, action, "--prefix", env_dir, "-c", "conda-forge", "-y", *packages]

    logger.info(
        "Syncing local environment with %d packages using %s...", len(packages), CONDA_EXE.name
    )
    subprocess.check_output(cmd)  # noqa: S603


def fetch_rdkit_headers(rdkit_ver: Version, env_dir: Path) -> tuple[Path, Path]:
    """Download the RDKit source code and build the headers."""
    cache_dir = Path(".build_cache").absolute()

    # RDKit GitHub tags zero-pad the month (e.g., 2024.9.6 -> 2024_09_6)
    underscore_version = f"{rdkit_ver.major}_{str(rdkit_ver.minor).zfill(2)}_{rdkit_ver.micro}"

    rdkit_repos_dir = cache_dir / "rdkit_repos"
    rdkit_repos_dir.mkdir(exist_ok=True)

    base_dir = rdkit_repos_dir / f"rdkit-Release_{underscore_version}"
    code_dir = base_dir / "Code"
    build_code_dir = base_dir / "build" / "Code"

    if not build_code_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = cache_dir / f"rdkit_{underscore_version}.tar.gz"
        url = (
            f"https://github.com/rdkit/rdkit/archive/refs/tags/Release_{underscore_version}.tar.gz"
        )

        logger.info("Downloading RDKit %s source headers from GitHub...", rdkit_ver)
        download_file(env_dir, url, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=rdkit_repos_dir)  # noqa: S202
        tar_path.unlink()

        logger.info("Configuring RDKit headers via CMake...")
        build_dir = base_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_exe = get_conda_prog_exe("cmake", env_dir)
        py_exe = get_conda_prog_exe("python", env_dir)
        subprocess.check_output(  # noqa: S603
            [
                cmake_exe,
                "..",
                f"-DPython3_EXECUTABLE={py_exe}",
                "-DRDK_BUILD_PYTHON_WRAPPERS=OFF",
            ],
            cwd=build_dir,
        )

    return code_dir, build_code_dir


def cache_libs(  # noqa: C901,PLR0912,PLR0913
    env_dir: StrPath,
    pip_libs_dir: StrPath,
    platform: Literal["darwin", "linux", "win32"],
    machine: Literal["arm64", "amd64"],
    py_ver: Version,
    rdkit_ver: Version,
    boost_ver: Version | None,
) -> tuple[Version, str]:
    """Prepare and cache the dynamic libraries from PyPI."""
    env_dir = Path(env_dir)
    pip_libs_dir = Path(pip_libs_dir)

    py_exe = get_conda_prog_exe("python", env_dir)

    platform_tag = get_wheel_platform(platform, machine)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pip_cmd = [
            py_exe,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--platform",
            platform_tag,
            "--python-version",
            f"{py_ver}",
            "--no-deps",
            f"rdkit=={rdkit_ver}",
            "-d",
            str(tmp_dir),
        ]
        logger.info("Downloading PyPI wheel (%s) for linked libraries...", platform_tag)
        subprocess.check_call(pip_cmd)  # noqa: S603

        extracted_path = tmp_dir / "extracted"

        wheel_file = next(f for f in tmp_dir.iterdir() if f.suffix == ".whl")
        with zipfile.ZipFile(tmp_dir / wheel_file, "r") as zip_ref:
            zip_ref.extractall(extracted_path)  # noqa: S202

        if not boost_ver:
            # Extract the Boost version from filenames (works mostly on Linux wheels)
            for file in extracted_path.rglob("*"):
                if any(ext in file.suffixes for ext in LIB_EXTS) and "boost_python" in file.stem:
                    match = re.search(r"\.(so|dylib|dll)\.(\d+\.\d+\.\d+)$", file.name)
                    if match:
                        boost_ver = Version(match.group(2))

            # Windows and macOS wheels often lack the version suffix.
            if not boost_ver:
                raise ValueError("Boost version not found in library names.")

        boost_cache = get_boost_cache(pip_libs_dir, py_ver, rdkit_ver, boost_ver, platform, machine)
        boost_cache.mkdir(exist_ok=True)

        # 2. Extract, clean, and map libraries
        for file in extracted_path.rglob("*"):
            # Catch files even if the extension is buried (e.g., .so.1.0 or .1.dylib)
            if any(ext in file.suffixes for ext in LIB_EXTS):
                # Extract the base library name, ignoring hashes and embedded versions
                # e.g., libRDKitDataStructs.1.dylib -> libRDKitDataStructs
                # e.g., libRDKitDataStructs-4e1124.so.1.0 -> libRDKitDataStructs
                core_match = re.match(r"^([a-zA-Z0-9_]+)", file.stem)
                if not core_match:
                    continue

                core_name = core_match.group(1)

                # Filter for actual shared libraries
                if core_name.startswith("lib") or platform == "win32":
                    ext = (
                        ".dylib"
                        if ".dylib" in file.suffixes
                        else ".dll"
                        if ".dll" in file.suffixes
                        else ".so"
                    )

                    # Force a perfectly clean linker name
                    clean_name = f"{core_name}{ext}"

                    original_dest = boost_cache / file.name
                    shutil.copy2(file, original_dest)

                    clean_dest = boost_cache / clean_name

                    if file != clean_name:
                        shutil.copy2(file, clean_dest)

                    if platform == "darwin":
                        for dest in {original_dest, clean_dest}:
                            subprocess.run(  # noqa: S603
                                [
                                    INSTALL_NAME_TOOL_EXE,
                                    "-id",
                                    f"@rpath/{file.name}",
                                    str(dest),
                                ],
                                check=False,
                                stderr=subprocess.DEVNULL,
                            )

                    if "boost_python" in clean_name:
                        link_match = re.search(r"^(?:lib)?(boost_python\d+)", clean_name)
                        if link_match:
                            boost_link_name: str = link_match.group(1)

    return boost_ver, boost_link_name


def get_boost_cache(  # noqa: PLR0913
    pip_libs_dir: StrPath,
    py_ver: Version,
    rdkit_ver: Version,
    boost_ver: Version,
    platform: Literal["darwin", "linux", "win32"] | None = None,
    machine: Literal["arm64", "amd64"] | None = None,
) -> Path:
    """Return the path to the dynamic libary cache for the given versions."""
    return (
        Path(pip_libs_dir) / f"{py_ver.major}.{py_ver.minor}_{rdkit_ver}"
        f"_{boost_ver}_{platform or sys.platform}"
        f"_{machine or platform_mod.machine()}"
    )


def build_env(rdkit_ver: Version, py_ver: Version, env_dir: StrPath) -> None:
    """Build the build environment for maturin."""
    env_dir = Path(env_dir)

    cache_dir = Path(".build_cache").absolute()
    cache_dir.mkdir(exist_ok=True)

    pip_libs_dir = cache_dir / "pip_libs"
    pip_libs_dir.mkdir(exist_ok=True)

    boost_versions_file = cache_dir / "boost_versions.json"
    boost_versions = (
        json.loads(boost_versions_file.read_text()) if boost_versions_file.exists() else {}
    )
    boost_ver = boost_versions.get(f"{py_ver}_{rdkit_ver}")

    boost_link_name: str | None = None

    if boost_ver:
        boost_cache = get_boost_cache(pip_libs_dir, py_ver, rdkit_ver, boost_ver)
        if boost_cache.exists():
            logger.info("Using cached pip libraries in %s", boost_cache)
            for file in boost_cache.iterdir():
                link_match = re.match(r"^(?:lib)?(boost_python\d+)", file.name)
                if link_match:
                    boost_link_name = link_match.group(1)

    init_conda_env(env_dir, py_ver)
    conda_py_exe = get_conda_prog_exe("python", env_dir)

    py_inc_cmd = [
        conda_py_exe,
        "-c",
        "import sysconfig; print(sysconfig.get_path('include'))",
    ]
    python_include_dir = subprocess.check_output(py_inc_cmd, text=True).strip()  # noqa: S603

    if not boost_ver or not boost_link_name:
        boost_ver, boost_link_name = cache_libs(
            env_dir, pip_libs_dir, "linux", "amd64", py_ver, rdkit_ver, boost_ver
        )
        boost_versions[f"{py_ver}_{rdkit_ver}"] = f"{boost_ver}"
        boost_versions_file.write_text(json.dumps(boost_versions))

        if sys.platform != "linux" or platform_mod.machine() != "amd64":
            _, boost_link_name = cache_libs(
                env_dir,
                pip_libs_dir,
                sys.platform,
                platform_mod.machine(),
                py_ver,
                rdkit_ver,
                boost_ver,
            )

        boost_cache = get_boost_cache(pip_libs_dir, py_ver, rdkit_ver, boost_ver)

    sync_conda_env(env_dir, [f"boost={boost_ver}"])
    conda_include_dir = env_dir / "include"
    code_dir, build_code_dir = fetch_rdkit_headers(rdkit_ver, env_dir)

    # ruff: noqa: T201
    print(f"CONDA_INCLUDE_DIR={conda_include_dir}")
    print(f"PYTHON_INCLUDE_DIR={python_include_dir}")
    print(f"RDKIT_CODE_DIR={code_dir}")
    print(f"RDKIT_BUILD_DIR={build_code_dir}")
    print(f"PIP_LIB_DIR={boost_cache}")
    print(f"BOOST_LINK_NAME={boost_link_name}")


class Namespace(argparse.Namespace):
    """Typed argparse namespace for our env builder."""

    rdkit_version: Version
    python_version: Version
    env_dir: Path


def build_env_cli() -> None:
    """Build the build environment for maturin based on the provided CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("rdkit_version", type=Version)
    parser.add_argument("python_version", type=Version)
    parser.add_argument("env_dir", type=Path)
    args = parser.parse_args(namespace=Namespace)

    build_env(args.rdkit_version, args.python_version, args.env_dir)


if __name__ == "__main__":
    raise SystemExit(build_env_cli())
