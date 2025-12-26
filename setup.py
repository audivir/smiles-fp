"""Install smiles-fp package."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import boost_headers
import numpy as np
import rdkit
import rdkit_headers
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

PARENT = Path(__file__).parent
RDKIT_VERSION: str = rdkit.__version__


def find_lib(libs_dir: Path, glob_pattern: str) -> Path:
    """Find the first library file matching the provided pattern."""
    try:
        return next(libs_dir.glob(glob_pattern))
    except StopIteration as e:
        raise FileNotFoundError(f"Library not found: {glob_pattern}") from e


class build_ext(_build_ext):  # noqa: N801
    """Custom build extension."""

    def run(self) -> None:
        """Build extension."""
        # First run the normal build
        super().run()

        if platform.system() == "Darwin":
            otool, install_name_tool = (
                shutil.which("otool"),
                shutil.which("install_name_tool"),
            )
            if not otool or not install_name_tool:
                raise ValueError("otool and/or install_name_tool not found")
            loader_base = Path("@loader_path/../rdkit/.dylibs")

            # Now fix up the .so paths
            for ext in self.extensions:
                ext_path = self.get_ext_fullpath(ext.name)

                otool_output = subprocess.check_output(  # noqa: S603
                    [otool, "-L", ext_path],
                    text=True,
                )
                dylibs = [Path(p.split("(")[0].strip()) for p in otool_output.splitlines()[1:]]

                # List of absolute -> relative library paths to patch
                for dylib in dylibs:
                    if not dylib.is_relative_to("/DLC"):
                        continue
                    subprocess.check_call(  # noqa: S603
                        [install_name_tool, "-change", dylib, loader_base / dylib.name, ext_path]
                    )


# find libs
system = platform.system()
if system == "Linux":
    rdkit_libs = Path(rdkit.__file__).parent.parent / "rdkit.libs"
elif system == "Darwin":
    rdkit_libs = Path(rdkit.__file__).parent / ".dylibs"
else:
    raise OSError("Only Linux and Darwin supported")

if not rdkit_libs.is_dir():
    raise FileNotFoundError("Missing RDKit libraries")

py_version = sys.version_info
py_str = f"{py_version[0]}{py_version[1]}"
boost_lib = find_lib(rdkit_libs, f"libboost_python{py_str}*")
datastructs_lib = find_lib(rdkit_libs, "libRDKitDataStructs*")

if system == "Linux":
    # as the libRDKitDataStructs*.so requires libRDKitRDGeneral*.so
    # these required a bunch of boost libraries.
    # The simplest solution would be to patch libRDKitGeneral to add RPATH $ORIGIN
    # but this would add a runtime dependency of patchelf and we could not
    # prebuilt the wheels, thus we add all downstream required
    # libraries with -Wl,--no-as-needed
    # which then are correctly found due to the set $ORIGIN/../rdkit.libs
    rel_lib = "$ORIGIN/../rdkit.libs"
    additional = [
        "libRDKitRDGeneral*",
        "libboost_log_setup-*",
        "libboost_log-*",
        "libboost_locale-*",
        "libboost_iostreams-*",
        "libboost_graph-*",
        "libboost_fiber_numa-*",
        "libboost_fiber-*",
        "libboost_contract-*",
        "libboost_wave-*",
        "libboost_unit_test_framework-*",
        "libboost_type_erasure-*",
        "libboost_thread-*",
        "libboost_random-*",
        "libboost_prg_exec_monitor-*",
        f"libboost_numpy{py_str}-*",
        "libboost_nowide-*",
        "libboost_json-*",
        "libboost_filesystem-*",
        "libboost_coroutine-*",
        "libboost_chrono-*",
        "libboost_wserialization-*",
        "libboost_url-*",
        "libboost_timer-*",
        "libboost_stacktrace_noop-*",
        "libboost_stacktrace_from_exception-*",
        "libboost_stacktrace_basic-*",
        "libboost_stacktrace_backtrace-*",
        "libboost_stacktrace_addr2line-*",
        "libboost_serialization-*",
        "libboost_regex-*",
        "libboost_program_options-*",
        "libboost_math_tr1l-*",
        "libboost_math_tr1f-*",
        "libboost_math_tr1-*",
        "libboost_math_c99l-*",
        "libboost_math_c99f-*",
        "libboost_math_c99-*",
        "libboost_date_time-*",
        "libboost_context-*",
        "libboost_container-*",
        "libboost_charconv-*",
        "libboost_atomic-*",
    ]
    additional_libraries: list[str] = [f"-l:{find_lib(rdkit_libs, lib).name}" for lib in additional]
    extra_link_args = [
        f"-l:{boost_lib.name}",
        f"-l:{datastructs_lib.name}",
        "-Wl,--no-as-needed",
        *additional_libraries,
        f"-Wl,-rpath,{rel_lib}",
    ]
    libraries = []
elif system == "Darwin":
    libraries = [lib.stem.removeprefix("lib") for lib in (boost_lib, datastructs_lib)]
    extra_link_args = []
else:
    raise RuntimeError("unexpected platform")

ext_modules = [
    Extension(
        "smiles_fp._smiles_fp",
        sources=["cpp/smiles_fp.cpp"],
        include_dirs=[
            np.get_include(),  # NumPy headers
            boost_headers.get_include(),  # Boost headers
            rdkit_headers.get_include() / "rdkit",  # RDKit headers
        ],
        library_dirs=[f"{rdkit_libs}"],
        libraries=libraries,
        extra_link_args=extra_link_args,
        extra_compile_args=["-D_GLIBCXX_USE_CXX11_ABI=0"],
        language="c++",
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
