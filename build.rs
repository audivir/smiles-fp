use std::env;
use std::fs;
use std::process::Command;

/// Checks whether the RDKit SMILES parser library exports `std::string` arguments with the
/// pre-C++11 libstdc++ ABI, which the mangled name `ERKSs` (`const std::string&`) marks.
fn uses_old_string_abi(lib_dir: &str) -> bool {
    let needle = b"SmilesParse13MolFromSmilesERKSs";
    let Ok(entries) = fs::read_dir(lib_dir) else {
        return false;
    };
    entries
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("libRDKitSmilesParse")
        })
        .filter_map(|e| fs::read(e.path()).ok())
        .any(|bytes| bytes.windows(needle.len()).any(|w| w == needle))
}

fn main() {
    let rdkit_ver = env::var("RDKIT_VERSION").unwrap();

    // pyo3-build-config sets PYO3_PYTHON to the interpreter maturin is building for; fall back
    // to whatever `python3` resolves to on PATH for plain `cargo build`/dev flows.
    let python_exe = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let python_include_output = Command::new(&python_exe)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_path('include'))",
        ])
        .output()
        .expect("Failed to query Python include directory.");
    if !python_include_output.status.success() {
        panic!(
            "Failed to query Python include directory from {}.",
            python_exe
        );
    }
    let python_include_dir = String::from_utf8_lossy(&python_include_output.stdout)
        .trim()
        .to_string();

    println!("cargo:warning=Building environment. This may take a moment...");

    // Run via python_exe (not a bare "python3") so build_env.py derives the target Python
    // version from its own running interpreter, guaranteeing it always matches
    // python_include_dir above instead of relying on a separately-passed version string.
    let output = Command::new(&python_exe)
        .arg("smiles-fp-pypi/build_env.py")
        .arg(&rdkit_ver)
        .output()
        .expect("Failed to execute environment builder.");

    let python_logs = String::from_utf8_lossy(&output.stderr);
    for line in python_logs.lines() {
        if !line.trim().is_empty() {
            println!("cargo:warning= > {}", line);
        }
    }

    if !output.status.success() {
        panic!("Building environment failed. Check the logs above.");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut boost_include_dir = String::new();
    let mut rdkit_include_dir = String::new();
    let mut pip_lib_dir = String::new();
    let mut boost_link_name = String::new();

    for line in stdout.lines() {
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "BOOST_INCLUDE_DIR" => boost_include_dir = value.trim().to_string(),
                "RDKIT_INCLUDE_DIR" => rdkit_include_dir = value.trim().to_string(),
                "PIP_LIB_DIR" => pip_lib_dir = value.trim().to_string(),
                "BOOST_LINK_NAME" => boost_link_name = value.trim().to_string(),
                _ => {}
            }
        }
    }

    // Compile C++ Shim
    let mut build = cxx_build::bridge("src/lib.rs");
    if env::consts::OS == "linux" && uses_old_string_abi(&pip_lib_dir) {
        // older RDKit wheels are built with the pre-C++11 libstdc++ string ABI, so std::string
        // arguments only link when the shim uses the same one.
        build.define("_GLIBCXX_USE_CXX11_ABI", "0");
    }
    build
        .file("src/rdkit_shim.cpp")
        .include(&boost_include_dir) // Boost headers (boost-headers PyPI package)
        .include(&rdkit_include_dir) // RDKit headers (rdkit-headers PyPI package)
        .include(&python_include_dir) // Python C headers
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers")
        .compile("smiles_fp_cxx");

    // Link Rust
    println!("cargo:rustc-link-search=native={}", pip_lib_dir);
    println!("cargo:rustc-link-lib=RDKitDataStructs");
    println!("cargo:rustc-link-lib=RDKitRDGeneral");
    println!("cargo:rustc-link-lib=RDKitGraphMol");
    println!("cargo:rustc-link-lib=RDKitSmilesParse");
    println!("cargo:rustc-link-lib=RDKitFingerprints");
    println!("cargo:rustc-link-lib=RDKitChemTransforms");
    println!("cargo:rustc-link-lib={}", boost_link_name);

    // Injecting rpath
    if env::consts::OS == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../rdkit/.dylibs");
    } else if env::consts::OS == "linux" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../rdkit.libs");
    }

    // Rebuild Triggers
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/rdkit_shim.cpp");
    println!("cargo:rerun-if-changed=src/salts.txt");
    println!("cargo:rerun-if-changed=smiles-fp-pypi/build_env.py");
    println!("cargo:rerun-if-env-changed=RDKIT_VERSION");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
