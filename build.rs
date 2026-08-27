use std::env;
use std::process::Command;

fn main() {
    let rdkit_ver = env::var("RDKIT_VERSION").unwrap();
    let python_ver = env::var("PYTHON_VERSION").unwrap();

    // pyo3-build-config sets PYO3_PYTHON to the interpreter maturin is building for; fall back
    // to whatever `python3` resolves to on PATH for plain `cargo build`/dev flows.
    let python_exe = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let python_include_output = Command::new(&python_exe)
        .args(["-c", "import sysconfig; print(sysconfig.get_path('include'))"])
        .output()
        .expect("Failed to query Python include directory.");
    if !python_include_output.status.success() {
        panic!("Failed to query Python include directory from {}.", python_exe);
    }
    let python_include_dir = String::from_utf8_lossy(&python_include_output.stdout)
        .trim()
        .to_string();

    println!("cargo:warning=Building environment. This may take a moment...");

    let output = Command::new("python3")
        .arg("smiles-fp-pypi/build_env.py")
        .arg(&rdkit_ver)
        .arg(&python_ver)
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
    cxx_build::bridge("src/lib.rs")
        .file("src/rdkit_shim.cpp")
        .include(&boost_include_dir) // Boost headers (boost-headers PyPI package)
        .include(&rdkit_include_dir) // RDKit headers (rdkit-headers PyPI package)
        .include(&python_include_dir) // Python C headers
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers")
        .compile("smiles_fp_cxx");

    // Link Rust
    println!("cargo:rustc-link-search=native={}", pip_lib_dir);
    println!("cargo:rustc-link-lib=RDKitDataStructs");
    println!("cargo:rustc-link-lib=RDKitRDGeneral");
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
    println!("cargo:rerun-if-changed=smiles-fp-pypi/build_env.py");
}
