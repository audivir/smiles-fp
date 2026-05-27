use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let rdkit_ver = env::var("RDKIT_VERSION").unwrap();
    let python_ver = env::var("PYTHON_VERSION").unwrap();
    let env_dir = env::var("ENV_DIR").unwrap();

    println!("cargo:warning=Building environment. This may take a moment...");

    let output = Command::new("python3")
        .arg("build_env.py")
        .arg(&rdkit_ver)
        .arg(&python_ver)
        .arg(&env_dir)
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
    let mut conda_include_dir = String::new();
    let mut python_include_dir = String::new();
    let mut rdkit_code_dir = String::new();
    let mut rdkit_build_dir = String::new();
    let mut pip_lib_dir = String::new();
    let mut boost_link_name = String::new();

    for line in stdout.lines() {
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "CONDA_INCLUDE_DIR" => conda_include_dir = value.trim().to_string(),
                "PYTHON_INCLUDE_DIR" => python_include_dir = value.trim().to_string(),
                "RDKIT_CODE_DIR" => rdkit_code_dir = value.trim().to_string(),
                "RDKIT_BUILD_DIR" => rdkit_build_dir = value.trim().to_string(),
                "PIP_LIB_DIR" => pip_lib_dir = value.trim().to_string(),
                "BOOST_LINK_NAME" => boost_link_name = value.trim().to_string(),
                _ => {}
            }
        }
    }

    // Compile C++ Shim
    cxx_build::bridge("src/lib.rs")
        .file("src/rdkit_shim.cpp")
        .include(&conda_include_dir) // Boost headers
        .include(&python_include_dir) // Python C headers
        .include(&rdkit_code_dir) // RDKit static headers
        .include(&rdkit_build_dir) // RDKit generated headers
        .include(PathBuf::from(&rdkit_code_dir).parent().unwrap())
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers")
        .compile("smiles_fp_rs_cxx");

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
    println!("cargo:rerun-if-changed=auto_env_builder.py");
}
