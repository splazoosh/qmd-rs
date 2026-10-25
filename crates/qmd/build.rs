use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CMAKE_PREFIX_PATH");
    if env::var_os("CARGO_FEATURE_PREBUILT_GGML").is_none() {
        return;
    }
    let prefixes = env::var_os("CMAKE_PREFIX_PATH")
        .expect("prebuilt-ggml: CMAKE_PREFIX_PATH must name the ggml install");
    let prefixes: Vec<PathBuf> = env::split_paths(&prefixes).collect();
    for prefix in &prefixes {
        println!(
            "cargo:rerun-if-changed={}",
            prefix.join("key.txt").display()
        );
    }
    let Some((prefix, key)) = prefixes.iter().find_map(|p| {
        fs::read_to_string(p.join("key.txt"))
            .ok()
            .map(|k| (p, k.trim().to_owned()))
    }) else {
        panic!(
            "prebuilt-ggml: no ggml install with a key.txt in CMAKE_PREFIX_PATH ({})",
            env::join_paths(&prefixes).unwrap_or_default().display()
        );
    };

    match locked_sys_version() {
        Some(version) => assert!(
            key.starts_with(&format!("llama-cpp-sys-2-{version}-")),
            "prebuilt-ggml: the ggml in {} is {key}, but Cargo.lock has llama-cpp-sys-2 \
             {version}; rebuild ggml from that version under a new key",
            prefix.display()
        ),
        None => println!(
            "cargo:warning=prebuilt-ggml: no Cargo.lock names llama-cpp-sys-2, so the ggml \
             in {} ({key}) is linked unchecked",
            prefix.display()
        ),
    }
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    if let Ok(arch) = env::var("CMAKE_CUDA_ARCHITECTURES") {
        assert!(
            key.ends_with(&format!("-sm{arch}")),
            "prebuilt-ggml: the ggml in {} is {key}, but CMAKE_CUDA_ARCHITECTURES is {arch}; \
             rebuild ggml for it under a new key",
            prefix.display()
        );
    }

    println!(
        "cargo:rustc-link-search=native={}",
        prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=ggml-cuda");
}

/// The llama-cpp-sys-2 version in the nearest Cargo.lock above this crate.
fn locked_sys_version() -> Option<String> {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR")?);
    let lock = manifest_dir
        .ancestors()
        .map(|d| d.join("Cargo.lock"))
        .find(|p| p.is_file())?;
    println!("cargo:rerun-if-changed={}", lock.display());
    let lock = fs::read_to_string(lock).ok()?;
    lock.split("[[package]]")
        .find(|p| p.contains("name = \"llama-cpp-sys-2\""))
        .and_then(|p| p.split("version = \"").nth(1))
        .and_then(|v| v.split('"').next())
        .map(str::to_owned)
}
