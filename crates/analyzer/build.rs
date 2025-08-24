// crates/analyzer/build.rs
#[cfg(target_os = "windows")]
use std::{env, fs, io, path::PathBuf};

#[cfg(target_os = "windows")]
fn main() -> anyhow::Result<()> {
    // Choose flavor via env: ORT_FLAVOR=cpu|gpu  (default: cpu)
    let flavor = env::var("ORT_FLAVOR").unwrap_or_else(|_| "cpu".into());
    let (pkg, version) = match flavor.as_str() {
        "gpu" => ("Microsoft.ML.OnnxRuntime.Gpu.Windows", "1.22.1"),
        _ => ("Microsoft.ML.OnnxRuntime", "1.22.1"),
    };

    // Map target arch -> NuGet runtimes subdir
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "x86_64" => "win-x64",
        "aarch64" => "win-arm64",
        other => panic!("Unsupported Windows arch for ORT: {other}"),
    };

    // Where we’ll stash the extracted DLL for this build
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let ort_dir = out_dir.join("ort").join(arch);
    fs::create_dir_all(&ort_dir)?;

    let dll_path = ort_dir.join("onnxruntime.dll");
    if !dll_path.exists() {
        // Download the NuGet .nupkg (it's a zip)
        let url = format!("https://www.nuget.org/api/v2/package/{pkg}/{version}");
        let nupkg_path = out_dir.join("ort_download.nupkg");
        if !nupkg_path.exists() {
            eprintln!("Downloading {pkg} {version} …");
            let bytes = reqwest::blocking::get(&url)?.error_for_status()?.bytes()?;
            fs::write(&nupkg_path, &bytes)?;
        }

        // Extract only the DLL we need
        let file = fs::File::open(&nupkg_path)?;
        let mut zip = zip::ZipArchive::new(file)?;
        let needle = format!("runtimes/{arch}/native/onnxruntime.dll").replace('/', "\\");
        let mut found = false;

        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            let name = entry.mangled_name();
            // zip returns a PathBuf with `/`; normalize to Windows-style for comparison
            let rel = name.to_string_lossy().replace('/', "\\");
            if rel.ends_with(&needle) {
                if let Some(parent) = dll_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut out = fs::File::create(&dll_path)?;
                io::copy(&mut entry, &mut out)?;
                found = true;
                break;
            }
        }

        if !found {
            anyhow::bail!(
                "onnxruntime.dll not found inside NuGet package {pkg} {version} for {arch}"
            );
        }
    }

    // Expose the absolute path to your Rust code at compile time
    println!("cargo:rustc-env=ORT_PREBUILT_DLL={}", dll_path.display());

    // Rebuild conditions
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_FLAVOR");
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn main() {
    // Prevent rerun on every build
    println!("cargo:rerun-if-changed=build.rs");
}
