// crates/analyzer/build.rs
use anyhow::{bail, Context};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    // Choose flavor via env: ORT_FLAVOR=cpu|gpu  (default: cpu)
    let flavor = env::var("ORT_FLAVOR").unwrap_or_else(|_| "cpu".into());
    let version = "1.22.1";

    // Resolve NuGet package, RID and file extension.
    #[cfg(target_os = "windows")]
    let (pkg, rid, ext) = {
        let rid = match env::var("CARGO_CFG_TARGET_ARCH")?.as_str() {
            "x86_64" => "win-x64",
            "aarch64" => "win-arm64",
            other => bail!("Unsupported Windows arch for ORT: {other}"),
        };
        let pkg = match flavor.as_str() {
            "gpu" => "Microsoft.ML.OnnxRuntime.Gpu.Windows",
            _ => "Microsoft.ML.OnnxRuntime",
        };
        (pkg, rid, ".dll")
    };

    #[cfg(target_os = "linux")]
    let (pkg, rid, ext) = {
        let rid = match env::var("CARGO_CFG_TARGET_ARCH")?.as_str() {
            "x86_64" => "linux-x64",
            "aarch64" => "linux-arm64",
            other => bail!("Unsupported Linux arch for ORT: {other}"),
        };
        let pkg = match flavor.as_str() {
            "gpu" => "Microsoft.ML.OnnxRuntime.Gpu.Linux",
            _ => "Microsoft.ML.OnnxRuntime",
        };
        (pkg, rid, ".so")
    };

    #[cfg(target_os = "macos")]
    let (pkg, rid, ext) = {
        // ONNX Runtime GPU via NuGet isn't provided on macOS (no CUDA). Force CPU.
        if flavor == "gpu" {
            bail!("ORT_FLAVOR=gpu is not supported on macOS NuGet packages; use CPU flavor");
        }
        let rid = match env::var("CARGO_CFG_TARGET_ARCH")?.as_str() {
            "x86_64" => "osx-x64",
            "aarch64" => "osx-arm64",
            other => bail!("Unsupported macOS arch for ORT: {other}"),
        };
        let pkg = "Microsoft.ML.OnnxRuntime";
        (pkg, rid, ".dylib")
    };

    // Cache the .nupkg under OUT_DIR so subsequent builds are offline.
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let cache_dir = out_dir.join("ort_cache");
    fs::create_dir_all(&cache_dir)?;
    let nupkg_path = cache_dir.join(format!("{pkg}-{version}.nupkg"));

    if !nupkg_path.exists() {
        let url = format!("https://www.nuget.org/api/v2/package/{pkg}/{version}");
        eprintln!("Downloading {pkg} {version} …");
        let bytes = reqwest::blocking::get(&url)?.error_for_status()?.bytes()?;
        fs::write(&nupkg_path, &bytes)?;
    }

    // Determine workspace root (prefer CARGO_WORKSPACE_DIR; robust fallback).
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_dir = env::var("CARGO_WORKSPACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| find_workspace_root(&manifest_dir).unwrap_or(manifest_dir.clone()));

    // Extract & copy all native libs for this RID into the WORKSPACE ROOT.
    // This includes provider libs for GPU.
    let file = fs::File::open(&nupkg_path)?;
    let mut zip = zip::ZipArchive::new(file)?;
    let needle_prefix = format!("runtimes/{rid}/native/");
    let mut copied = 0usize;

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().to_string(); // always '/' in zips
        if name.starts_with(&needle_prefix) && name.ends_with(ext) {
            let fname = Path::new(&name)
                .file_name()
                .context("zip entry without filename")?;
            let dest = workspace_dir.join(fname);

            // Always overwrite so version bumps take effect.
            let mut out =
                fs::File::create(&dest).with_context(|| format!("creating {}", dest.display()))?;
            io::copy(&mut entry, &mut out)
                .with_context(|| format!("writing {}", dest.display()))?;
            copied += 1;
        }
    }

    if copied == 0 {
        bail!("No {ext} files under {needle_prefix} in NuGet package {pkg} {version}");
    }

    // Rebuild conditions
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_FLAVOR");
    Ok(())
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn main() {
    // Prevent rerun every build on other OSes.
    println!("cargo:rerun-if-changed=build.rs");
}

// ---- helpers ----

/// Walk up from `start` to find a Cargo.toml containing `[workspace]`.
fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let toml = d.join("Cargo.toml");
        if toml.exists() {
            if let Ok(s) = fs::read_to_string(&toml) {
                if s.contains("[workspace]") {
                    return Some(d.to_path_buf());
                }
            }
        }
        dir = d.parent();
    }
    None
}
