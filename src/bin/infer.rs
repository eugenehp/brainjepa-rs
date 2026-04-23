/// Brain-JEPA fMRI — encoder inference CLI.
///
/// Build — CPU (default):
///   cargo build --release
///
/// Build — GPU (Metal on macOS, Vulkan on Linux):
///   cargo build --release --no-default-features --features wgpu
///
/// Usage:
///   infer --weights model.safetensors --gradient gradient_mapping_450.csv \
///         --input fmri.safetensors --output embeddings.safetensors
use std::time::Instant;

use clap::Parser;
use brainjepa::{BrainJepaEncoder, ModelConfig, DataConfig};

// ── Backend ──────────────────────────────────────────────────────────────────

#[cfg(all(feature = "wgpu-f16", not(feature = "ndarray"), not(feature = "wgpu")))]
mod backend {
    pub type B = burn::backend::wgpu::Wgpu<half::f16, i32, u32>;
    pub type Device = burn::backend::wgpu::WgpuDevice;
    pub fn device() -> Device { Device::DefaultDevice }
    pub const NAME: &str = "GPU (wgpu f16)";
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray"), not(feature = "wgpu-f16")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
    pub const NAME: &str = "GPU (wgpu f32)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(feature = "openblas-system")]
    pub const NAME: &str = "CPU (NdArray + OpenBLAS)";
    #[cfg(not(any(feature = "blas-accelerate", feature = "openblas-system")))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::{B, device};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Brain-JEPA fMRI encoder inference (Burn 0.20.1)")]
struct Args {
    /// Safetensors weights file.
    #[arg(long)]
    weights: String,

    /// Brain gradient mapping CSV (450 ROIs × 3 gradient axes).
    #[arg(long)]
    gradient: String,

    /// fMRI input file (.safetensors or .csv).
    #[arg(long)]
    input: String,

    /// Output safetensors file for embeddings.
    #[arg(long, default_value = "embeddings.safetensors")]
    output: String,

    /// Model variant: vit_small, vit_base, vit_large.
    #[arg(long, default_value = "vit_base")]
    model: String,

    /// YAML config file (optional, overrides --model).
    #[arg(long)]
    config: Option<String>,

    /// Number of CPU threads (0 = all cores).
    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    /// Verbose output.
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n_threads = brainjepa::init_threads(args.threads);
    let t0 = Instant::now();
    let dev = device();

    println!("Backend  : {}  ({n_threads} threads)", backend::NAME);

    // Parse config
    let (model_cfg, data_cfg) = if let Some(ref cfg_path) = args.config {
        let yaml = brainjepa::YamlConfig::from_file(cfg_path)?;
        (yaml.to_model_config()?, yaml.to_data_config())
    } else {
        (ModelConfig::from_variant(&args.model)?, DataConfig::default())
    };

    // Load model
    println!("Loading  : {}", args.weights);
    let (encoder, ms_weights) = BrainJepaEncoder::<B>::from_weights(
        &args.weights,
        &args.gradient,
        &model_cfg,
        &data_cfg,
        &dev,
    )?;
    println!("Model    : {}  ({ms_weights:.0} ms)", encoder.describe());

    // Encode
    println!("Input    : {}", args.input);
    let result = if args.input.ends_with(".csv") {
        encoder.encode_csv(&args.input)?
    } else {
        encoder.encode_safetensors(&args.input)?
    };

    println!("Encoding : {} patches × {} dims  ({:.1} ms)",
        result.n_patches(), result.embed_dim(), result.ms_encode);

    if args.verbose {
        let mean: f64 = result.embeddings.iter().map(|&v| v as f64).sum::<f64>()
            / result.embeddings.len() as f64;
        let std: f64 = (result.embeddings.iter().map(|&v| {
            let d = v as f64 - mean; d * d
        }).sum::<f64>() / result.embeddings.len() as f64).sqrt();
        println!("  mean={mean:.4}  std={std:.4}");
    }

    // Save
    result.save_safetensors(&args.output)?;
    println!("Output   : {}", args.output);

    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Total    : {ms_total:.0} ms");
    eprintln!("TIMING weights={ms_weights:.1}ms encode={:.1}ms total={ms_total:.1}ms",
              result.ms_encode);

    Ok(())
}
