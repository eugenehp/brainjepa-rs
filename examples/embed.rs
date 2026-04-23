/// Minimal embedding example — 20 lines of actual code.
///
/// ```sh
/// cargo run --example embed --release -- \
///     data/brainjepa.safetensors \
///     data/gradient_mapping_450.csv \
///     data/test_fmri.safetensors
/// ```
use brainjepa::prelude::*;
use burn::backend::NdArray;

type B = NdArray;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: embed <weights.safetensors> <gradient.csv> <input.safetensors> [output.safetensors]");
        std::process::exit(1);
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    brainjepa::init_threads(None);

    let (encoder, ms) = BrainJepaEncoder::<B>::from_weights(
        &args[1], &args[2],
        &ModelConfig::default(),
        &DataConfig::default(),
        &device,
    )?;
    println!("Loaded in {ms:.0} ms: {}", encoder.describe());

    let result = encoder.encode_safetensors(&args[3])?;
    println!("Encoded: {} patches x {} dims in {:.1} ms",
        result.n_patches(), result.embed_dim(), result.ms_encode);

    let out = args.get(4).map(|s| s.as_str()).unwrap_or("embeddings.safetensors");
    result.save_safetensors(out)?;
    println!("Saved: {out}");

    Ok(())
}
