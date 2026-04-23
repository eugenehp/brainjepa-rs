/// Batch embedding — encode multiple fMRI files in one call.
///
/// ```sh
/// cargo run --example batch --release -- \
///     data/brainjepa.safetensors \
///     data/gradient_mapping_450.csv \
///     data/test_fmri.safetensors data/test_fmri.safetensors
/// ```
use brainjepa::prelude::*;
use burn::backend::NdArray;

type B = NdArray;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: batch <weights> <gradient.csv> <input1> [input2] ...");
        std::process::exit(1);
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    brainjepa::init_threads(None);

    let (encoder, _) = BrainJepaEncoder::<B>::from_weights(
        &args[1], &args[2],
        &ModelConfig::default(),
        &DataConfig::default(),
        &device,
    )?;

    let input_paths: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();
    println!("Encoding {} files...", input_paths.len());

    let results = encoder.encode_safetensors_batch(&input_paths)?;

    for (path, result) in input_paths.iter().zip(&results) {
        println!("  {path}: {} patches x {} dims  ({:.1} ms)",
            result.n_patches(), result.embed_dim(), result.ms_encode);
    }

    // Save each result
    for (i, result) in results.iter().enumerate() {
        let out = format!("embeddings_{i}.safetensors");
        result.save_safetensors(&out)?;
    }
    println!("Saved {} files", results.len());

    Ok(())
}
