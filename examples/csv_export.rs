/// Export embeddings to CSV for downstream analysis in R/pandas.
///
/// ```sh
/// cargo run --example csv_export --release -- \
///     data/brainjepa.safetensors \
///     data/gradient_mapping_450.csv \
///     data/test_fmri.safetensors \
///     embeddings.csv
/// ```
use brainjepa::prelude::*;
use burn::backend::NdArray;

type B = NdArray;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: csv_export <weights> <gradient.csv> <input> <output.csv>");
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

    let result = encoder.encode_safetensors(&args[3])?;
    println!("{} patches x {} dims", result.n_patches(), result.embed_dim());

    // Export with ROI and time indices
    brainjepa::csv_export::save_embeddings_csv_with_metadata(
        &result, &args[4], result.n_rois, result.n_time_patches,
    )?;
    println!("Saved: {}", args[4]);

    Ok(())
}
