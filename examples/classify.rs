/// Classification example — encoder + linear head.
///
/// Demonstrates using BrainJepaEncoder with ClassificationHead
/// for downstream binary classification (e.g., sex prediction from HCP-Aging).
///
/// Without a trained head, this just shows random logits.
/// Load real head weights with `head.load_weights()`.
///
/// ```sh
/// cargo run --example classify --release -- \
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
        eprintln!("usage: classify <weights> <gradient.csv> <input.safetensors>");
        std::process::exit(1);
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    brainjepa::init_threads(None);

    // Load encoder
    let cfg = ModelConfig::default();
    let (encoder, _) = BrainJepaEncoder::<B>::from_weights(
        &args[1], &args[2], &cfg, &DataConfig::default(), &device,
    )?;

    // Create classification head (2 classes: e.g., male/female)
    let head = ClassificationHead::<B>::new(cfg.embed_dim, 2, &device);

    // Load fMRI and encode
    let input = brainjepa::data::load_fmri_safetensors::<B>(&args[3], &device)?;
    let enc_out = encoder.encode_tensor(input.data)?;

    // Reshape embeddings back to [1, N, D] for the classification head
    let n = enc_out.n_patches();
    let d = enc_out.embed_dim();
    let emb_tensor = burn::prelude::Tensor::<B, 2>::from_data(
        burn::prelude::TensorData::new(enc_out.embeddings.clone(), vec![n, d]),
        &device,
    ).unsqueeze_dim::<3>(0); // [1, N, D]

    // Classify
    let logits = head.forward(emb_tensor);
    let classes = predict_classes(logits.clone());

    // Print results
    let logit_data: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();
    let class_data: Vec<i64> = classes.into_data().to_vec::<i64>().unwrap();

    println!("Logits: [{:.4}, {:.4}]", logit_data[0], logit_data[1]);
    println!("Predicted class: {} (untrained head — random)",
        class_data[0]);

    Ok(())
}
