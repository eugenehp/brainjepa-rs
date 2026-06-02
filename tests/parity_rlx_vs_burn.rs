//! Parity: RLX `BrainJepaEncoder` vs Burn reference on real HuggingFace weights.
//!
//! ```text
//! cargo run --release --bin download_weights --features hf-download
//! cargo test --release --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu,hf-download \
//!     --test parity_rlx_vs_burn -- --nocapture
//! ```

#![cfg(all(feature = "burn", feature = "rlx"))]

mod parity_harness;

use brainjepa::rlx::BrainJepaEncoder as RlxEncoder;
use brainjepa::burn::BrainJepaEncoder as BurnEncoder;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use parity_harness::{
    default_configs, locate_weights, max_abs_diff, write_fmri_sample, TOL_RLX_CPU_VS_BURN,
    TOL_RLX_GPU_VS_CPU,
};
use rlx::Device;

type B = NdArray;

fn pick_rlx_device() -> Device {
    let name = std::env::var("BRAINJEPA_RLX_DEVICE").unwrap_or_else(|_| "cpu".into());
    let dev = brainjepa::rlx::parse_device(&name).expect("BRAINJEPA_RLX_DEVICE");
    brainjepa::rlx::ensure_device(dev).expect("RLX device availability");
    dev
}

fn tolerance_for(rlx_dev: Device) -> f32 {
    match rlx_dev {
        Device::Cpu => TOL_RLX_CPU_VS_BURN,
        _ => TOL_RLX_GPU_VS_CPU,
    }
}

#[test]
fn rlx_encoder_matches_burn_encoder() {
    let (weights, gradient) = match locate_weights() {
        Some(t) => t,
        None => {
            eprintln!("\n[SKIP] parity — missing weights.");
            eprintln!("       run: cargo run --release --bin download_weights --features hf-download");
            eprintln!("       or set BRAINJEPA_WEIGHTS / BRAINJEPA_GRADIENT");
            return;
        }
    };

    let fmri_path = std::env::temp_dir().join("brainjepa_parity_fmri.safetensors");
    write_fmri_sample(&fmri_path).expect("write fmri sample");

    eprintln!("→ weights  = {}", weights.display());
    eprintln!("→ gradient = {}", gradient.display());
    eprintln!("→ fmri     = {}", fmri_path.display());
    let rlx_dev = pick_rlx_device();
    eprintln!("→ rlx dev  = {}", brainjepa::rlx::device::display_name(rlx_dev));

    let (model_cfg, data_cfg) = default_configs();
    let w = weights.to_str().unwrap();
    let g = gradient.to_str().unwrap();

    let device = NdArrayDevice::Cpu;
    let (burn_enc, _) = BurnEncoder::<B>::from_weights(w, g, &model_cfg, &data_cfg, &device)
        .expect("burn load");
    let burn_out = burn_enc
        .encode_safetensors(fmri_path.to_str().unwrap())
        .expect("burn encode");

    let (mut rlx_enc, _) = RlxEncoder::from_weights(w, g, &model_cfg, &data_cfg, &rlx_dev)
        .expect("rlx load");
    let rlx_out = rlx_enc
        .encode_safetensors(fmri_path.to_str().unwrap())
        .expect("rlx encode");

    assert_eq!(burn_out.shape, rlx_out.shape);
    assert_eq!(burn_out.embeddings.len(), rlx_out.embeddings.len());

    let tol = tolerance_for(rlx_dev);
    let diff = max_abs_diff(&burn_out.embeddings, &rlx_out.embeddings);
    eprintln!("max abs diff = {diff:.6} (tol {tol})");
    assert!(
        diff < tol,
        "RLX ({}) vs Burn mismatch: max_abs {diff:.6} >= {tol}",
        brainjepa::rlx::device::display_name(rlx_dev),
    );
}
