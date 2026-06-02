//! `standardize_f32_inplace` matches Burn `standardize` on the same buffer.

#![cfg(feature = "burn-engine")]

use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

#[test]
fn standardize_f32_matches_burn() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let raw: Vec<f32> = (0..450 * 490).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut flat = raw.clone();
    brainjepa::data::standardize_f32_inplace(&mut flat);

    let t = Tensor::<B, 4>::from_data(
        TensorData::new(raw, vec![1, 1, 450, 490]),
        &device,
    );
    let out = brainjepa::data::standardize(t);
    let burn_flat: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();

    let diff = flat
        .iter()
        .zip(burn_flat.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Burn reductions use tensor kernels; order differs slightly from a plain f32 loop.
    assert!(diff < 2e-5, "standardize mismatch max_abs {diff:.6}");
}
