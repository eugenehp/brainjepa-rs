//! Parity: RLX `BrainJepaPredictor` vs Burn on real weights (deterministic masks).

#![cfg(all(feature = "burn", feature = "rlx"))]

mod parity_harness;

use brainjepa::burn::{jepa_masks, BrainJepaPredictor as BurnPredictor};
use brainjepa::data::load_fmri_safetensors;
use brainjepa::mask_config_for;
use brainjepa::rlx::BrainJepaPredictor as RlxPredictor;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::prelude::*;
use parity_harness::{
    default_configs, locate_weights, max_abs_diff, write_fmri_sample,
    TOL_RLX_PREDICTOR_ENC_VS_BURN, TOL_RLX_PREDICTOR_VS_BURN,
};

type B = NdArray;

fn tensor_to_f32<B: Backend>(t: Tensor<B, 3>) -> Vec<f32> {
    let data = t.into_data();
    match data.dtype {
        burn::tensor::DType::F32 => data
            .as_bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        burn::tensor::DType::F16 => data
            .as_bytes()
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

#[test]
fn rlx_predictor_matches_burn_predictor() {
    let (weights, gradient) = match locate_weights() {
        Some(t) => t,
        None => {
            eprintln!("\n[SKIP] predictor parity — missing weights.");
            return;
        }
    };

    let fmri_path = std::env::temp_dir().join("brainjepa_predictor_parity_fmri.safetensors");
    write_fmri_sample(&fmri_path).expect("write fmri");

    let device = NdArrayDevice::Cpu;
    let rlx_dev = rlx::Device::Cpu;
    brainjepa::rlx::ensure_device(rlx_dev).expect("rlx cpu");

    let (model_cfg, data_cfg) = default_configs();
    let w = weights.to_str().unwrap();
    let g = gradient.to_str().unwrap();

    let n_rois = data_cfg.crop_size.0;
    let n_time_patches = data_cfg.crop_size.1 / model_cfg.patch_size;
    let mask_cfg = mask_config_for(n_rois, n_time_patches);
    let (enc_idx, pred_masks) = jepa_masks::<B>(&mask_cfg, &device);

    let (burn_pred, _) =
        BurnPredictor::<B>::from_weights(w, g, &model_cfg, &data_cfg, &device).expect("burn");
    let input = load_fmri_safetensors::<B>(fmri_path.to_str().unwrap(), &device).expect("fmri");
    let mut x = input.data;
    let target = data_cfg.crop_size.1;
    if input.n_time != target && data_cfg.downsample {
        x = brainjepa::data::temporal_downsample(x, target).expect("downsample");
    }
    x = brainjepa::data::standardize(x);

    let (mut rlx_pred, _) =
        RlxPredictor::from_weights(w, g, &model_cfg, &data_cfg, &rlx_dev).expect("rlx");
    let fmri = brainjepa::data::load_fmri_safetensors_f32(fmri_path.to_str().unwrap()).expect("fmri");

    let enc_idx_vec = rlx_pred.enc_indices.clone();
    let compiled_n_pred = rlx_pred.pred_indices.len();

    for (i, pred_mask) in pred_masks.iter().enumerate() {
        if pred_mask.dims()[1] != compiled_n_pred {
            eprintln!(
                "[SKIP] pred mask {i}: len {} != compiled n_pred {}",
                pred_mask.dims()[1],
                compiled_n_pred
            );
            continue;
        }

        let pred_idx_vec: Vec<i64> = pred_mask
            .clone()
            .squeeze::<1>()
            .into_data()
            .to_vec::<i64>()
            .unwrap();

        let (burn_enc, burn_pred_out) =
            burn_pred.predict(x.clone(), &[enc_idx.clone()], &[pred_mask.clone()]);
        let burn_enc_f = tensor_to_f32(burn_enc);
        let burn_pred_f = tensor_to_f32(burn_pred_out);

        let (rlx_enc, rlx_pred_out) = rlx_pred
            .predict_f32(
                fmri.data.clone(),
                fmri.n_rois,
                fmri.n_time,
                &enc_idx_vec,
                &pred_idx_vec,
            )
            .expect("rlx predict");

        let enc_diff = max_abs_diff(&burn_enc_f, &rlx_enc);
        let pred_diff = max_abs_diff(&burn_pred_f, &rlx_pred_out);
        eprintln!("mask[{i}] encoder ctx max_abs = {enc_diff:.6}");
        eprintln!("mask[{i}] predictor out max_abs = {pred_diff:.6}");
        assert!(
            enc_diff < TOL_RLX_PREDICTOR_ENC_VS_BURN,
            "encoder context mask[{i}]: {enc_diff:.6} >= {TOL_RLX_PREDICTOR_ENC_VS_BURN}"
        );
        assert!(
            pred_diff < TOL_RLX_PREDICTOR_VS_BURN,
            "predictor mask[{i}]: {pred_diff:.6} >= {TOL_RLX_PREDICTOR_VS_BURN}"
        );
    }
}
