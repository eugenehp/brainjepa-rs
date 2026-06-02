use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ── full_context_mask ────────────────────────────────────────────────────────

#[test]
fn full_context_mask_shape() {
    let n_rois = 10;
    let n_time = 5;
    let mask = brainjepa::burn::full_context_mask::<B>(n_rois, n_time, &device());
    assert_eq!(mask.dims(), [1, n_rois * n_time]);
}

#[test]
fn full_context_mask_indices_sequential() {
    let n_rois = 8;
    let n_time = 4;
    let n = n_rois * n_time;
    let mask = brainjepa::burn::full_context_mask::<B>(n_rois, n_time, &device());
    let data = mask.squeeze::<1>().into_data();
    let indices: Vec<i64> = data.to_vec::<i64>().unwrap();
    let expected: Vec<i64> = (0..n as i64).collect();
    assert_eq!(indices, expected);
}

#[test]
fn full_context_mask_single_patch() {
    let mask = brainjepa::burn::full_context_mask::<B>(1, 1, &device());
    assert_eq!(mask.dims(), [1, 1]);
    let data = mask.into_data();
    let indices: Vec<i64> = data.to_vec::<i64>().unwrap();
    assert_eq!(indices, vec![0]);
}

// ── random_block_mask ────────────────────────────────────────────────────────

#[test]
fn random_block_mask_indices_in_bounds() {
    let n_rois = 20;
    let n_time = 10;
    let n = n_rois * n_time;
    let mask = brainjepa::burn::random_block_mask::<B>(
        n_rois, n_time, 0.5, 0.5, 4, &device(),
    );
    assert_eq!(mask.dims()[0], 1);
    let data = mask.squeeze::<1>().into_data();
    let indices: Vec<i64> = data.to_vec::<i64>().unwrap();
    for &idx in &indices {
        assert!(idx >= 0 && idx < n as i64, "index {idx} out of range [0, {n})");
    }
}

#[test]
fn random_block_mask_respects_min_keep() {
    let n_rois = 5;
    let n_time = 5;
    let min_keep = 10;
    let mask = brainjepa::burn::random_block_mask::<B>(
        n_rois, n_time, 0.1, 0.1, min_keep, &device(),
    );
    let k = mask.dims()[1];
    assert!(k >= min_keep, "expected at least {min_keep} patches, got {k}");
}

#[test]
fn random_block_mask_sorted() {
    let mask = brainjepa::burn::random_block_mask::<B>(
        15, 8, 0.4, 0.3, 4, &device(),
    );
    let data = mask.squeeze::<1>().into_data();
    let indices: Vec<i64> = data.to_vec::<i64>().unwrap();
    for w in indices.windows(2) {
        assert!(w[0] <= w[1], "indices not sorted: {} > {}", w[0], w[1]);
    }
}

// ── jepa_masks ───────────────────────────────────────────────────────────────

#[test]
fn jepa_masks_returns_enc_plus_three_pred() {
    let cfg = brainjepa::MaskConfig {
        n_rois: 20,
        n_time_patches: 8,
        seed: Some(42),
        ..Default::default()
    };
    let (enc_mask, pred_masks) = brainjepa::burn::jepa_masks::<B>(&cfg, &device());
    assert_eq!(enc_mask.dims()[0], 1);
    assert_eq!(pred_masks.len(), 3);
}

#[test]
fn jepa_masks_indices_in_range() {
    let n_rois = 15;
    let n_time = 6;
    let n = n_rois * n_time;
    let cfg = brainjepa::MaskConfig {
        n_rois,
        n_time_patches: n_time,
        seed: Some(123),
        ..Default::default()
    };
    let (enc_mask, pred_masks) = brainjepa::burn::jepa_masks::<B>(&cfg, &device());

    let check = |mask: Tensor<B, 2, Int>, label: &str| {
        let data = mask.squeeze::<1>().into_data();
        let indices: Vec<i64> = data.to_vec::<i64>().unwrap();
        for &idx in &indices {
            assert!(
                idx >= 0 && idx < n as i64,
                "{label}: index {idx} out of range [0, {n})"
            );
        }
    };

    check(enc_mask, "enc_mask");
    for (i, pm) in pred_masks.into_iter().enumerate() {
        check(pm, &format!("pred_mask[{i}]"));
    }
}

#[test]
fn jepa_masks_seed_reproducibility() {
    let cfg = brainjepa::MaskConfig {
        n_rois: 20,
        n_time_patches: 8,
        seed: Some(999),
        ..Default::default()
    };

    let (enc1, pred1) = brainjepa::burn::jepa_masks::<B>(&cfg, &device());
    let (enc2, pred2) = brainjepa::burn::jepa_masks::<B>(&cfg, &device());

    let to_vec = |t: Tensor<B, 2, Int>| -> Vec<i64> {
        t.squeeze::<1>().into_data().to_vec::<i64>().unwrap()
    };

    assert_eq!(to_vec(enc1), to_vec(enc2), "encoder masks differ with same seed");
    for (a, b) in pred1.into_iter().zip(pred2.into_iter()) {
        assert_eq!(to_vec(a), to_vec(b), "predictor masks differ with same seed");
    }
}

#[test]
fn jepa_masks_pred_not_in_encoder() {
    let n_rois = 20;
    let n_time = 8;
    let cfg = brainjepa::MaskConfig {
        n_rois,
        n_time_patches: n_time,
        seed: Some(77),
        ..Default::default()
    };
    let (enc_mask, pred_masks) = brainjepa::burn::jepa_masks::<B>(&cfg, &device());

    let enc_data = enc_mask.squeeze::<1>().into_data();
    let enc_indices: Vec<i64> = enc_data.to_vec::<i64>().unwrap();
    let enc_set: std::collections::HashSet<i64> = enc_indices.into_iter().collect();

    for (i, pm) in pred_masks.into_iter().enumerate() {
        let data = pm.squeeze::<1>().into_data();
        let pred_indices: Vec<i64> = data.to_vec::<i64>().unwrap();
        for &idx in &pred_indices {
            assert!(
                !enc_set.contains(&idx),
                "pred_mask[{i}] index {idx} overlaps with encoder mask"
            );
        }
    }
}
