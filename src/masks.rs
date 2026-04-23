/// Spatiotemporal masking for Brain-JEPA.
///
/// Three masking strategies from the paper:
/// - **Cross-ROI**: sample ROIs within the same time window as the context
/// - **Cross-Time**: sample time patches from a different time window
/// - **Double-Cross**: sample both different ROIs and different time patches
///
/// At inference time, `full_context_mask` passes all patches to the encoder.
/// The random masks are used for JEPA evaluation (encoder + predictor).
use burn::prelude::*;

/// Configuration for spatiotemporal mask generation.
#[derive(Debug, Clone)]
pub struct MaskConfig {
    pub n_rois: usize,
    pub n_time_patches: usize,
    /// Fraction of patches to keep for encoder context (min, max).
    pub enc_mask_scale: (f64, f64),
    /// Fraction of ROI range for predictor targets.
    pub pred_mask_r_scale: (f64, f64),
    /// Fraction of time range for predictor targets.
    pub pred_mask_t_scale: (f64, f64),
    /// Minimum patches to keep in any mask.
    pub min_keep: usize,
    /// Random seed (None = non-deterministic).
    pub seed: Option<u64>,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            n_rois: 450,
            n_time_patches: 10,
            enc_mask_scale: (0.84, 1.0),
            pred_mask_r_scale: (0.45, 0.6),
            pred_mask_t_scale: (0.0, 0.4),
            min_keep: 4,
            seed: None,
        }
    }
}

/// Generate a full context mask (no masking — keep all patches).
///
/// Returns [1, N] with indices [0, 1, ..., N-1].
pub fn full_context_mask<B: Backend>(
    n_rois: usize,
    n_time_patches: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let n = n_rois * n_time_patches;
    let indices: Vec<i64> = (0..n as i64).collect();
    Tensor::<B, 1, Int>::from_data(TensorData::new(indices, vec![n]), device)
        .unsqueeze_dim::<2>(0)
}

/// Generate a random block mask for a 2D grid (ROIs × time patches).
///
/// Selects a contiguous block of `roi_frac` of ROIs and `time_frac` of
/// time patches, then returns the flattened indices of patches inside
/// the block.
///
/// Returns [1, K] where K >= min_keep.
pub fn random_block_mask<B: Backend>(
    n_rois: usize,
    n_time_patches: usize,
    roi_frac: f64,
    time_frac: f64,
    min_keep: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let n_r = ((n_rois as f64 * roi_frac).round() as usize).max(1);
    let n_t = ((n_time_patches as f64 * time_frac).round() as usize).max(1);

    // Random start positions
    let r_start = fastrand::usize(..=(n_rois.saturating_sub(n_r)));
    let t_start = fastrand::usize(..=(n_time_patches.saturating_sub(n_t)));

    let mut indices = Vec::with_capacity(n_r * n_t);
    for r in r_start..(r_start + n_r) {
        for t in t_start..(t_start + n_t) {
            indices.push((r * n_time_patches + t) as i64);
        }
    }

    // Ensure min_keep
    while indices.len() < min_keep {
        let idx = fastrand::usize(..(n_rois * n_time_patches)) as i64;
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    indices.sort();

    let k = indices.len();
    Tensor::<B, 1, Int>::from_data(TensorData::new(indices, vec![k]), device)
        .unsqueeze_dim::<2>(0)
}

/// Generate encoder context mask and predictor target masks for JEPA evaluation.
///
/// Returns `(enc_mask, pred_masks)`:
/// - `enc_mask`: [1, K_enc] — context patches for the encoder
/// - `pred_masks`: Vec of [1, K_pred] — target patches for the predictor
///   (cross-ROI, cross-time, double-cross)
pub fn jepa_masks<B: Backend>(
    cfg: &MaskConfig,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Vec<Tensor<B, 2, Int>>) {
    if let Some(seed) = cfg.seed {
        fastrand::seed(seed);
    }

    let n_r = cfg.n_rois;
    let n_t = cfg.n_time_patches;
    let n = n_r * n_t;

    // Encoder context: large block
    let enc_roi_frac = uniform(cfg.enc_mask_scale.0, cfg.enc_mask_scale.1);
    let enc_time_frac = uniform(cfg.enc_mask_scale.0, cfg.enc_mask_scale.1);
    let enc_mask = random_block_mask::<B>(n_r, n_t, enc_roi_frac, enc_time_frac, cfg.min_keep, device);

    // Collect encoder indices as a set for complement computation
    let enc_data = enc_mask.clone().squeeze::<1>().into_data();
    let enc_indices: Vec<i64> = enc_data.to_vec::<i64>().unwrap_or_default();
    let enc_set: std::collections::HashSet<i64> = enc_indices.into_iter().collect();

    // Complement: patches NOT in encoder context
    let complement: Vec<i64> = (0..n as i64).filter(|i| !enc_set.contains(i)).collect();

    // Predictor targets: sample from complement
    let mut pred_masks = Vec::with_capacity(3);

    for _ in 0..3 {
        let frac_r = uniform(cfg.pred_mask_r_scale.0, cfg.pred_mask_r_scale.1);
        let frac_t = uniform(cfg.pred_mask_t_scale.0, cfg.pred_mask_t_scale.1);
        let target_count = ((n as f64 * frac_r * frac_t).round() as usize)
            .max(cfg.min_keep)
            .min(complement.len());

        // Sample target_count indices from complement
        let mut sampled = complement.clone();
        fastrand_shuffle(&mut sampled);
        sampled.truncate(target_count);
        sampled.sort();

        let k = sampled.len();
        let mask = Tensor::<B, 1, Int>::from_data(
            TensorData::new(sampled, vec![k]),
            device,
        )
        .unsqueeze_dim::<2>(0);
        pred_masks.push(mask);
    }

    (enc_mask, pred_masks)
}

fn uniform(lo: f64, hi: f64) -> f64 {
    lo + fastrand::f64() * (hi - lo)
}

fn fastrand_shuffle(v: &mut [i64]) {
    for i in (1..v.len()).rev() {
        let j = fastrand::usize(..=i);
        v.swap(i, j);
    }
}
