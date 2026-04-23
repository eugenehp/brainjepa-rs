/// 2D Sinusoidal Positional Embeddings with Brain Gradient Positioning (burn 0.20.1)
///
/// Python: `GradTs_2dPE` in vision_transformer.py.
///
/// The height dimension uses sincos positional encoding (fixed).
/// The width (temporal) dimension can use:
///   - 'mapping': learned projection from brain gradient coordinates
///   - 'origin': standard sincos on grid indices
///
/// Output: [1, H*W, embed_dim]
use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::prelude::*;

use crate::error::BrainJepaError;
use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct GradientPosEmbed<B: Backend> {
    /// Fixed sincos embeddings for the height (ROI) dimension: [H*W, D/2]
    pub emb_h: Param<Tensor<B, 2>>,
    /// Learned projection from gradient coords to embed_dim/2 (for 'mapping' mode)
    pub grad_proj: Option<Linear<B>>,
    /// Fixed sincos for width dimension (for 'origin' mode): [H*W, D/2]
    pub emb_w: Option<Param<Tensor<B, 2>>>,
    pub embed_dim: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub mode: String,
}

impl<B: Backend> GradientPosEmbed<B> {
    pub fn new(
        in_chan: usize,
        embed_dim: usize,
        grid_size: (usize, usize),
        mode: &str,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        let (gh, gw) = grid_size;
        let n = gh * gw;
        let half_dim = embed_dim / 2;

        // Height (ROI) positional embeddings: fixed sincos
        let emb_h_data = sincos_1d_grid(half_dim, gh, gw);
        let emb_h = Param::initialized(
            ParamId::new(),
            Tensor::<B, 2>::from_data(TensorData::new(emb_h_data, vec![n, half_dim]), device),
        );

        let (grad_proj, emb_w) = match mode {
            "mapping" => {
                let proj = linear_zeros(in_chan, half_dim, true, device);
                (Some(proj), None)
            }
            "origin" => {
                let emb_w_data = sincos_1d_width(half_dim, gh, gw);
                let t = Param::initialized(
                    ParamId::new(),
                    Tensor::<B, 2>::from_data(
                        TensorData::new(emb_w_data, vec![n, half_dim]),
                        device,
                    ),
                );
                (None, Some(t))
            }
            _ => {
                return Err(BrainJepaError::InvalidPosMode {
                    mode: mode.to_string(),
                })
            }
        };

        Ok(Self {
            emb_h,
            grad_proj,
            emb_w,
            embed_dim,
            grid_h: gh,
            grid_w: gw,
            mode: mode.to_string(),
        })
    }

    /// gradient: [1, N_rois, grad_dim]  (brain gradient coordinates)
    /// Returns: [1, H*W, embed_dim]
    pub fn forward(&self, gradient: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let emb_w = if self.mode == "mapping" {
            // Both grad and grad_proj are guaranteed present when mode == "mapping"
            // (enforced by the constructor match arm).
            let grad = gradient.expect("BUG: gradient tensor required for 'mapping' mode");
            let proj = self.grad_proj.as_ref().expect("BUG: grad_proj missing in mapping mode");
            let projected = proj.forward(grad).squeeze::<2>(); // [N_rois, D/2]
            let repeated = repeat_interleave_dim0(projected, self.grid_w);
            let min_val: f32 = repeated.clone().min().into_scalar().elem();
            let max_val: f32 = repeated.clone().max().into_scalar().elem();
            let range = max_val - min_val;
            repeated
                .sub_scalar(min_val)
                .div_scalar(range)
                .mul_scalar(2.0f32)
                .sub_scalar(1.0f32)
        } else {
            // origin mode — emb_w is guaranteed present by the constructor.
            self.emb_w
                .as_ref()
                .expect("BUG: emb_w missing in origin mode")
                .val()
        };

        let emb = Tensor::cat(vec![self.emb_h.val(), emb_w], 1);
        emb.unsqueeze_dim::<3>(0)
    }
}

/// Repeat each row of a 2D tensor `repeats` times along dim 0.
fn repeat_interleave_dim0<B: Backend>(t: Tensor<B, 2>, repeats: usize) -> Tensor<B, 2> {
    let [n, d] = t.dims();
    t.unsqueeze_dim::<3>(1)
        .expand([n, repeats, d])
        .reshape([n * repeats, d])
}

/// Generate 1D sincos positional embeddings for the height (ROI) dimension.
fn sincos_1d_grid(half_dim: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let n = grid_h * grid_w;
    let quarter = half_dim / 2;
    let mut data = vec![0.0f32; n * half_dim];

    for h in 0..grid_h {
        for w in 0..grid_w {
            let pos = h as f64;
            let idx = h * grid_w + w;
            for k in 0..quarter {
                let omega = 1.0 / 10000.0_f64.powf(k as f64 / quarter as f64);
                let angle = pos * omega;
                data[idx * half_dim + k] = angle.sin() as f32;
                data[idx * half_dim + quarter + k] = angle.cos() as f32;
            }
        }
    }
    data
}

/// Generate 1D sincos positional embeddings for the width (temporal) dimension.
fn sincos_1d_width(half_dim: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let n = grid_h * grid_w;
    let quarter = half_dim / 2;
    let mut data = vec![0.0f32; n * half_dim];

    for h in 0..grid_h {
        for w in 0..grid_w {
            let pos = w as f64;
            let idx = h * grid_w + w;
            for k in 0..quarter {
                let omega = 1.0 / 10000.0_f64.powf(k as f64 / quarter as f64);
                let angle = pos * omega;
                data[idx * half_dim + k] = angle.sin() as f32;
                data[idx * half_dim + quarter + k] = angle.cos() as f32;
            }
        }
    }
    data
}
