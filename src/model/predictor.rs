/// Vision Transformer Predictor (burn 0.20.1)
///
/// Python: `VisionTransformerPredictor` in vision_transformer.py.
///
/// Takes encoder context embeddings and predicts target patch representations:
///   1. Project encoder dim -> predictor dim
///   2. Add positional embeddings to context tokens
///   3. Initialize learnable mask tokens with target positions
///   4. Concatenate context + mask tokens
///   5. Run through predictor transformer blocks
///   6. Extract mask token predictions
///   7. Project back to encoder dim
use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::prelude::*;

use crate::model::block::Block;
use crate::model::encoder::apply_masks;
use crate::model::norm::LNorm;
use crate::model::pos_embed::GradientPosEmbed;
use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct VisionTransformerPredictor<B: Backend> {
    /// Project encoder dim -> predictor dim
    pub predictor_embed: Linear<B>,
    /// Learnable mask token: [1, 1, pred_dim]
    pub mask_token: Param<Tensor<B, 3>>,
    /// Gradient positional embedding for predictor
    pub pos_embed: GradientPosEmbed<B>,
    /// Predictor transformer blocks
    pub predictor_blocks: Vec<Block<B>>,
    /// Final LayerNorm
    pub predictor_norm: LNorm<B>,
    /// Project predictor dim -> encoder dim
    pub predictor_proj: Linear<B>,
    pub predictor_embed_dim: usize,
    pub embed_dim: usize,
}

impl<B: Backend> VisionTransformerPredictor<B> {
    pub fn new(
        _num_patches: usize,
        num_patches_2d: (usize, usize),
        embed_dim: usize,
        predictor_embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        norm_eps: f64,
        gradient_in_chan: usize,
        pos_mode: &str,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        let predictor_embed =
            linear_zeros(embed_dim, predictor_embed_dim, true, device);

        let mask_token = Param::initialized(
            ParamId::new(),
            Tensor::zeros([1, 1, predictor_embed_dim], device),
        );

        let pos_embed = GradientPosEmbed::new(
            gradient_in_chan,
            predictor_embed_dim,
            num_patches_2d,
            pos_mode,
            device,
        )?;

        let predictor_blocks = (0..depth)
            .map(|_| {
                Block::new(
                    predictor_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    norm_eps,
                    device,
                )
            })
            .collect();

        let predictor_norm = LNorm::new(predictor_embed_dim, norm_eps, device);
        let predictor_proj =
            linear_zeros(predictor_embed_dim, embed_dim, true, device);

        Ok(Self {
            predictor_embed,
            mask_token,
            pos_embed,
            predictor_blocks,
            predictor_norm,
            predictor_proj,
            predictor_embed_dim,
            embed_dim,
        })
    }

    /// Forward pass.
    ///
    /// x: [B_enc, N_ctxt, encoder_dim]  — encoder context features
    /// gradient: brain gradient tensor for positional embeddings
    /// masks_x: context masks [B, K_ctx] — which patches are context
    /// masks: target masks [B, K_pred] — which patches to predict
    ///
    /// Returns: [B_total, K_pred, encoder_dim] — predicted target representations
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        gradient: Option<Tensor<B, 3>>,
        masks_x: &[Tensor<B, 2, Int>],
        masks: &[Tensor<B, 2, Int>],
    ) -> Tensor<B, 3> {
        let b = x.dims()[0] / masks_x.len();

        // 1. Project to predictor dim
        let mut x = self.predictor_embed.forward(x);

        // 2. Add positional embeddings to context
        let pos_emb = self.pos_embed.forward(gradient.clone()); // [1, N, pred_dim]
        let [_, n_pos, d_pos] = pos_emb.dims();
        let pos_emb_ctx = pos_emb.clone().expand([b, n_pos, d_pos]);
        let ctx_pos = apply_masks(pos_emb_ctx, masks_x);
        x = x + ctx_pos;

        let [_, n_ctxt, _d] = x.dims();

        // 3. Build mask tokens with target positions
        let pos_emb_tgt = pos_emb.expand([b, n_pos, d_pos]);
        let tgt_pos = apply_masks(pos_emb_tgt, masks);
        // Repeat target positions for each context mask variation
        let tgt_pos = repeat_interleave_batch(tgt_pos, b, masks_x.len());

        let [n_total, n_pred, pred_dim] = tgt_pos.dims();
        let pred_tokens = self.mask_token.val().expand([n_total, n_pred, pred_dim]);
        let pred_tokens = pred_tokens + tgt_pos;

        // 4. Concatenate context + mask tokens
        let x = x.repeat_dim(0, masks.len());
        let x = Tensor::cat(vec![x, pred_tokens], 1); // [B_total, N_ctxt + N_pred, pred_dim]

        // 5. Transformer blocks
        let mut x = x;
        for block in &self.predictor_blocks {
            x = block.forward(x);
        }

        // 6. Norm and extract predictions
        let x = self.predictor_norm.forward(x);
        let x = x.narrow(1, n_ctxt, n_pred); // take only the predicted tokens

        // 7. Project back to encoder dim
        self.predictor_proj.forward(x)
    }
}

/// Repeat batch interleave: replicate each batch element `repeat` times.
///
/// Python: `repeat_interleave_batch(x, B, repeat)` in tensors.py.
fn repeat_interleave_batch<B: Backend>(
    x: Tensor<B, 3>,
    batch_size: usize,
    repeat: usize,
) -> Tensor<B, 3> {
    let n = x.dims()[0] / batch_size;
    let parts: Vec<Tensor<B, 3>> = (0..n)
        .flat_map(|i| {
            let chunk = x.clone().narrow(0, i * batch_size, batch_size);
            (0..repeat).map(move |_| chunk.clone())
        })
        .collect();
    Tensor::cat(parts, 0)
}
