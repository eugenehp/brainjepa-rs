/// Vision Transformer Encoder (burn 0.20.1)
///
/// Python: `VisionTransformer` in vision_transformer.py.
///
/// Architecture:
///   1. PatchEmbed: Conv2d(1, embed_dim, (1, ps), (1, ps)) -> [B, N, D]
///   2. Add gradient positional embeddings
///   3. Apply masks (index-gather for JEPA context selection)
///   4. 12 transformer blocks (pre-norm, GELU MLP)
///   5. Final LayerNorm
use burn::prelude::*;

use crate::model::block::Block;
use crate::model::norm::LNorm;
use crate::model::patch_embed::PatchEmbed;
use crate::model::pos_embed::GradientPosEmbed;

#[derive(Module, Debug)]
pub struct VisionTransformer<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub pos_embed: GradientPosEmbed<B>,
    pub blocks: Vec<Block<B>>,
    pub norm: LNorm<B>,
    pub embed_dim: usize,
    pub num_heads: usize,
}

impl<B: Backend> VisionTransformer<B> {
    pub fn new(
        img_size: (usize, usize),
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        norm_eps: f64,
        gradient_in_chan: usize,
        pos_mode: &str,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        let patch_embed: PatchEmbed<B> =
            PatchEmbed::new(img_size, patch_size, in_chans, embed_dim, device);
        let grid_size = patch_embed.num_patches_2d;

        let pos_embed =
            GradientPosEmbed::new(gradient_in_chan, embed_dim, grid_size, pos_mode, device)?;

        let blocks = (0..depth)
            .map(|_| Block::new(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_eps, device))
            .collect();

        let norm = LNorm::new(embed_dim, norm_eps, device);

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            embed_dim,
            num_heads,
        })
    }

    /// Forward pass.
    ///
    /// x: [B, 1, H, W] fMRI input
    /// gradient: brain gradient tensor for positional embeddings
    /// masks: optional list of index masks for JEPA context selection
    ///
    /// Returns: [B, N_masked, embed_dim] or [B, N, embed_dim] if no masks
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        gradient: Option<Tensor<B, 3>>,
        masks: Option<&[Tensor<B, 2, Int>]>,
    ) -> Tensor<B, 3> {
        // 1. Patch embed: [B, 1, H, W] -> [B, N, D]
        let mut x = self.patch_embed.forward(x);
        let [_b, _n, _d] = x.dims();

        // 2. Add positional embeddings
        let pos_emb = self.pos_embed.forward(gradient); // [1, N, D]
        x = x + pos_emb;

        // 3. Apply masks if present (gather context patches)
        if let Some(mask_list) = masks {
            x = apply_masks(x, mask_list);
        }

        // 4. Transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // 5. Final norm
        self.norm.forward(x)
    }
}

/// Gather patches from x using mask indices.
///
/// Python: `apply_masks(x, masks)` in src/masks/utils.py.
///
/// x: [B, N, D]
/// masks: list of [B, K] int tensors (indices into dim 1)
/// Returns: [B * len(masks), K, D]
pub fn apply_masks<B: Backend>(
    x: Tensor<B, 3>,
    masks: &[Tensor<B, 2, Int>],
) -> Tensor<B, 3> {
    let [_b, _n, d] = x.dims();
    let parts: Vec<Tensor<B, 3>> = masks
        .iter()
        .map(|m| {
            let [b_m, k] = m.dims();
            // Expand mask to [B, K, D] for gather
            let mask_exp = m.clone().unsqueeze_dim::<3>(2).expand([b_m, k, d]);
            x.clone().gather(1, mask_exp)
        })
        .collect();
    Tensor::cat(parts, 0)
}
