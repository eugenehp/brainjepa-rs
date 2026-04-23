/// Temporal Patch Embedding via Conv2d (burn 0.20.1)
///
/// Python: `PatchEmbed` in vision_transformer.py.
///   Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
///
/// Input: [B, 1, 450, T] fMRI data (450 ROIs × T time steps)
/// Patches only along the time dimension: each ROI keeps its identity.
/// Output: [B, 450 * (T / patch_size), embed_dim]
use burn::nn::Linear;
use burn::prelude::*;

use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    /// Linear projection simulating Conv2d(1, embed_dim, kernel=(1, ps), stride=(1, ps)).
    /// We slice temporal windows manually and project with a linear layer.
    pub proj: Linear<B>,
    pub patch_size: usize,
    pub num_patches: usize,
    pub num_patches_2d: (usize, usize),
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(
        img_size: (usize, usize), // (n_rois, n_time)
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        device: &B::Device,
    ) -> Self {
        let n_rois = img_size.0;
        let n_time_patches = img_size.1 / patch_size;
        let num_patches = n_rois * n_time_patches;

        // Conv2d(1, embed_dim, (1, patch_size), (1, patch_size)) is equivalent to
        // a linear projection of each (1 * patch_size) window.
        let proj = linear_zeros(in_chans * patch_size, embed_dim, true, device);

        Self {
            proj,
            patch_size,
            num_patches,
            num_patches_2d: (n_rois, n_time_patches),
        }
    }

    /// x: [B, 1, H, W] -> [B, H * (W/ps), embed_dim]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _c, h, w] = x.dims();
        let ps = self.patch_size;
        let n_t = w / ps;

        // Reshape: [B, 1, H, W] -> [B, H, W] -> [B, H, n_t, ps]
        let x = x.reshape([b, h, w]);
        let x = x.reshape([b, h, n_t, ps]);

        // Transpose to [B, H, n_t, ps] -> [B, H * n_t, ps]
        let x = x.reshape([b, h * n_t, ps]);

        // Project: [B, H*n_t, ps] -> [B, H*n_t, embed_dim]
        self.proj.forward(x)
    }
}
