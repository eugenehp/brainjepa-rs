/// Transformer Block with pre-norm (burn 0.20.1)
///
/// Python: `Block` in vision_transformer.py.
///   h   = x + Attn(LayerNorm(x))
///   out = h + MLP(LayerNorm(h))
use burn::prelude::*;

use crate::model::attention::Attention;
use crate::model::feedforward::MLP;
use crate::model::norm::LNorm;

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub norm1: LNorm<B>,
    pub attn: Attention<B>,
    pub norm2: LNorm<B>,
    pub mlp: MLP<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let mlp_hidden = (dim as f64 * mlp_ratio) as usize;
        Self {
            norm1: LNorm::new(dim, norm_eps, device),
            attn: Attention::new(dim, num_heads, qkv_bias, device),
            norm2: LNorm::new(dim, norm_eps, device),
            mlp: MLP::new(dim, mlp_hidden, device),
        }
    }

    /// x: [B, N, dim] -> [B, N, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = x.clone() + self.attn.forward(self.norm1.forward(x));
        h.clone() + self.mlp.forward(self.norm2.forward(h))
    }
}
