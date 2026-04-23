/// Multi-Head Self-Attention (burn 0.20.1)
///
/// Python: `Attention` in vision_transformer.py.
/// Uses packed QKV projection and scaled dot-product attention.
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, num_heads: usize, qkv_bias: bool, device: &B::Device) -> Self {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5) as f32;
        Self {
            qkv: linear_zeros(dim, dim * 3, qkv_bias, device),
            proj: linear_zeros(dim, dim, true, device),
            num_heads,
            head_dim,
            scale,
        }
    }

    /// x: [B, N, C] -> [B, N, C]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let h = self.num_heads;
        let dh = self.head_dim;

        // QKV: [B, N, 3*C] -> [B, N, 3, H, Dh]
        let qkv = self.qkv.forward(x);
        let qkv = qkv.reshape([b, n, 3, h, dh]);

        let q = qkv.clone().narrow(2, 0, 1).reshape([b, n, h, dh]);
        let k = qkv.clone().narrow(2, 1, 1).reshape([b, n, h, dh]);
        let v = qkv.narrow(2, 2, 1).reshape([b, n, h, dh]);

        // Transpose to [B, H, N, Dh]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Attention
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(self.scale), 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2).reshape([b, n, c]);
        self.proj.forward(out)
    }
}
