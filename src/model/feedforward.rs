/// MLP with fast GELU activation (burn 0.20.1)
///
/// Python: `MLP` in vision_transformer.py.
///   fc1: Linear(dim, hidden_dim)
///   fc2: Linear(hidden_dim, dim)
///   forward(x) = fc2(gelu(fc1(x)))
///
/// Uses the tanh GELU approximation instead of burn's erf-based GELU
/// (2-3x faster on CPU for large tensors).
use burn::nn::Linear;
use burn::prelude::*;

use crate::model::linear_zeros;

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            fc1: linear_zeros(dim, hidden_dim, true, device),
            fc2: linear_zeros(hidden_dim, dim, true, device),
        }
    }

    /// x: [B, N, dim] -> [B, N, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = fast_gelu(x);
        self.fc2.forward(x)
    }
}

/// Fast GELU using tanh approximation (avoids expensive erf).
/// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
fn fast_gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x3 = x.clone() * x.clone() * x.clone(); // x^3 without powf (safe for negatives)
    let inner = (x3.mul_scalar(0.044715f32) + x.clone()).mul_scalar(0.7978845608f32);
    x.mul_scalar(0.5f32) * (inner.tanh() + 1.0)
}
