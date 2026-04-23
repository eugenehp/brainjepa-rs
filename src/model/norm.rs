/// LayerNorm wrapper (burn 0.20.1)
///
/// Brain-JEPA uses pre-norm LayerNorm (eps=1e-6), matching PyTorch nn.LayerNorm.
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct LNorm<B: Backend> {
    pub inner: LayerNorm<B>,
}

impl<B: Backend> LNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: LayerNormConfig::new(dim).with_epsilon(eps).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}
