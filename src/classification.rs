/// Classification head for downstream tasks.
///
/// Wraps the Brain-JEPA encoder with a linear classification layer:
///   embeddings = encoder(fmri)        → [B, N, embed_dim]
///   pooled     = mean(embeddings, 1)  → [B, embed_dim]
///   logits     = head(pooled)         → [B, num_classes]
///
/// Weights for the classification head can be loaded from a downstream
/// checkpoint (safetensors).
use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::prelude::*;

use crate::model::linear_zeros;

/// Linear classification head with global average pooling.
#[derive(Module, Debug)]
pub struct ClassificationHead<B: Backend> {
    pub fc_norm: burn::nn::LayerNorm<B>,
    pub head: Linear<B>,
    pub num_classes: usize,
}

impl<B: Backend> ClassificationHead<B> {
    /// Create a new classification head.
    pub fn new(embed_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        Self {
            fc_norm: burn::nn::LayerNormConfig::new(embed_dim)
                .with_epsilon(1e-6)
                .init(device),
            head: linear_zeros(embed_dim, num_classes, true, device),
            num_classes,
        }
    }

    /// Forward pass: pool encoder output and classify.
    ///
    /// encoder_output: [B, N, embed_dim]
    /// Returns: [B, num_classes] logits
    pub fn forward(&self, encoder_output: Tensor<B, 3>) -> Tensor<B, 2> {
        let [b, _n, d] = encoder_output.dims();
        // Global average pooling over patches: [B, N, D] -> [B, 1, D] -> [B, D]
        let pooled = encoder_output.mean_dim(1).reshape([b, d]);
        let normed = self.fc_norm.forward(pooled);
        self.head.forward(normed)
    }

    /// Load classification head weights from a safetensors file.
    ///
    /// Expected keys (shapes in parens):
    ///   `head.weight`    (num_classes x embed_dim)
    ///   `head.bias`      (num_classes)
    ///   `fc_norm.weight` (embed_dim)
    ///   `fc_norm.bias`   (embed_dim)
    pub fn load_weights(
        &mut self,
        wm: &mut crate::weights::WeightMap,
        prefix: &str,
        device: &<B as Backend>::Device,
    ) -> anyhow::Result<()> {
        // Head
        if wm.has(&format!("{prefix}.head.weight")) {
            let w: Tensor<B, 2> = wm.take(&format!("{prefix}.head.weight"), device)?;
            self.head.weight = Param::initialized(ParamId::new(), w.transpose());
        }
        if wm.has(&format!("{prefix}.head.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{prefix}.head.bias"), device)?;
            self.head.bias = Some(Param::initialized(ParamId::new(), b));
        }

        // LayerNorm
        if wm.has(&format!("{prefix}.fc_norm.weight")) {
            let w: Tensor<B, 1> = wm.take(&format!("{prefix}.fc_norm.weight"), device)?;
            self.fc_norm.gamma = Param::initialized(ParamId::new(), w);
        }
        if wm.has(&format!("{prefix}.fc_norm.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{prefix}.fc_norm.bias"), device)?;
            self.fc_norm.beta = Some(Param::initialized(ParamId::new(), b));
        }

        Ok(())
    }
}

/// Argmax along the last dimension — returns predicted class indices.
pub fn predict_classes<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1, Int> {
    let [b, _c] = logits.dims();
    logits.argmax(1).reshape([b])
}
