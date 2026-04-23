/// Load pretrained Brain-JEPA weights from a safetensors/PyTorch checkpoint.
///
/// The checkpoint stores the encoder and predictor as separate state dicts.
/// Keys follow the pattern:
///   encoder.patch_embed.proj.weight        [embed_dim, 1, 1, patch_size]
///   encoder.blocks.{i}.norm1.weight        [embed_dim]
///   encoder.blocks.{i}.attn.qkv.weight     [3*embed_dim, embed_dim]
///   encoder.blocks.{i}.attn.proj.weight    [embed_dim, embed_dim]
///   encoder.blocks.{i}.norm2.weight        [embed_dim]
///   encoder.blocks.{i}.mlp.fc1.weight      [hidden_dim, embed_dim]
///   encoder.blocks.{i}.mlp.fc2.weight      [embed_dim, hidden_dim]
///   encoder.norm.weight                    [embed_dim]
///
///   predictor.predictor_embed.weight       [pred_dim, embed_dim]
///   predictor.predictor_blocks.{i}. ...
///   predictor.predictor_norm.weight        [pred_dim]
///   predictor.predictor_proj.weight        [embed_dim, pred_dim]
///   predictor.mask_token                   [1, 1, pred_dim]

use std::collections::HashMap;

use burn::module::{Param, ParamId};
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::config::ModelConfig;
use crate::model::encoder::VisionTransformer;
use crate::model::predictor::VisionTransformerPredictor;

// ── Raw tensor map ───────────────────────────────────────────────────────────

pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFilter {
    All,
    Encoder,
    Predictor,
    TargetEncoder,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        Self::from_file_filtered(path, WeightFilter::All)
    }

    pub fn from_file_filtered(path: &str, filter: WeightFilter) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (raw_key, view) in st.tensors() {
            // Strip "module." prefix from DDP-wrapped checkpoints
            let key = raw_key
                .strip_prefix("module.")
                .unwrap_or(raw_key.as_str())
                .to_string();

            // Apply filter
            match filter {
                WeightFilter::Encoder if !key.starts_with("encoder.") => continue,
                WeightFilter::Predictor if !key.starts_with("predictor.") => continue,
                WeightFilter::TargetEncoder if !key.starts_with("target_encoder.") => continue,
                _ => {}
            }

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by key, removing it from the map.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self
            .tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!(
                "rank mismatch for {key}: expected {N}, got {}",
                shape.len()
            );
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data, shape),
            device,
        ))
    }

    /// Check if a key exists.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }

    pub fn remaining(&self) -> usize {
        self.tensors.len()
    }
}

// ── Weight assignment helpers ────────────────────────────────────────────────

/// Assign a 2-D weight tensor (transposing from PyTorch [out, in] to burn [in, out]).
fn set_linear_w<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>) {
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
}

/// Assign weight + bias (transposing weight).
#[allow(dead_code)]
fn set_linear_wb<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
    linear.bias = Some(Param::initialized(ParamId::new(), b));
}

/// Assign LayerNorm weight and bias.
fn set_layernorm<B: Backend>(norm: &mut burn::nn::LayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.gamma = Param::initialized(ParamId::new(), w);
    norm.beta = Some(Param::initialized(ParamId::new(), b));
}

// ── Encoder weight loading ───────────────────────────────────────────────────

pub fn load_encoder_weights<B: Backend>(
    _cfg: &ModelConfig,
    wm: &mut WeightMap,
    enc: &mut VisionTransformer<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    // Patch embedding (Conv2d stored as [out, in, 1, ps] -> we need [in*ps, out])
    let conv_key = format!("{prefix}.patch_embed.proj.weight");
    if wm.has(&conv_key) {
        let conv_w: Tensor<B, 4> = wm.take(&conv_key, device)?;
        let [out_c, in_c, _h, ps] = conv_w.dims();
        // Reshape [out, in, 1, ps] -> [out, in*ps] then transpose for burn
        let w2d = conv_w.reshape([out_c, in_c * ps]);
        set_linear_w(&mut enc.patch_embed.proj, w2d);

        let bias_key = format!("{prefix}.patch_embed.proj.bias");
        if wm.has(&bias_key) {
            let b: Tensor<B, 1> = wm.take(&bias_key, device)?;
            enc.patch_embed.proj.bias = Some(Param::initialized(ParamId::new(), b));
        }
    }

    // Positional embedding gradient projection (if mapping mode)
    let grad_proj_key = format!("{prefix}.pos_embed_proj.predictor_pos_embed_proj.weight");
    if wm.has(&grad_proj_key) {
        if let Some(ref mut proj) = enc.pos_embed.grad_proj {
            let w: Tensor<B, 2> = wm.take(&grad_proj_key, device)?;
            set_linear_w(proj, w);
            let bias_key = format!("{prefix}.pos_embed_proj.predictor_pos_embed_proj.bias");
            if wm.has(&bias_key) {
                let b: Tensor<B, 1> = wm.take(&bias_key, device)?;
                proj.bias = Some(Param::initialized(ParamId::new(), b));
            }
        }
    }

    // Transformer blocks
    for (i, block) in enc.blocks.iter_mut().enumerate() {
        let p = format!("{prefix}.blocks.{i}");

        // norm1
        set_layernorm(
            &mut block.norm1.inner,
            wm.take(&format!("{p}.norm1.weight"), device)?,
            wm.take(&format!("{p}.norm1.bias"), device)?,
        );

        // attention QKV
        set_linear_w(
            &mut block.attn.qkv,
            wm.take(&format!("{p}.attn.qkv.weight"), device)?,
        );
        if wm.has(&format!("{p}.attn.qkv.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.attn.qkv.bias"), device)?;
            block.attn.qkv.bias = Some(Param::initialized(ParamId::new(), b));
        }

        // attention output projection
        set_linear_w(
            &mut block.attn.proj,
            wm.take(&format!("{p}.attn.proj.weight"), device)?,
        );
        if wm.has(&format!("{p}.attn.proj.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.attn.proj.bias"), device)?;
            block.attn.proj.bias = Some(Param::initialized(ParamId::new(), b));
        }

        // norm2
        set_layernorm(
            &mut block.norm2.inner,
            wm.take(&format!("{p}.norm2.weight"), device)?,
            wm.take(&format!("{p}.norm2.bias"), device)?,
        );

        // MLP
        set_linear_w(
            &mut block.mlp.fc1,
            wm.take(&format!("{p}.mlp.fc1.weight"), device)?,
        );
        if wm.has(&format!("{p}.mlp.fc1.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.mlp.fc1.bias"), device)?;
            block.mlp.fc1.bias = Some(Param::initialized(ParamId::new(), b));
        }
        set_linear_w(
            &mut block.mlp.fc2,
            wm.take(&format!("{p}.mlp.fc2.weight"), device)?,
        );
        if wm.has(&format!("{p}.mlp.fc2.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.mlp.fc2.bias"), device)?;
            block.mlp.fc2.bias = Some(Param::initialized(ParamId::new(), b));
        }
    }

    // Final norm
    set_layernorm(
        &mut enc.norm.inner,
        wm.take(&format!("{prefix}.norm.weight"), device)?,
        wm.take(&format!("{prefix}.norm.bias"), device)?,
    );

    Ok(())
}

// ── Predictor weight loading ─────────────────────────────────────────────────

pub fn load_predictor_weights<B: Backend>(
    _cfg: &ModelConfig,
    wm: &mut WeightMap,
    pred: &mut VisionTransformerPredictor<B>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    // predictor_embed
    set_linear_w(
        &mut pred.predictor_embed,
        wm.take(&format!("{prefix}.predictor_embed.weight"), device)?,
    );
    if wm.has(&format!("{prefix}.predictor_embed.bias")) {
        let b: Tensor<B, 1> = wm.take(&format!("{prefix}.predictor_embed.bias"), device)?;
        pred.predictor_embed.bias = Some(Param::initialized(ParamId::new(), b));
    }

    // mask_token
    if wm.has(&format!("{prefix}.mask_token")) {
        let mt: Tensor<B, 3> = wm.take(&format!("{prefix}.mask_token"), device)?;
        pred.mask_token = Param::initialized(ParamId::new(), mt);
    }

    // Gradient position projection
    let grad_proj_key = format!("{prefix}.predictor_2dpe_proj.predictor_pos_embed_proj.weight");
    if wm.has(&grad_proj_key) {
        if let Some(ref mut proj) = pred.pos_embed.grad_proj {
            let w: Tensor<B, 2> = wm.take(&grad_proj_key, device)?;
            set_linear_w(proj, w);
            let bias_key =
                format!("{prefix}.predictor_2dpe_proj.predictor_pos_embed_proj.bias");
            if wm.has(&bias_key) {
                let b: Tensor<B, 1> = wm.take(&bias_key, device)?;
                proj.bias = Some(Param::initialized(ParamId::new(), b));
            }
        }
    }

    // Predictor blocks
    for (i, block) in pred.predictor_blocks.iter_mut().enumerate() {
        let p = format!("{prefix}.predictor_blocks.{i}");

        set_layernorm(
            &mut block.norm1.inner,
            wm.take(&format!("{p}.norm1.weight"), device)?,
            wm.take(&format!("{p}.norm1.bias"), device)?,
        );

        set_linear_w(
            &mut block.attn.qkv,
            wm.take(&format!("{p}.attn.qkv.weight"), device)?,
        );
        if wm.has(&format!("{p}.attn.qkv.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.attn.qkv.bias"), device)?;
            block.attn.qkv.bias = Some(Param::initialized(ParamId::new(), b));
        }

        set_linear_w(
            &mut block.attn.proj,
            wm.take(&format!("{p}.attn.proj.weight"), device)?,
        );
        if wm.has(&format!("{p}.attn.proj.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.attn.proj.bias"), device)?;
            block.attn.proj.bias = Some(Param::initialized(ParamId::new(), b));
        }

        set_layernorm(
            &mut block.norm2.inner,
            wm.take(&format!("{p}.norm2.weight"), device)?,
            wm.take(&format!("{p}.norm2.bias"), device)?,
        );

        set_linear_w(
            &mut block.mlp.fc1,
            wm.take(&format!("{p}.mlp.fc1.weight"), device)?,
        );
        if wm.has(&format!("{p}.mlp.fc1.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.mlp.fc1.bias"), device)?;
            block.mlp.fc1.bias = Some(Param::initialized(ParamId::new(), b));
        }
        set_linear_w(
            &mut block.mlp.fc2,
            wm.take(&format!("{p}.mlp.fc2.weight"), device)?,
        );
        if wm.has(&format!("{p}.mlp.fc2.bias")) {
            let b: Tensor<B, 1> = wm.take(&format!("{p}.mlp.fc2.bias"), device)?;
            block.mlp.fc2.bias = Some(Param::initialized(ParamId::new(), b));
        }
    }

    // Final norm
    set_layernorm(
        &mut pred.predictor_norm.inner,
        wm.take(&format!("{prefix}.predictor_norm.weight"), device)?,
        wm.take(&format!("{prefix}.predictor_norm.bias"), device)?,
    );

    // Output projection
    set_linear_w(
        &mut pred.predictor_proj,
        wm.take(&format!("{prefix}.predictor_proj.weight"), device)?,
    );
    if wm.has(&format!("{prefix}.predictor_proj.bias")) {
        let b: Tensor<B, 1> = wm.take(&format!("{prefix}.predictor_proj.bias"), device)?;
        pred.predictor_proj.bias = Some(Param::initialized(ParamId::new(), b));
    }

    Ok(())
}
