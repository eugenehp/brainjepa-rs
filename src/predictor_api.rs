/// Public JEPA predictor API — encoder + predictor pipeline.
///
/// Loads both the encoder and predictor from a checkpoint and runs the
/// full JEPA pipeline: encoder produces context representations, predictor
/// predicts target patch representations from context + mask tokens.
///
/// # Usage
/// ```rust,ignore
/// use brainjepa::BrainJepaPredictor;
///
/// let (jepa, ms) = BrainJepaPredictor::<B>::from_weights(
///     "data/brainjepa.safetensors",
///     "data/gradient_mapping_450.csv",
///     &ModelConfig::default(),
///     &DataConfig::default(),
///     &device,
/// )?;
/// let (enc_out, pred_out) = jepa.predict(fmri_tensor, &enc_masks, &pred_masks)?;
/// ```
use std::time::Instant;

use burn::prelude::*;

use crate::config::{DataConfig, ModelConfig};
use crate::data::GradientData;
use crate::error::BrainJepaError;
use crate::model::encoder::VisionTransformer;
use crate::model::predictor::VisionTransformerPredictor;
use crate::weights::{load_encoder_weights, load_predictor_weights, WeightMap};

/// Brain-JEPA encoder + predictor for JEPA evaluation.
pub struct BrainJepaPredictor<B: Backend> {
    pub encoder: VisionTransformer<B>,
    pub predictor: VisionTransformerPredictor<B>,
    pub gradient: Tensor<B, 3>,
    pub model_cfg: ModelConfig,
    pub data_cfg: DataConfig,
    device: B::Device,
}

impl<B: Backend> BrainJepaPredictor<B> {
    /// Load encoder + predictor from a safetensors checkpoint.
    ///
    /// Returns `(model, weight_load_ms)`.
    pub fn from_weights(
        weights_path: &str,
        gradient_csv_path: &str,
        model_cfg: &ModelConfig,
        data_cfg: &DataConfig,
        device: &B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        if !std::path::Path::new(weights_path).exists() {
            return Err(BrainJepaError::FileNotFound {
                kind: "weights",
                path: weights_path.into(),
            }
            .into());
        }

        let grad_data = GradientData::from_csv(gradient_csv_path)?;
        let expected_rois = data_cfg.crop_size.0;
        if grad_data.n_rois != expected_rois {
            return Err(BrainJepaError::GradientRoiMismatch {
                expected: expected_rois,
                got: grad_data.n_rois,
            }
            .into());
        }
        let gradient = grad_data.to_tensor::<B>(device);

        // Build encoder
        let mut encoder = VisionTransformer::new(
            data_cfg.crop_size,
            model_cfg.patch_size,
            1,
            model_cfg.embed_dim,
            model_cfg.depth,
            model_cfg.num_heads,
            model_cfg.mlp_ratio,
            true,
            model_cfg.norm_eps,
            grad_data.grad_dim,
            &model_cfg.pos_mode,
            device,
        )?;

        // Build predictor
        let num_patches = encoder.patch_embed.num_patches;
        let num_patches_2d = encoder.patch_embed.num_patches_2d;
        let mut predictor = VisionTransformerPredictor::new(
            num_patches,
            num_patches_2d,
            model_cfg.embed_dim,
            model_cfg.pred_emb_dim,
            model_cfg.pred_depth,
            model_cfg.num_heads,
            model_cfg.mlp_ratio,
            true,
            model_cfg.norm_eps,
            grad_data.grad_dim,
            &model_cfg.pos_mode,
            device,
        )?;

        // Load weights
        let t = Instant::now();
        let mut wm = WeightMap::from_file(weights_path)?;

        let enc_prefix = if wm.has("target_encoder.blocks.0.norm1.weight") {
            "target_encoder"
        } else {
            "encoder"
        };
        load_encoder_weights(model_cfg, &mut wm, &mut encoder, enc_prefix, device)?;
        load_predictor_weights(model_cfg, &mut wm, &mut predictor, "predictor", device)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        println!(
            "Loaded encoder ({enc_prefix}) + predictor weights ({} remaining keys)",
            wm.remaining()
        );

        Ok((
            Self {
                encoder,
                predictor,
                gradient,
                model_cfg: model_cfg.clone(),
                data_cfg: data_cfg.clone(),
                device: device.clone(),
            },
            ms,
        ))
    }

    /// Run the full JEPA pipeline: encode context, predict targets.
    ///
    /// x: [B, 1, H, W] fMRI input
    /// enc_masks: context masks (which patches the encoder sees)
    /// pred_masks: target masks (which patches to predict)
    ///
    /// Returns (encoder_output, predictor_output):
    ///   encoder_output: [B_enc, N_ctx, embed_dim]
    ///   predictor_output: [B_total, N_pred, embed_dim]
    pub fn predict(
        &self,
        x: Tensor<B, 4>,
        enc_masks: &[Tensor<B, 2, Int>],
        pred_masks: &[Tensor<B, 2, Int>],
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let enc_out =
            self.encoder
                .forward(x, Some(self.gradient.clone()), Some(enc_masks));

        let pred_out = self.predictor.forward(
            enc_out.clone(),
            Some(self.gradient.clone()),
            enc_masks,
            pred_masks,
        );

        (enc_out, pred_out)
    }

    /// Run encoder only (same as BrainJepaEncoder but using this model's weights).
    pub fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        self.encoder
            .forward(x, Some(self.gradient.clone()), None)
    }

    /// Describe the loaded model.
    pub fn describe(&self) -> String {
        format!(
            "Brain-JEPA  encoder: {}-dim x {} layers  predictor: {}-dim x {} layers",
            self.model_cfg.embed_dim,
            self.model_cfg.depth,
            self.model_cfg.pred_emb_dim,
            self.model_cfg.pred_depth,
        )
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}
