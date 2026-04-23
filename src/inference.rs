/// Brain-JEPA encoder inference — produce fMRI embeddings.
///
/// The encoder maps fMRI time series to latent representations.
/// At inference, we run the full encoder without masking.
///
/// # Usage
/// ```rust,ignore
/// use brainjepa::BrainJepaEncoder;
///
/// let enc = BrainJepaEncoder::<B>::from_weights(
///     "model.safetensors",
///     "gradient_mapping_450.csv",
///     &ModelConfig::default(),
///     &DataConfig::default(),
///     &device,
/// )?;
///
/// let embeddings = enc.encode_safetensors("data/fmri.safetensors")?;
/// ```
use std::path::Path;
use std::time::Instant;

use burn::prelude::*;

use crate::config::{DataConfig, ModelConfig};
use crate::data::{self, FmriInput, GradientData};
use crate::error::BrainJepaError;
use crate::model::encoder::VisionTransformer;
use crate::weights::{load_encoder_weights, WeightMap};

// ── Output types ─────────────────────────────────────────────────────────────

/// Encoder embedding output.
pub struct EmbeddingResult {
    /// Latent embeddings: row-major f32, shape [n_patches, embed_dim]
    pub embeddings: Vec<f32>,
    /// Shape: [n_patches, embed_dim]
    pub shape: Vec<usize>,
    /// Number of ROI patches
    pub n_rois: usize,
    /// Number of temporal patches
    pub n_time_patches: usize,
    /// Encoding time in milliseconds
    pub ms_encode: f64,
}

impl EmbeddingResult {
    /// Total number of output patches (n_rois * n_time_patches).
    pub fn n_patches(&self) -> usize {
        self.n_rois * self.n_time_patches
    }

    /// Output embedding dimension (768 for ViT-Base).
    pub fn embed_dim(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(0)
    }

    /// Save embeddings to a safetensors file.
    pub fn save_safetensors(&self, path: &str) -> anyhow::Result<()> {
        use safetensors::{Dtype, View};
        use std::borrow::Cow;

        struct RawTensor {
            data: Vec<u8>,
            shape: Vec<usize>,
        }
        impl View for RawTensor {
            fn dtype(&self) -> Dtype { Dtype::F32 }
            fn shape(&self) -> &[usize] { &self.shape }
            fn data(&self) -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize { self.data.len() }
        }

        let bytes: Vec<u8> = self.embeddings.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = RawTensor {
            data: bytes,
            shape: self.shape.clone(),
        };

        let pairs: Vec<(&str, RawTensor)> = vec![("embeddings", tensor)];
        let out = safetensors::serialize(pairs, None)?;
        std::fs::write(path, out)?;
        Ok(())
    }
}

// ── BrainJepaEncoder ─────────────────────────────────────────────────────────

/// Brain-JEPA fMRI encoder for producing latent embeddings.
pub struct BrainJepaEncoder<B: Backend> {
    encoder: VisionTransformer<B>,
    gradient: Tensor<B, 3>,
    pub model_cfg: ModelConfig,
    pub data_cfg: DataConfig,
    device: B::Device,
}

impl<B: Backend> BrainJepaEncoder<B> {
    /// Load encoder from safetensors weights and gradient CSV.
    ///
    /// Returns `(encoder, weight_load_ms)`.
    ///
    /// # Errors
    ///
    /// - [`BrainJepaError::FileNotFound`] if weights or gradient path doesn't exist
    /// - [`BrainJepaError::GradientRoiMismatch`] if gradient ROIs don't match `data_cfg.crop_size.0`
    /// - [`BrainJepaError::InvalidPosMode`] if `model_cfg.pos_mode` is invalid
    pub fn from_weights(
        weights_path: &str,
        gradient_csv_path: &str,
        model_cfg: &ModelConfig,
        data_cfg: &DataConfig,
        device: &B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        // Pre-validate file existence
        if !Path::new(weights_path).exists() {
            return Err(BrainJepaError::FileNotFound {
                kind: "weights",
                path: weights_path.into(),
            }
            .into());
        }

        // Load gradient (file-existence checked inside from_csv)
        let grad_data = GradientData::from_csv(gradient_csv_path)?;

        // Validate gradient ROI count matches expected crop size
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
            1, // in_chans
            model_cfg.embed_dim,
            model_cfg.depth,
            model_cfg.num_heads,
            model_cfg.mlp_ratio,
            true, // qkv_bias
            model_cfg.norm_eps,
            grad_data.grad_dim,
            &model_cfg.pos_mode,
            device,
        )?;

        // Load weights
        let t = Instant::now();
        let mut wm = WeightMap::from_file(weights_path)?;
        let prefix = if wm.has("target_encoder.blocks.0.norm1.weight") {
            "target_encoder"
        } else {
            "encoder"
        };
        load_encoder_weights(model_cfg, &mut wm, &mut encoder, prefix, device)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("Loaded encoder weights ({} remaining keys)", wm.remaining());

        Ok((
            Self {
                encoder,
                gradient,
                model_cfg: model_cfg.clone(),
                data_cfg: data_cfg.clone(),
                device: device.clone(),
            },
            ms,
        ))
    }

    /// One-line description of the loaded encoder.
    pub fn describe(&self) -> String {
        format!(
            "Brain-JEPA encoder  embed_dim={}  depth={}  heads={}  patch={}",
            self.model_cfg.embed_dim,
            self.model_cfg.depth,
            self.model_cfg.num_heads,
            self.model_cfg.patch_size,
        )
    }

    /// Encode fMRI data from a safetensors file.
    pub fn encode_safetensors(&self, fmri_path: &str) -> anyhow::Result<EmbeddingResult> {
        let input = data::load_fmri_safetensors::<B>(fmri_path, &self.device)?;
        self.encode_input(input)
    }

    /// Encode fMRI data from a CSV file.
    pub fn encode_csv(&self, csv_path: &str) -> anyhow::Result<EmbeddingResult> {
        let input = data::load_fmri_csv::<B>(csv_path, &self.device)?;
        self.encode_input(input)
    }

    /// Encode a raw tensor input.
    pub fn encode_tensor(&self, data: Tensor<B, 4>) -> anyhow::Result<EmbeddingResult> {
        let [_b, _c, n_rois, n_time] = data.dims();
        let input = FmriInput { data, n_rois, n_time };
        self.encode_input(input)
    }

    /// Encode multiple safetensors files.
    ///
    /// Returns one `EmbeddingResult` per file, in the same order as the input.
    pub fn encode_safetensors_batch(
        &self,
        paths: &[impl AsRef<str>],
    ) -> anyhow::Result<Vec<EmbeddingResult>> {
        paths
            .iter()
            .map(|p| {
                let input = data::load_fmri_safetensors::<B>(p.as_ref(), &self.device)?;
                self.encode_input(input)
            })
            .collect()
    }

    /// Encode multiple CSV files.
    pub fn encode_csv_batch(
        &self,
        paths: &[impl AsRef<str>],
    ) -> anyhow::Result<Vec<EmbeddingResult>> {
        paths
            .iter()
            .map(|p| {
                let input = data::load_fmri_csv::<B>(p.as_ref(), &self.device)?;
                self.encode_input(input)
            })
            .collect()
    }

    /// Reference to the Burn device this encoder was loaded on.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    fn encode_input(&self, input: FmriInput<B>) -> anyhow::Result<EmbeddingResult> {
        let mut x = input.data;

        // Optional temporal downsampling
        let target = self.data_cfg.crop_size.1;
        if input.n_time != target && self.data_cfg.downsample {
            x = data::temporal_downsample(x, target)?;
        }

        // Standardize
        x = data::standardize(x);

        let n_time_patches = target / self.model_cfg.patch_size;

        let t = Instant::now();
        let enc_out = self.encoder.forward(x, Some(self.gradient.clone()), None);
        let ms_encode = t.elapsed().as_secs_f64() * 1000.0;

        let [_b, n_patches, embed_dim] = enc_out.dims();

        let embeddings = tensor_data_to_f32(enc_out.squeeze::<2>().into_data())
            .map_err(|e| BrainJepaError::TensorConversion { reason: e })?;

        Ok(EmbeddingResult {
            embeddings,
            shape: vec![n_patches, embed_dim],
            n_rois: input.n_rois,
            n_time_patches,
            ms_encode,
        })
    }
}

/// Convert TensorData bytes to Vec<f32>, handling both f32 and f16 element types.
fn tensor_data_to_f32(data: burn::tensor::TensorData) -> Result<Vec<f32>, String> {
    if let Ok(v) = data.to_vec::<f32>() {
        return Ok(v);
    }
    let converted = data.clone().convert::<f32>();
    if let Ok(v) = converted.to_vec::<f32>() {
        return Ok(v);
    }
    let bytes = &data.bytes;
    if bytes.len() % 2 == 0 {
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        return Ok(values);
    }
    Err(format!("cannot convert tensor data ({} bytes) to f32", bytes.len()))
}
