//! # brainjepa-rs — Brain-JEPA fMRI Foundation Model inference in Rust
//!
//! Pure-Rust inference for the [Brain-JEPA](https://github.com/hzlab/Brain-JEPA)
//! fMRI foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! Brain-JEPA maps parcellated fMRI time series (450 ROIs × T time points)
//! to latent representations using a Vision Transformer with:
//! - **Brain gradient positioning** for ROI spatial embeddings
//! - **Temporal patch embedding** via 1D convolution along time
//! - **JEPA architecture** (encoder + predictor with momentum target)
//!
//! ## Three entry points
//!
//! | Type | Loads | Use case |
//! |---|---|---|
//! | [`BrainJepaEncoder`] | encoder only | produce latent embeddings |
//! | [`BrainJepaPredictor`] | encoder + predictor | JEPA evaluation with masking |
//! | [`ClassificationHead`] | classification layer | downstream classification |
//!
//! ## Quick start — encode fMRI
//!
//! ```rust,ignore
//! use brainjepa::{BrainJepaEncoder, ModelConfig, DataConfig};
//!
//! let (enc, ms) = BrainJepaEncoder::<B>::from_weights(
//!     "model.safetensors",
//!     "gradient_mapping_450.csv",
//!     &ModelConfig::default(),
//!     &DataConfig::default(),
//!     &device,
//! )?;
//! let result = enc.encode_safetensors("data/fmri.safetensors")?;
//! result.save_safetensors("embeddings.safetensors")?;
//! ```
//!
//! ## Backends
//!
//! | Feature | Backend | Notes |
//! |---|---|---|
//! | `ndarray` (default) | CPU (NdArray + Rayon) | Add `blas-accelerate` on macOS |
//! | `wgpu` | GPU (Metal / Vulkan) | `--no-default-features --features wgpu` |
//! | `wgpu-f16` | GPU (half precision) | `--no-default-features --features wgpu-f16` |

// ── Thread configuration ─────────────────────────────────────────────────────

/// Configure the global Rayon thread pool.
///
/// Call this **once**, before any model operations.
/// Returns the actual number of threads in the pool.
pub fn init_threads(n: Option<usize>) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(count) = n {
        if count > 0 {
            builder = builder.num_threads(count);
        }
    }
    let _ = builder.build_global();
    rayon::current_num_threads()
}

// ── Internal modules ─────────────────────────────────────────────────────────

pub mod classification;
pub mod config;
pub mod csv_export;
pub mod data;
pub mod error;
pub mod hf_download;
pub mod inference;
pub mod masks;
pub mod model;
pub mod predictor_api;
pub mod prelude;
pub mod weights;

// ── Flat re-exports ──────────────────────────────────────────────────────────

// Configs
pub use config::{DataConfig, ModelConfig, YamlConfig};

// Encoder-only inference
pub use inference::{BrainJepaEncoder, EmbeddingResult};

// Encoder + Predictor (JEPA evaluation)
pub use predictor_api::BrainJepaPredictor;

// Classification head
pub use classification::{ClassificationHead, predict_classes};

// Data types
pub use data::{FmriInput, GradientData};

// Masking
pub use masks::{MaskConfig, full_context_mask, jepa_masks};

// Errors
pub use error::{BrainJepaError, Result};

// Model internals (advanced usage)
pub use model::encoder::apply_masks;

// Weights
pub use weights::{WeightFilter, WeightMap};

// CSV export
pub use csv_export::save_embeddings_csv;

// HuggingFace download
pub use hf_download::{resolve as resolve_weights, ResolvedWeights, DEFAULT_REPO};
