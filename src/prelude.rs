//! Convenience re-exports for common usage.

pub use crate::config::{DataConfig, ModelConfig, YamlConfig};
pub use crate::csv_export::save_embeddings_csv;
pub use crate::data::{FmriInputF32, GradientData};
pub use crate::error::{BrainJepaError, Result};
pub use crate::hf_download::{resolve as resolve_weights, ResolvedWeights, DEFAULT_REPO};
pub use crate::masks::{full_context_mask, jepa_masks, random_block_mask, MaskConfig};

#[cfg(feature = "rlx")]
pub use crate::{
    predict_class, BrainJepaEncoder, BrainJepaPredictor, ClassificationHead, EmbeddingResult,
};

#[cfg(feature = "burn")]
pub use crate::burn::{
    predict_classes, apply_masks, full_context_mask as burn_full_context_mask,
    jepa_masks as burn_jepa_masks, BrainJepaEncoder as BurnBrainJepaEncoder,
    BrainJepaPredictor as BurnBrainJepaPredictor, ClassificationHead as BurnClassificationHead,
    EmbeddingResult as BurnEmbeddingResult, FmriInput, WeightFilter, WeightMap,
};
