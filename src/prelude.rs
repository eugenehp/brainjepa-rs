/// Convenience re-exports for common usage.
///
/// ```rust,ignore
/// use brainjepa::prelude::*;
/// ```
pub use crate::classification::{ClassificationHead, predict_classes};
pub use crate::config::{DataConfig, ModelConfig, YamlConfig};
pub use crate::csv_export::save_embeddings_csv;
pub use crate::data::{FmriInput, GradientData};
pub use crate::error::{BrainJepaError, Result};
pub use crate::hf_download::{resolve as resolve_weights, ResolvedWeights, DEFAULT_REPO};
pub use crate::inference::{BrainJepaEncoder, EmbeddingResult};
pub use crate::masks::{full_context_mask, jepa_masks, MaskConfig};
pub use crate::predictor_api::BrainJepaPredictor;
pub use crate::weights::{WeightFilter, WeightMap};
