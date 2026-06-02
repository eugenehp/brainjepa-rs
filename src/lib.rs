//! # brainjepa-rs — Brain-JEPA fMRI Foundation Model inference in Rust
//!
//! Default inference uses [RLX](https://docs.rs/rlx). [Burn](https://burn.dev) is
//! optional (`burn-engine`) for benchmarks and parity comparison only.
//!
//! | Feature | Binaries | Use |
//! |---------|----------|-----|
//! | `rlx-engine` (default) | `infer`, `classify` | Production inference |
//! | `burn-engine` | `infer-burn` | Parity / benchmarks |

#[cfg(not(any(feature = "burn", feature = "rlx")))]
compile_error!("enable at least one inference engine: `rlx-engine` (default) and/or `burn-engine`");

/// Configure the global Rayon thread pool.
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

pub mod config;
pub mod csv_export;
pub mod data;
pub mod error;
pub mod hf_download;
pub mod masks;
pub mod prelude;

#[cfg(feature = "rlx")]
pub mod rlx;

#[cfg(feature = "burn")]
pub mod burn {
    //! Burn reference implementation — parity and benchmarks only.
    pub use crate::classification::{predict_classes, ClassificationHead};
    pub use crate::inference::{BrainJepaEncoder, EmbeddingResult};
    pub use crate::predictor_api::BrainJepaPredictor;
    pub use crate::masks::mask_config_for;
    pub use crate::data::FmriInput;
    pub use crate::masks::{
        full_context_mask_tensor as full_context_mask, jepa_masks_tensor as jepa_masks,
        random_block_mask_tensor as random_block_mask, MaskConfig,
    };
    pub use crate::model::encoder::apply_masks;
    pub use crate::weights::{WeightFilter, WeightMap};
}

#[cfg(feature = "burn")]
mod classification;
#[cfg(feature = "burn")]
mod inference;
#[cfg(feature = "burn")]
mod model;
#[cfg(feature = "burn")]
mod predictor_api;
#[cfg(feature = "burn")]
mod weights;

pub use config::{DataConfig, ModelConfig, YamlConfig};
pub use data::{FmriInputF32, GradientData};
pub use error::{BrainJepaError, Result};
pub use csv_export::save_embeddings_csv;
pub use hf_download::{resolve as resolve_weights, ResolvedWeights, DEFAULT_REPO};
pub use masks::{
    full_context_mask, jepa_masks, mask_config_for, random_block_mask, MaskConfig,
};

#[cfg(feature = "rlx")]
pub use rlx::{
    predict_class, AttnLayout, BrainJepaEncoder, BrainJepaPredictor,
    RlxClassificationHead as ClassificationHead,
};

#[cfg(feature = "rlx")]
pub use rlx::EmbeddingResult;

#[cfg(all(feature = "burn", not(feature = "rlx")))]
pub use inference::EmbeddingResult;
