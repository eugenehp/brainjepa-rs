# RLX-first inference

**Production:** `rlx-engine` (default) — no Burn dependency.

**Reference:** `burn-engine` — parity tests, `infer-burn`, `brainjepa::burn::*` only.

## Binaries

| Binary | Engine | Purpose |
|--------|--------|---------|
| `infer` | RLX | Encode fMRI → embeddings |
| `classify` | RLX | Encoder + classification head |
| `predict` | RLX | JEPA encoder + predictor (masked) |
| `infer-burn` | Burn | Reference encode |

## Crate API (RLX at root)

- `BrainJepaEncoder`, `BrainJepaPredictor`, `ClassificationHead`
- `masks::{full_context_mask, jepa_masks, mask_config_for}`

## Parity (CI)

```bash
bash scripts/parity.sh --quick   # encoder + predictor vs Burn
bash scripts/parity.sh           # + RLX cross-backend (Metal/wgpu/…)
```

Measured on vit_base + HF weights:

| Check | Tolerance | Typical |
|-------|-----------|---------|
| Encoder RLX vs Burn | 1e-4 | ~2e-5 |
| Predictor RLX vs Burn | 1e-3 | ~1e-5 |

## Build

```bash
cargo build --release
cargo build --release --features rlx-engine,rlx-metal
cargo build --no-default-features --features burn-engine --bin infer-burn
```

See [PARITY.md](./PARITY.md) for backend layout and env vars.

Burn is not required for production builds. Remove `burn-engine` from your dependency only after you no longer need `parity_rlx_*` regression tests.
