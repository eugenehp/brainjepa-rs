# Parity gate (RLX vs Burn)

Tolerances are set from release runs on vit_base + HF weights; typical errors are much smaller.

## Run

```bash
bash scripts/parity.sh --quick
bash scripts/parity.sh
```

## RLX vs Burn (CPU)

| Check | Tolerance | Typical |
|-------|-----------|---------|
| Encoder (full) | **5×10⁻⁵** | ~2×10⁻⁵ |
| Predictor encoder context | **5×10⁻⁵** | ~2×10⁻⁵ |
| Predictor targets (mask 0; others need matching `n_pred`) | **5×10⁻⁵** | ~1.5×10⁻⁵ |

Preprocessing: RLX uses [`preprocess_fmri_f32`](../src/data.rs) matching Burn `standardize` (f32, `eps=1e-8`).

## RLX cross-backend vs RLX CPU

Each backend runs **full encoder** and **JEPA predictor** (compiled mask 0) against an RLX CPU reference.

| Backend | Tolerance (encoder + predictor) | Typical (Metal, MPSGraph on) |
|---------|-------------------------------|----------------------------|
| wgpu / MLX / CUDA / ROCm | **5×10⁻³** | ~3×10⁻⁵ (wgpu) |
| Metal (default, BHSD, BSNH) | **1×10⁻²** | ~1.6×10⁻⁵ |

Metal uses RLX MPSGraph lowering by default (`prepare_device` does not set `RLX_DISABLE_MPSGRAPH`). Set `RLX_DISABLE_MPSGRAPH=1` only for encode-latency experiments (predictor can drift ~3% vs CPU).

## Tests

- `parity_rlx_vs_burn` — encoder
- `parity_rlx_predictor_vs_burn` — JEPA vs Burn (mask 0; others need matching `n_pred`)
- `parity_rlx_cross_backend` — encoder + predictor on Metal / wgpu / MLX / CUDA / ROCm
- `standardize_parity` — preprocessing only
