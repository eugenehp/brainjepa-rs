#!/bin/bash
# benchmark.sh — Brain-JEPA multi-backend benchmark
#
# Usage:
#   bash benchmark.sh                 # build & bench all backends, 3 runs each
#   bash benchmark.sh --runs 5        # 5 iterations per backend
#   bash benchmark.sh --no-build      # skip cargo build (use existing binaries)
#   bash benchmark.sh --gpu-only      # skip CPU backends, only bench wgpu
#
# On macOS  -> builds ndarray, ndarray+accelerate, wgpu (Metal)
# On Linux  -> builds ndarray, wgpu (Vulkan if GPU present)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ───────────────────────────────────────────────────────────────
RUNS=3; NO_BUILD=0; GPU_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --runs)      shift; RUNS="$1" ;;
        --runs=*)    RUNS="${1#--runs=}" ;;
        --no-build)  NO_BUILD=1 ;;
        --gpu-only)  GPU_ONLY=1 ;;
        -h|--help)
            printf 'Usage: bash %s [--runs N] [--no-build] [--gpu-only]\n' "$0"
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { printf '\033[31m✗  %s\033[0m\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;34m━━━  %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m  %s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m  %s\n' "$*"; }
bold() { printf '\033[1m%s\033[0m' "$*"; }

# ── Platform detect ──────────────────────────────────────────────────────────
OS="$(uname -s)"
NCPUS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
export RAYON_NUM_THREADS="$NCPUS"

if [ "$OS" = "Darwin" ]; then
    PLATFORM="macOS"
else
    PLATFORM="Linux"
fi

# ── Cargo / Rust ─────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
command -v cargo >/dev/null 2>&1 || die "cargo not found — install Rust: https://rustup.rs"

# ── Data files ───────────────────────────────────────────────────────────────
WEIGHTS="$SCRIPT_DIR/data/brainjepa.safetensors"
GRADIENT="$SCRIPT_DIR/data/gradient_mapping_450.csv"
INPUT="$SCRIPT_DIR/data/test_fmri.safetensors"
OUTPUT="/tmp/brainjepa_bench_embeddings.safetensors"

[ -f "$WEIGHTS"  ] || die "Weights not found: $WEIGHTS"
[ -f "$GRADIENT" ] || die "Gradient mapping not found: $GRADIENT"
[ -f "$INPUT"    ] || die "Test fMRI not found: $INPUT"

# ── Define backends ──────────────────────────────────────────────────────────
# Each backend:  label | feature flags | binary suffix
BACKENDS=()

if [ "$GPU_ONLY" = "0" ]; then
    BACKENDS+=("ndarray|--features ndarray|ndarray")

    if [ "$PLATFORM" = "macOS" ]; then
        BACKENDS+=("accelerate|--features ndarray,accelerate|accelerate")
    fi
fi

BACKENDS+=("wgpu|--no-default-features --features wgpu|wgpu")

# ── Build target ─────────────────────────────────────────────────────────────
TARGET_DIR=/tmp/brainjepa-bench-target

# ── Header ───────────────────────────────────────────────────────────────────
step "Brain-JEPA benchmark  —  $PLATFORM ($NCPUS threads)"
info "runs=$RUNS  no-build=$NO_BUILD  gpu-only=$GPU_ONLY"
info "weights  : $WEIGHTS"
info "gradient : $GRADIENT"
info "input    : $INPUT"

# ── Step 1: Build ────────────────────────────────────────────────────────────
step "[1/3] Build"

build_backend() {
    local label="$1" features="$2" suffix="$3"
    local bin_out="/tmp/brainjepa-${suffix}"

    if [ "$NO_BUILD" = "1" ]; then
        if [ -f "$bin_out" ]; then
            ok "$label: skipped (--no-build), binary exists"
            return 0
        else
            warn "$label: --no-build but no binary at $bin_out — skipping"
            return 1
        fi
    fi

    info "Building $label ..."
    info "  cargo build --release $features --bin infer"

    if CARGO_TARGET_DIR="$TARGET_DIR" \
        cargo build --release $features --bin infer 2>&1 \
        | grep -E "^(error|warning\[|   Compiling|    Finished)" || true; then

        cp "$TARGET_DIR/release/infer" "$bin_out"
        chmod +x "$bin_out"
        ok "$label  ->  $bin_out"
        return 0
    else
        warn "$label: build failed — skipping"
        return 1
    fi
}

BUILT_BACKENDS=()
for entry in "${BACKENDS[@]}"; do
    IFS='|' read -r label features suffix <<< "$entry"
    if build_backend "$label" "$features" "$suffix"; then
        BUILT_BACKENDS+=("$entry")
    fi
done

if [ ${#BUILT_BACKENDS[@]} -eq 0 ]; then
    die "No backends were built successfully."
fi

# ── Step 2: Benchmark ────────────────────────────────────────────────────────
step "[2/3] Benchmark  ($RUNS iterations each)"

# Associative arrays for results
declare -A ENCODE_BEST
declare -A WEIGHTS_BEST
declare -A TOTAL_BEST
declare -A ENCODE_ALL

run_backend() {
    local label="$1" suffix="$2"
    local bin="/tmp/brainjepa-${suffix}"

    if [ ! -x "$bin" ]; then
        warn "$label: binary not found at $bin — skipping"
        return
    fi

    info ""
    info "--- $label ---"
    local best_encode=999999999
    local best_weights=999999999
    local best_total=999999999
    local all_encodes=""

    for i in $(seq 1 "$RUNS"); do
        info "  Run $i/$RUNS ..."

        # Capture stderr (contains TIMING line) while letting stdout print
        local stderr_file
        stderr_file=$(mktemp /tmp/brainjepa_bench_stderr.XXXXXX)

        "$bin" \
            --weights "$WEIGHTS" \
            --gradient "$GRADIENT" \
            --input "$INPUT" \
            --output "$OUTPUT" \
            2>"$stderr_file" || {
                warn "  Run $i failed"
                rm -f "$stderr_file"
                continue
            }

        # Parse TIMING line: TIMING weights=Xms encode=Xms total=Xms
        local timing_line
        timing_line=$(grep '^TIMING ' "$stderr_file" 2>/dev/null || true)
        rm -f "$stderr_file"

        if [ -z "$timing_line" ]; then
            warn "  No TIMING line found in stderr"
            continue
        fi

        local w_ms e_ms t_ms
        w_ms=$(echo "$timing_line" | sed -n 's/.*weights=\([0-9.]*\)ms.*/\1/p')
        e_ms=$(echo "$timing_line" | sed -n 's/.*encode=\([0-9.]*\)ms.*/\1/p')
        t_ms=$(echo "$timing_line" | sed -n 's/.*total=\([0-9.]*\)ms.*/\1/p')

        info "  weights=${w_ms}ms  encode=${e_ms}ms  total=${t_ms}ms"

        # Track all encode times for this backend
        if [ -n "$all_encodes" ]; then
            all_encodes="${all_encodes},${e_ms}"
        else
            all_encodes="${e_ms}"
        fi

        # Update best (integer comparison via awk for floats)
        best_encode=$(awk "BEGIN { print ($e_ms < $best_encode) ? $e_ms : $best_encode }")
        best_weights=$(awk "BEGIN { print ($w_ms < $best_weights) ? $w_ms : $best_weights }")
        best_total=$(awk "BEGIN { print ($t_ms < $best_total) ? $t_ms : $best_total }")
    done

    if [ "$best_encode" = "999999999" ]; then
        warn "$label: no successful runs"
        return
    fi

    ENCODE_BEST[$suffix]="$best_encode"
    WEIGHTS_BEST[$suffix]="$best_weights"
    TOTAL_BEST[$suffix]="$best_total"
    ENCODE_ALL[$suffix]="$all_encodes"

    ok "$label  best-of-${RUNS}:  encode=${best_encode}ms  weights=${best_weights}ms  total=${best_total}ms"
}

for entry in "${BUILT_BACKENDS[@]}"; do
    IFS='|' read -r label features suffix <<< "$entry"
    run_backend "$label" "$suffix"
done

# ── Step 3: Summary table ────────────────────────────────────────────────────
step "[3/3] Summary"

if [ ${#ENCODE_BEST[@]} -eq 0 ]; then
    die "No successful benchmark results."
fi

# Find the slowest encode time for speedup ratio (baseline = slowest)
baseline_encode=0
baseline_label=""
for entry in "${BUILT_BACKENDS[@]}"; do
    IFS='|' read -r label features suffix <<< "$entry"
    if [ -n "${ENCODE_BEST[$suffix]+x}" ]; then
        is_slower=$(awk "BEGIN { print (${ENCODE_BEST[$suffix]} > $baseline_encode) }")
        if [ "$is_slower" = "1" ]; then
            baseline_encode="${ENCODE_BEST[$suffix]}"
            baseline_label="$label"
        fi
    fi
done

info "Baseline (slowest): $baseline_label @ ${baseline_encode}ms"
info ""

# Print table header
printf '  \033[1m%-24s  %10s  %10s  %10s  %10s\033[0m\n' \
    "Backend" "Weights" "Encode" "Total" "Speedup"
printf '  %-24s  %10s  %10s  %10s  %10s\n' \
    "------------------------" "----------" "----------" "----------" "----------"

for entry in "${BUILT_BACKENDS[@]}"; do
    IFS='|' read -r label features suffix <<< "$entry"
    if [ -z "${ENCODE_BEST[$suffix]+x}" ]; then
        printf '  %-24s  %10s  %10s  %10s  %10s\n' \
            "$label" "FAIL" "FAIL" "FAIL" "-"
        continue
    fi

    local_encode="${ENCODE_BEST[$suffix]}"
    local_weights="${WEIGHTS_BEST[$suffix]}"
    local_total="${TOTAL_BEST[$suffix]}"
    speedup=$(awk "BEGIN { printf \"%.2f\", $baseline_encode / $local_encode }")

    # Color fastest green
    if [ "$speedup" = "1.00" ]; then
        printf '  %-24s  %8sms  %8sms  %8sms  %9sx\n' \
            "$label" "$local_weights" "$local_encode" "$local_total" "$speedup"
    else
        printf '  \033[32m%-24s  %8sms  %8sms  %8sms  %9sx\033[0m\n' \
            "$label" "$local_weights" "$local_encode" "$local_total" "$speedup"
    fi
done

info ""

# Print per-run details
info "Per-run encode times (ms):"
for entry in "${BUILT_BACKENDS[@]}"; do
    IFS='|' read -r label features suffix <<< "$entry"
    if [ -n "${ENCODE_ALL[$suffix]+x}" ]; then
        info "  $label: ${ENCODE_ALL[$suffix]}"
    fi
done

info ""
info "Platform : $PLATFORM  ($NCPUS threads)"
info "Runs     : $RUNS per backend"
info "Binary   : /tmp/brainjepa-{backend}"
ok "Done."
