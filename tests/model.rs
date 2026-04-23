use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ── PatchEmbed ───────────────────────────────────────────────────────────────

#[test]
fn patch_embed_output_shape() {
    let patch_size = 16;
    let n_rois = 10;
    let n_time = 160;
    let embed_dim = 768;

    let pe = brainjepa::model::patch_embed::PatchEmbed::<B>::new(
        (n_rois, n_time),
        patch_size,
        1,
        embed_dim,
        &device(),
    );

    let input = Tensor::<B, 4>::zeros([1, 1, n_rois, n_time], &device());
    let output = pe.forward(input);

    let expected_patches = n_rois * (n_time / patch_size);
    assert_eq!(output.dims(), [1, expected_patches, embed_dim]);
}

#[test]
fn patch_embed_num_patches() {
    let pe = brainjepa::model::patch_embed::PatchEmbed::<B>::new(
        (20, 80),
        16,
        1,
        384,
        &device(),
    );
    assert_eq!(pe.num_patches, 20 * 5);
    assert_eq!(pe.num_patches_2d, (20, 5));
}

#[test]
fn patch_embed_batch_forward() {
    let pe = brainjepa::model::patch_embed::PatchEmbed::<B>::new(
        (8, 32),
        16,
        1,
        128,
        &device(),
    );
    let input = Tensor::<B, 4>::zeros([4, 1, 8, 32], &device());
    let output = pe.forward(input);
    assert_eq!(output.dims(), [4, 8 * 2, 128]);
}

// ── Block ────────────────────────────────────────────────────────────────────

#[test]
fn block_forward_preserves_shape() {
    let dim = 64;
    let num_heads = 4;
    let block = brainjepa::model::block::Block::<B>::new(
        dim, num_heads, 4.0, true, 1e-6, &device(),
    );

    let input = Tensor::<B, 3>::zeros([1, 100, dim], &device());
    let output = block.forward(input);
    assert_eq!(output.dims(), [1, 100, dim]);
}

#[test]
fn block_forward_batch() {
    let dim = 64;
    let block = brainjepa::model::block::Block::<B>::new(
        dim, 4, 4.0, true, 1e-6, &device(),
    );

    let input = Tensor::<B, 3>::zeros([2, 50, dim], &device());
    let output = block.forward(input);
    assert_eq!(output.dims(), [2, 50, dim]);
}

// ── Attention ────────────────────────────────────────────────────────────────

#[test]
fn attention_forward_preserves_shape() {
    let dim = 64;
    let num_heads = 4;
    let attn = brainjepa::model::attention::Attention::<B>::new(
        dim, num_heads, true, &device(),
    );

    let input = Tensor::<B, 3>::zeros([1, 100, dim], &device());
    let output = attn.forward(input);
    assert_eq!(output.dims(), [1, 100, dim]);
}

#[test]
fn attention_head_dim_and_scale() {
    let dim = 128;
    let num_heads = 8;
    let attn = brainjepa::model::attention::Attention::<B>::new(
        dim, num_heads, true, &device(),
    );
    assert_eq!(attn.num_heads, 8);
    assert_eq!(attn.head_dim, 16);
    let expected_scale = (16.0f64).powf(-0.5) as f32;
    assert!((attn.scale - expected_scale).abs() < 1e-6);
}

#[test]
fn attention_no_qkv_bias() {
    let attn = brainjepa::model::attention::Attention::<B>::new(
        64, 4, false, &device(),
    );
    assert!(attn.qkv.bias.is_none());
}

// ── MLP ──────────────────────────────────────────────────────────────────────

#[test]
fn mlp_forward_preserves_shape() {
    let dim = 64;
    let hidden = 256;
    let mlp = brainjepa::model::feedforward::MLP::<B>::new(dim, hidden, &device());

    let input = Tensor::<B, 3>::zeros([1, 100, dim], &device());
    let output = mlp.forward(input);
    assert_eq!(output.dims(), [1, 100, dim]);
}

#[test]
fn mlp_forward_batch() {
    let dim = 32;
    let hidden = 128;
    let mlp = brainjepa::model::feedforward::MLP::<B>::new(dim, hidden, &device());

    let input = Tensor::<B, 3>::zeros([3, 20, dim], &device());
    let output = mlp.forward(input);
    assert_eq!(output.dims(), [3, 20, dim]);
}

#[test]
fn mlp_zeros_produce_zeros() {
    // With zero-initialized weights, MLP(0) should be 0
    // gelu(0) = 0, and fc2(0) = 0
    let dim = 16;
    let mlp = brainjepa::model::feedforward::MLP::<B>::new(dim, 64, &device());

    let input = Tensor::<B, 3>::zeros([1, 5, dim], &device());
    let output = mlp.forward(input);

    use burn::prelude::ElementConversion;
    let max_abs: f32 = output.abs().max().into_scalar().elem();
    assert!(max_abs < 1e-6, "expected all zeros, max abs = {max_abs}");
}
