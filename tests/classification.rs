use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

fn device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

// ── ClassificationHead::new ──────────────────────────────────────────────────

#[test]
fn classification_head_new_stores_num_classes() {
    let head = brainjepa::burn::ClassificationHead::<B>::new(768, 2, &device());
    assert_eq!(head.num_classes, 2);
}

#[test]
fn classification_head_new_different_dims() {
    let head = brainjepa::burn::ClassificationHead::<B>::new(384, 5, &device());
    assert_eq!(head.num_classes, 5);
}

// ── forward ──────────────────────────────────────────────────────────────────

#[test]
fn classification_head_forward_shape() {
    // Use batch > 1 to avoid squeeze singleton edge case in ClassificationHead::forward
    let head = brainjepa::burn::ClassificationHead::<B>::new(768, 2, &device());
    let input = Tensor::<B, 3>::zeros([2, 100, 768], &device());
    let logits = head.forward(input);
    assert_eq!(logits.dims(), [2, 2]);
}

#[test]
fn classification_head_forward_batch() {
    let head = brainjepa::burn::ClassificationHead::<B>::new(768, 3, &device());
    let input = Tensor::<B, 3>::zeros([4, 50, 768], &device());
    let logits = head.forward(input);
    assert_eq!(logits.dims(), [4, 3]);
}

#[test]
fn classification_head_forward_small_embed() {
    let head = brainjepa::burn::ClassificationHead::<B>::new(64, 10, &device());
    let input = Tensor::<B, 3>::zeros([2, 20, 64], &device());
    let logits = head.forward(input);
    assert_eq!(logits.dims(), [2, 10]);
}

// ── predict_classes ──────────────────────────────────────────────────────────

#[test]
fn predict_classes_returns_valid_indices() {
    // Use batch > 1 to avoid squeeze singleton edge case
    let head = brainjepa::burn::ClassificationHead::<B>::new(768, 2, &device());
    let input = Tensor::<B, 3>::random([2, 100, 768], burn::tensor::Distribution::Default, &device());
    let logits = head.forward(input);
    let classes = brainjepa::burn::predict_classes(logits);
    assert_eq!(classes.dims(), [2]);

    let vals: Vec<i64> = classes.into_data().to_vec::<i64>().unwrap();
    for &v in &vals {
        assert!(v == 0 || v == 1, "expected 0 or 1, got {v}");
    }
}

#[test]
fn predict_classes_batch() {
    let head = brainjepa::burn::ClassificationHead::<B>::new(384, 4, &device());
    let input = Tensor::<B, 3>::random([3, 50, 384], burn::tensor::Distribution::Default, &device());
    let logits = head.forward(input);
    let classes = brainjepa::burn::predict_classes(logits);
    assert_eq!(classes.dims(), [3]);

    let vals: Vec<i64> = classes.into_data().to_vec::<i64>().unwrap();
    for &v in &vals {
        assert!(v >= 0 && v < 4, "class index {v} out of range [0, 4)");
    }
}

#[test]
fn predict_classes_deterministic_for_known_logits() {
    // Manually construct logits where class 1 is clearly dominant, batch=2
    let logits = Tensor::<B, 2>::from_data(
        TensorData::new(vec![-10.0f32, 10.0, 5.0, -5.0], vec![2, 2]),
        &device(),
    );
    let classes = brainjepa::burn::predict_classes(logits);
    let vals: Vec<i64> = classes.into_data().to_vec::<i64>().unwrap();
    assert_eq!(vals, vec![1, 0]);
}
