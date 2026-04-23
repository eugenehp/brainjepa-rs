use std::io::Write;
use brainjepa::{GradientData, BrainJepaError};

fn temp_csv(content: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

// ── GradientData ─────────────────────────────────────────────────────────────

#[test]
fn gradient_csv_valid() {
    let f = temp_csv("1.0,2.0,3.0\n4.0,5.0,6.0\n");
    let g = GradientData::from_csv(f.path().to_str().unwrap()).unwrap();
    assert_eq!(g.n_rois, 2);
    assert_eq!(g.grad_dim, 3);
    assert_eq!(g.values.len(), 6);
}

#[test]
fn gradient_csv_skips_comments_and_blanks() {
    let f = temp_csv("# header\n\n1.0,2.0\n\n3.0,4.0\n# end\n");
    let g = GradientData::from_csv(f.path().to_str().unwrap()).unwrap();
    assert_eq!(g.n_rois, 2);
    assert_eq!(g.grad_dim, 2);
}

#[test]
fn gradient_csv_inconsistent_columns() {
    let f = temp_csv("1.0,2.0,3.0\n4.0,5.0\n");
    let err = GradientData::from_csv(f.path().to_str().unwrap()).unwrap_err();
    match err {
        BrainJepaError::InconsistentCsvRow { row, expected, got, .. } => {
            assert_eq!(row, 2);
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        other => panic!("expected InconsistentCsvRow, got: {other}"),
    }
}

#[test]
fn gradient_csv_empty() {
    let f = temp_csv("# only comments\n\n");
    let err = GradientData::from_csv(f.path().to_str().unwrap()).unwrap_err();
    assert!(matches!(err, BrainJepaError::EmptyCsv { .. }));
}

#[test]
fn gradient_csv_file_not_found() {
    let err = GradientData::from_csv("/nonexistent/path.csv").unwrap_err();
    assert!(matches!(err, BrainJepaError::FileNotFound { .. }));
}

// ── fMRI CSV ─────────────────────────────────────────────────────────────────

#[test]
fn fmri_csv_valid() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let f = temp_csv("1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n9.0,10.0,11.0,12.0\n");
    let input = brainjepa::data::load_fmri_csv::<B>(
        f.path().to_str().unwrap(),
        &device,
    ).unwrap();
    assert_eq!(input.n_rois, 3);
    assert_eq!(input.n_time, 4);
    assert_eq!(input.data.dims(), [1, 1, 3, 4]);
}

#[test]
fn fmri_csv_inconsistent_columns() {
    let f = temp_csv("1.0,2.0,3.0\n4.0,5.0\n");
    let err = brainjepa::data::load_fmri_csv::<burn::backend::NdArray>(
        f.path().to_str().unwrap(),
        &burn::backend::ndarray::NdArrayDevice::Cpu,
    ).unwrap_err();
    assert!(matches!(err, BrainJepaError::InconsistentCsvRow { .. }));
}

#[test]
fn fmri_csv_empty() {
    let f = temp_csv("");
    let err = brainjepa::data::load_fmri_csv::<burn::backend::NdArray>(
        f.path().to_str().unwrap(),
        &burn::backend::ndarray::NdArrayDevice::Cpu,
    ).unwrap_err();
    assert!(matches!(err, BrainJepaError::EmptyCsv { .. }));
}

// ── Downsample ───────────────────────────────────────────────────────────────

#[test]
fn downsample_rejects_upscale() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let x = burn::prelude::Tensor::<B, 4>::zeros([1, 1, 10, 50], &device);
    let err = brainjepa::data::temporal_downsample(x, 100).unwrap_err();
    assert!(matches!(err, BrainJepaError::DownsampleUpscale { src: 50, dst: 100 }));
}

#[test]
fn downsample_identity() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let x = burn::prelude::Tensor::<B, 4>::zeros([1, 1, 10, 160], &device);
    let y = brainjepa::data::temporal_downsample(x, 160).unwrap();
    assert_eq!(y.dims(), [1, 1, 10, 160]);
}

#[test]
fn downsample_halves() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let x = burn::prelude::Tensor::<B, 4>::zeros([1, 1, 10, 200], &device);
    let y = brainjepa::data::temporal_downsample(x, 100).unwrap();
    assert_eq!(y.dims(), [1, 1, 10, 100]);
}

// ── Standardize ──────────────────────────────────────────────────────────────

#[test]
fn standardize_produces_zero_mean() {
    use burn::backend::NdArray;
    type B = NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let x = burn::prelude::Tensor::<B, 4>::from_data(
        burn::prelude::TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 2, 3],
        ),
        &device,
    );
    let y = brainjepa::data::standardize(x);
    use burn::prelude::ElementConversion;
    let mean: f32 = y.mean().into_scalar().elem();
    assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
}
