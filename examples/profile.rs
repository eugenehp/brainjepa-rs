/// Profile per-layer costs to find bottlenecks.
use std::time::Instant;
use burn::prelude::*;
use burn::backend::NdArray;
use burn::tensor::activation::{softmax, gelu};

type B = NdArray;

fn main() {
    let d = burn::backend::ndarray::NdArrayDevice::Cpu;
    let _ = brainjepa::init_threads(None);

    let embed = 768usize;
    let heads = 12usize;
    let seq = 4500usize;
    let hdim = embed / heads;
    let mlp_h = embed * 4;
    let iters = 3usize;

    println!("Profile: seq={seq} embed={embed} heads={heads} hdim={hdim} mlp_h={mlp_h}  ({iters} iters)");
    println!();

    let x: Tensor<B, 2> = Tensor::random([seq, embed], burn::tensor::Distribution::Normal(0.0, 1.0), &d);

    // QKV projection: [S, D] @ [D, 3D] -> [S, 3D]
    let w_qkv: Tensor<B, 2> = Tensor::random([embed, 3*embed], burn::tensor::Distribution::Normal(0.0, 0.01), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = x.clone().matmul(w_qkv.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("QKV proj       [4500,768]@[768,2304]:  {ms:>8.1} ms");

    // Q@K^T: [H, S, Dh] @ [H, Dh, S] -> [H, S, S]
    let q: Tensor<B, 3> = Tensor::random([heads, seq, hdim], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let k: Tensor<B, 3> = Tensor::random([heads, seq, hdim], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = q.clone().matmul(k.clone().transpose()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Q@K^T          [12,4500,64]@[12,64,4500]: {ms:>8.1} ms");

    // Softmax over [H, S, S]
    let scores: Tensor<B, 3> = Tensor::random([heads, seq, seq], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = softmax(scores.clone(), 2); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Softmax        [12,4500,4500]:         {ms:>8.1} ms");

    // Attn@V: [H, S, S] @ [H, S, Dh] -> [H, S, Dh]
    let attn: Tensor<B, 3> = Tensor::random([heads, seq, seq], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let v: Tensor<B, 3> = Tensor::random([heads, seq, hdim], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = attn.clone().matmul(v.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Attn@V         [12,4500,4500]@[12,4500,64]: {ms:>8.1} ms");

    // Output projection: [S, D] @ [D, D] -> [S, D]
    let w_o: Tensor<B, 2> = Tensor::random([embed, embed], burn::tensor::Distribution::Normal(0.0, 0.01), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = x.clone().matmul(w_o.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Out proj       [4500,768]@[768,768]:   {ms:>8.1} ms");

    // MLP fc1: [S, D] @ [D, 4D] -> [S, 4D]
    let w1: Tensor<B, 2> = Tensor::random([embed, mlp_h], burn::tensor::Distribution::Normal(0.0, 0.01), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = x.clone().matmul(w1.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("MLP fc1        [4500,768]@[768,3072]:  {ms:>8.1} ms");

    // GELU: [S, 4D]
    let h: Tensor<B, 2> = Tensor::random([seq, mlp_h], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = gelu(h.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("GELU           [4500,3072]:            {ms:>8.1} ms");

    // MLP fc2: [S, 4D] @ [4D, D] -> [S, D]
    let w2: Tensor<B, 2> = Tensor::random([mlp_h, embed], burn::tensor::Distribution::Normal(0.0, 0.01), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = h.clone().matmul(w2.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("MLP fc2        [4500,3072]@[3072,768]: {ms:>8.1} ms");

    // LayerNorm
    let ln = burn::nn::LayerNormConfig::new(embed).with_epsilon(1e-6).init::<B>(&d);
    let x3: Tensor<B, 3> = Tensor::random([1, seq, embed], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
    let t0 = Instant::now();
    for _ in 0..iters { let _ = ln.forward(x3.clone()); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("LayerNorm      [1,4500,768]:           {ms:>8.1} ms");

    println!();
    println!("Total for 12 blocks ~ 12 * (QKV + Q@K + Softmax + A@V + OutProj + fc1 + GELU + fc2 + 2*LN)");
}
