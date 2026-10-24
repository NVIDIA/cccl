// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Compile a host-side provider without a CUDA context or a device consumer.

use cuda_coop_codegen::{Algorithm, Operation, Provider, Request};
use std::{env, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = env::args().collect();
    if arguments.len() != 4 {
        return Err("usage: specialize CUDA_TOOLKIT COMPUTE_TARGET OUTPUT_LTOIR".into());
    }
    // This example is built inside the selected source checkout. Consumers
    // supply their own explicit source root instead of relying on their cwd.
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let started = Instant::now();
    let provider = Provider::from_checkout(checkout, &arguments[1])?;
    let initialized = started.elapsed();
    let request = Request::new(Operation::LoadFilled, Algorithm::Transpose, 128, 4)?;
    let started = Instant::now();
    let artifact = provider.compile(&[request], &arguments[2])?;
    let compiled = started.elapsed();
    std::fs::write(&arguments[3], &artifact.ltoir)?;
    println!("symbol: {}", artifact.specializations[0].symbol);
    println!(
        "CUB layout: {:?}; scratch: {:?}; scratch free: {}",
        artifact.specializations[0].cub_layout,
        artifact.specializations[0].scratch,
        artifact.specializations[0].scratch_free
    );
    println!("initialization: {initialized:?}; compilation: {compiled:?}");
    print!("{}", artifact.compiler_log);
    Ok(())
}
