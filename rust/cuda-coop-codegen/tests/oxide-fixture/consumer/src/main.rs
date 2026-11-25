// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#[cfg(not(feature = "storage"))]
use cuda_device::{kernel, launch_bounds, launch_contract};
#[cfg(not(feature = "storage"))]
use cuda_host::cuda_module;

#[cfg(not(feature = "storage"))]
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn direct_1(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::roundtrip::<1, false>(input, output, post, valid, mode);
        }
    }
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn direct_4(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::roundtrip::<4, false>(input, output, post, valid, mode);
        }
    }
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn transpose_1(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::roundtrip::<1, true>(input, output, post, valid, mode);
        }
    }
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn transpose_4(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::roundtrip::<4, true>(input, output, post, valid, mode);
        }
    }
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (128, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn shared_helper_root(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::shared_roundtrip(input, output, post, valid, mode);
        }
    }
    #[cfg(feature = "root-context")]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (64, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn smaller_1(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::roundtrip::<1, true>(input, output, post, valid, mode);
        }
    }
    #[cfg(feature = "root-context")]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(block = (64, 1, 1), domain = 3, dynamic_shared = inferred)]
    pub unsafe fn smaller_4(
        input: *const i32,
        output: *mut i32,
        post: *mut i32,
        valid: i32,
        mode: u32,
    ) {
        unsafe {
            coop_marker_fixture_helper::shared_roundtrip(input, output, post, valid, mode);
        }
    }
}

#[cfg(not(feature = "storage"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::{CudaContext, DeviceBuffer};
    use cuda_core::{LaunchConfig3D, LaunchContractError};
    use oxide_artifacts::{ArtifactPayloadKind, read_artifact_bundles_from_object_bytes};
    let executable = std::fs::read(std::env::current_exe()?)?;
    let bundles = read_artifact_bundles_from_object_bytes(&executable)?;
    let bundle = bundles
        .iter()
        .find(|b| {
            b.entry("transpose_4").is_some() && b.payload(ArtifactPayloadKind::Cubin).is_some()
        })
        .ok_or("missing embedded native artifact")?;
    let cubin = bundle.payload(ArtifactPayloadKind::Cubin).unwrap();
    std::fs::write("fixture.cubin", cubin)?;
    let trap_mode = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("--trap-before") => 8,
        Some("--trap-after-load") => 16,
        Some(_) => return Err("expected --trap-before or --trap-after-load".into()),
    };
    if trap_mode != 0 && !cfg!(feature = "checked") {
        return Err("trap modes require the checked fixture".into());
    }
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the embedded artifact is generated from these kernel declarations.
    let module = unsafe { kernels::load(&context)? };
    macro_rules! check_plan {
        ($prepare:ident, $bytes:expr, $alignment:expr, $threads:expr) => {{
            let prepared =
                module.$prepare(LaunchConfig3D::auto_shared((1, 1, 1), ($threads, 1, 1)))?;
            let facts = prepared.function().launch_facts().unwrap();
            assert_eq!(facts.exact_block, Some(($threads, 1, 1)));
            assert_eq!(facts.dynamic_shared_bytes, $bytes);
            assert_eq!(facts.dynamic_shared_alignment, $alignment);
            assert_eq!(prepared.__raw_config().shared_mem_bytes, $bytes);
            // Equal thread products do not imply the required exact shape.
            for shape in [
                ($threads / 2, 1, 1),
                ($threads / 2, 2, 1),
                ($threads * 2, 1, 1),
            ] {
                assert!(matches!(
                    module.$prepare(LaunchConfig3D::auto_shared((1, 1, 1), shape)),
                    Err(LaunchContractError::BlockShapeMismatch { .. })
                ));
            }
            let required: u32 = $bytes;
            for bytes in [required.saturating_sub(1), required + 1] {
                if bytes != required {
                    assert!(matches!(
                        module.$prepare(LaunchConfig3D::new((1, 1, 1), ($threads, 1, 1), bytes)),
                        Err(LaunchContractError::DynamicSharedMemoryExactMismatch { .. })
                    ));
                }
            }
        }};
    }
    check_plan!(prepare_direct_1, 0, 1, 128);
    check_plan!(prepare_direct_4, 0, 1, 128);
    check_plan!(prepare_transpose_1, 528, 16, 128);
    check_plan!(prepare_transpose_4, 2064, 16, 128);
    check_plan!(prepare_shared_helper_root, 2064, 16, 128);
    #[cfg(feature = "root-context")]
    {
        check_plan!(prepare_smaller_1, 272, 16, 64);
        check_plan!(prepare_smaller_4, 1040, 16, 64);
    }
    let mut cases = 0;
    for (kernel, items, transpose, threads) in [
        (0, 1, false, 128),
        (1, 4, false, 128),
        (2, 1, true, 128),
        (3, 4, true, 128),
        (4, 4, true, 128),
        #[cfg(feature = "root-context")]
        (5, 1, true, 64),
        #[cfg(feature = "root-context")]
        (6, 4, true, 64),
    ] {
        let tile = threads * items;
        for blocks in [1, 3] {
            let extent = tile * blocks;
            let input: Vec<i32> = (0..extent).map(|i| i as i32 * 3 - 11).collect();
            let device_input = DeviceBuffer::from_host(&stream, &input)?;
            for valid in [0, 1, tile - 1, tile, extent - 1, extent] {
                for mode in 0..8_u32 {
                    let mode = mode | trap_mode;
                    let output = DeviceBuffer::from_host(&stream, &vec![-99_i32; extent])?;
                    let post = DeviceBuffer::from_host(&stream, &vec![-99_i32; extent])?;
                    let config =
                        LaunchConfig3D::auto_shared((blocks as u32, 1, 1), (threads as u32, 1, 1));
                    let input_ptr = device_input.cu_deviceptr() as *const i32;
                    let output_ptr = output.cu_deviceptr() as *mut i32;
                    let post_ptr = post.cu_deviceptr() as *mut i32;
                    unsafe {
                        match kernel {
                            0 => module.direct_1(
                                &stream,
                                &module.prepare_direct_1(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            1 => module.direct_4(
                                &stream,
                                &module.prepare_direct_4(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            2 => module.transpose_1(
                                &stream,
                                &module.prepare_transpose_1(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            3 => module.transpose_4(
                                &stream,
                                &module.prepare_transpose_4(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            4 => module.shared_helper_root(
                                &stream,
                                &module.prepare_shared_helper_root(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            #[cfg(feature = "root-context")]
                            5 => module.smaller_1(
                                &stream,
                                &module.prepare_smaller_1(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            #[cfg(feature = "root-context")]
                            6 => module.smaller_4(
                                &stream,
                                &module.prepare_smaller_4(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
                            _ => unreachable!(),
                        }
                    }
                    let loaded: Vec<i32> = input
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            (if mode & 1 != 0 && i >= valid {
                                -7
                            } else {
                                *value
                            }) + 1
                        })
                        .collect();
                    let expected_output: Vec<i32> = loaded
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            if mode & 2 != 0 && i >= valid {
                                -99
                            } else {
                                *value
                            }
                        })
                        .collect();
                    let expected_post: Vec<i32> = (0..extent)
                        .map(|index| {
                            if mode & 4 != 0 {
                                return expected_output[index];
                            }
                            let origin = index / tile * tile;
                            let lane = (index - origin) / items;
                            let item = index % items;
                            loaded[if transpose {
                                origin + item * threads + lane
                            } else {
                                index
                            }]
                        })
                        .collect();
                    assert_eq!(
                        output.to_host_vec(&stream)?,
                        expected_output,
                        "output kernel={kernel} blocks={blocks} valid={valid} mode={mode}"
                    );
                    assert_eq!(
                        post.to_host_vec(&stream)?,
                        expected_post,
                        "post-store kernel={kernel} blocks={blocks} valid={valid} mode={mode}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(
        cases,
        if cfg!(feature = "root-context") {
            672
        } else {
            480
        }
    );
    println!(
        "PASS {cases} native marker GPU cases with inferred scratch through shared generic helpers; {} CUBIN bytes",
        cubin.len()
    );
    Ok(())
}

#[cfg(feature = "storage")]
mod storage;
#[cfg(feature = "storage")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    storage::run()
}
