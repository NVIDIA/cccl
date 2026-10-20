// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_device::{kernel, launch_bounds, launch_contract};
use cuda_host::cuda_module;

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
            coop_marker_fixture_helper::roundtrip::<4, true>(input, output, post, valid, mode);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_core::{CudaContext, DeviceBuffer};
    use cuda_core::{LaunchConfig3D, LaunchContractError};
    use oxide_artifacts::{ArtifactPayloadKind, read_artifact_bundles_from_object_bytes};
    let executable = std::fs::read(std::env::current_exe()?)?;
    let bundles = read_artifact_bundles_from_object_bytes(&executable)?;
    let bundle = bundles
        .iter()
        .find(|b| b.payload(ArtifactPayloadKind::Cubin).is_some())
        .ok_or("missing embedded native artifact")?;
    let cubin = bundle.payload(ArtifactPayloadKind::Cubin).unwrap();
    std::fs::write("fixture.cubin", cubin)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the embedded artifact is generated from these kernel declarations.
    let module = unsafe { kernels::load(&context)? };
    macro_rules! check_plan {
        ($prepare:ident, $bytes:expr, $alignment:expr) => {{
            let prepared = module.$prepare(LaunchConfig3D::auto_shared((1, 1, 1), (128, 1, 1)))?;
            let facts = prepared.function().launch_facts().unwrap();
            assert_eq!(facts.exact_block, Some((128, 1, 1)));
            assert_eq!(facts.dynamic_shared_bytes, $bytes);
            assert_eq!(facts.dynamic_shared_alignment, $alignment);
            assert_eq!(prepared.__raw_config().shared_mem_bytes, $bytes);
            // Equal thread products do not imply the required exact shape.
            for shape in [(64, 2, 1), (64, 1, 1)] {
                assert!(matches!(
                    module.$prepare(LaunchConfig3D::auto_shared((1, 1, 1), shape)),
                    Err(LaunchContractError::BlockShapeMismatch { .. })
                ));
            }
            let required: u32 = $bytes;
            for bytes in [required.saturating_sub(1), required + 1] {
                if bytes != required {
                    assert!(matches!(
                        module.$prepare(LaunchConfig3D::new((1, 1, 1), (128, 1, 1), bytes)),
                        Err(LaunchContractError::DynamicSharedMemoryExactMismatch { .. })
                    ));
                }
            }
        }};
    }
    check_plan!(prepare_direct_1, 0, 1);
    check_plan!(prepare_direct_4, 0, 1);
    check_plan!(prepare_transpose_1, 528, 16);
    check_plan!(prepare_transpose_4, 2064, 16);
    check_plan!(prepare_shared_helper_root, 2064, 16);
    let mut cases = 0;
    for (kernel, items, transpose) in [
        (0, 1, false),
        (1, 4, false),
        (2, 1, true),
        (3, 4, true),
        (4, 4, true),
    ] {
        let tile = 128 * items;
        for blocks in [1, 3] {
            let extent = tile * blocks;
            let input: Vec<i32> = (0..extent).map(|i| i as i32 * 3 - 11).collect();
            let device_input = DeviceBuffer::from_host(&stream, &input)?;
            for valid in [0, 1, tile - 1, tile, extent - 1, extent] {
                for mode in 0..8_u32 {
                    let output = DeviceBuffer::from_host(&stream, &vec![-99_i32; extent])?;
                    let post = DeviceBuffer::from_host(&stream, &vec![-99_i32; extent])?;
                    let config = LaunchConfig3D::auto_shared((blocks as u32, 1, 1), (128, 1, 1));
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
                            _ => module.shared_helper_root(
                                &stream,
                                &module.prepare_shared_helper_root(config)?,
                                input_ptr,
                                output_ptr,
                                post_ptr,
                                valid as i32,
                                mode,
                            )?,
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
                                origin + item * 128 + lane
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
    assert_eq!(cases, 480);
    println!(
        "PASS {cases} native marker GPU cases with inferred scratch through five roots and sixteen provider specializations; {} CUBIN bytes",
        cubin.len()
    );
    Ok(())
}
