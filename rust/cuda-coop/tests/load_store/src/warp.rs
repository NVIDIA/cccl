// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Physical Warp Direct uses the ordinary public API and owned async launch.
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, library, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn exercise<const I: usize>(
        input: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) {
        let warp = coop::this_warp();
        let q = library::warp_group_index_v1::<32>() as usize;
        let block = thread::blockIdx_x() as usize;
        let offset = block * 64 * I;
        let lane = (thread::threadIdx_x() as usize
            + thread::blockDim_x() as usize
                * (thread::threadIdx_y() as usize
                    + thread::blockDim_y() as usize * thread::threadIdx_z() as usize))
            % 32;
        let count = if mode == 0 {
            32 * I
        } else if mode == 3 {
            (valid + q).min(32 * I)
        } else {
            valid
        };
        let mut values = coop::ThreadData::<_, I>::new();
        if mode == 4 {
            let mut other = coop::ThreadData::<_, I>::new();
            let mut i = 0;
            while i < I {
                if lane * I + i < count || i % 2 == 0 {
                    values.write(i, -7);
                }
                if lane * I + i < count || i % 2 != 0 {
                    other.write(i, -7);
                }
                i += 1;
            }
            values = if lane & 1 == 0 { values } else { other };
        } else if mode == 0 {
            coop::load(
                warp,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
        } else if mode == 2 {
            coop::load(
                warp,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(count)
                    .oob_default(-7),
            );
        } else {
            // A preceding full Load makes the exact masked reset observable.
            coop::load(
                warp,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
            coop::load(
                warp,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(count),
            );
        }
        let before = values;
        coop::store(
            warp,
            &mut *output,
            &values,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(count),
        );
        let after = values;
        thread::sync_threads();
        coop::load(
            warp,
            output,
            &mut values,
            coop::LoadOptions::default()
                .offset(offset)
                .valid_items(count),
        );
        let mut matches = true;
        let mut i = 0;
        while i < I {
            matches &= before.get(i) == after.get(i);
            if lane * I + i < count {
                let expected = if mode == 4 {
                    -7
                } else {
                    input[offset + q * 32 * I + lane * I + i]
                };
                matches &= values.get(i) == Some(expected);
            } else {
                matches &= values.get(i).is_none();
                if mode == 1 || mode == 3 {
                    matches &= before.get(i).is_none();
                }
                if mode == 2 {
                    matches &= before.get(i) == Some(-7);
                }
            }
            i += 1;
        }
        assert!(matches);
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn warp_direct_64<const I: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) {
        exercise::<I>(input, &mut output, valid, mode);
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,2),dynamic_shared=inferred)]
    pub fn warp_direct_xyz<const I: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) {
        exercise::<I>(input, &mut output, valid, mode);
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn warp_mixed_block_transpose(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        let offset = thread::blockIdx_x() as usize * 256;
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(
            coop::this_block(),
            input,
            &mut values,
            coop::LoadOptions::default()
                .offset(offset)
                .algorithm(coop::Transpose),
        );
        coop::store(
            coop::this_warp(),
            &mut output,
            &values,
            coop::StoreOptions::default().offset(offset),
        );
        thread::sync_threads();
        coop::load(
            coop::this_warp(),
            &mut output,
            &mut values,
            coop::LoadOptions::default().offset(offset).valid_items(127),
        );
        let lane = thread::threadIdx_x() as usize;
        let mut matches = true;
        let mut i = 0;
        while i < 4 {
            let expected = if (lane % 32) * 4 + i < 127 {
                Some(input[offset + lane * 4 + i])
            } else {
                None
            };
            matches &= values.get(i) == expected;
            i += 1;
        }
        assert!(matches);
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn warp_fixed_grain(
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        adapt: bool,
    ) {
        let q = library::warp_group_index_v1::<32>() as usize;
        let offset = thread::blockIdx_x() as usize * 256 + if adapt { q * 96 } else { 0 };
        // Automatic origin is q*32 for this one-item scalar. Carrier capacity
        // stays four, so warp one needs caller offset96 to reach its region128.
        coop::store(
            coop::this_warp(),
            &mut output,
            7_i32,
            coop::StoreOptions::default().offset(offset),
        );
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let stream = with_cuda_context(0, |context| context.default_stream())?;
    // SAFETY: the typed launcher and embedded native kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    if std::env::args().any(|argument| argument == "--warp-grain-trap") {
        let plan = module.prepare_warp_fixed_grain(LaunchConfig1D::auto_shared(1, 64))?;
        let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; 256])?;
        module
            .warp_fixed_grain_async_owned(&plan, target, false)
            .sync()?;
        panic!("expected ownership trap returned successfully");
    }
    for groups in [1_u32, 3] {
        let plan = module.prepare_warp_fixed_grain(LaunchConfig1D::auto_shared(groups, 64))?;
        let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; groups as usize * 256])?;
        let target = module
            .warp_fixed_grain_async_owned(&plan, target, true)
            .sync()?;
        let mut result = vec![0; groups as usize * 256];
        target.copy_to_host(&stream, &mut result)?;
        stream.synchronize()?;
        let mut expected = vec![-99_i32; result.len()];
        for block in 0..groups as usize {
            for warp in 0..2 {
                expected[block * 256 + warp * 128..block * 256 + warp * 128 + 32].fill(7);
            }
        }
        assert_eq!(result, expected);
    }
    // A storage-bearing Block operation and scratch-free Warp operations
    // share one root plan and the same collective allocation.
    let plan = module.prepare_warp_mixed_block_transpose(LaunchConfig1D::auto_shared(2, 64))?;
    assert!(plan.__raw_config().shared_mem_bytes > 0);
    let input: Vec<i32> = (0..512).map(|i| i + 17).collect();
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
    let (_, target) = module
        .warp_mixed_block_transpose_async_owned(&plan, source, target)
        .sync()?;
    let mut result = vec![0; input.len()];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert_eq!(result, input);
    let mut cases = 0;
    macro_rules! run {
        ($i:literal,$prepare:ident,$launch:ident,$config:expr,$wrong:expr) => {{
            assert!(module.$prepare::<$i>($wrong).is_err());
            for groups in [1_u32, 3] {
                let input: Vec<i32> = (0..groups as usize * 64 * $i)
                    .map(|i| i as i32 + 11)
                    .collect();
                let plan = module.$prepare::<$i>($config(groups))?;
                assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
                for mode in 0..5 {
                    for valid in [0, 1, 32 * $i - 1, 32 * $i] {
                        let source = DeviceBuffer::from_host(&stream, &input)?;
                        let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
                        let (_, target) = module
                            .$launch::<$i, _, _>(&plan, source, target, valid, mode)
                            .sync()?;
                        let mut result = vec![0; input.len()];
                        target.copy_to_host(&stream, &mut result)?;
                        stream.synchronize()?;
                        let mut expected = vec![-99; input.len()];
                        for block in 0..groups as usize {
                            for q in 0..2 {
                                let count = if mode == 0 {
                                    32 * $i
                                } else if mode == 3 {
                                    (valid + q).min(32 * $i)
                                } else {
                                    valid
                                };
                                for i in 0..count {
                                    let index = block * 64 * $i + q * 32 * $i + i;
                                    expected[index] = if mode == 4 { -7 } else { input[index] };
                                }
                            }
                        }
                        assert_eq!(result, expected);
                        cases += 1;
                    }
                }
            }
        }};
    }
    run!(
        1,
        prepare_warp_direct_64,
        warp_direct_64_async_owned,
        |g| LaunchConfig1D::auto_shared(g, 64),
        LaunchConfig1D::auto_shared(1, 32)
    );
    run!(
        4,
        prepare_warp_direct_64,
        warp_direct_64_async_owned,
        |g| LaunchConfig1D::auto_shared(g, 64),
        LaunchConfig1D::auto_shared(1, 32)
    );
    run!(
        1,
        prepare_warp_direct_xyz,
        warp_direct_xyz_async_owned,
        |g| LaunchConfig3D::auto_shared((g, 1, 1), (8, 4, 2)),
        LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 4))
    );
    run!(
        4,
        prepare_warp_direct_xyz,
        warp_direct_xyz_async_owned,
        |g| LaunchConfig3D::auto_shared((g, 1, 1), (8, 4, 2)),
        LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 4))
    );
    println!(
        "PASS {cases} physical Warp Direct cases and 2 fixed-grain cases; mixed Block Transpose/Warp Direct passed"
    );
    Ok(())
}
