// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Replicated implicit Warp scratch and observed Transpose payload mutation.
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exercise<G: coop::LoadStoreGroup<coop::Transpose, ()>, const I: usize>(
        group: coop::ThreadGroup<G>,
        input: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        width: usize,
        q: usize,
        lane: usize,
        offset: usize,
        valid: usize,
        mode: u32,
    ) {
        let count = if mode == 0 {
            width * I
        } else if mode == 3 {
            (valid + q).min(width * I)
        } else {
            valid
        };
        let mut values = coop::ThreadData::<_, I>::new();
        if mode == 4 {
            let mut i = 0;
            while i < I {
                if lane * I + i < count || (lane + i) % 3 == 1 {
                    values.write(i, (lane * I + i) as i32 + 17);
                }
                i += 1;
            }
        } else {
            coop::load(
                group,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .algorithm(coop::Transpose)
                    .offset(offset),
            );
            if mode != 0 {
                let options = coop::LoadOptions::default()
                    .algorithm(coop::Transpose)
                    .offset(offset)
                    .valid_items(count);
                if mode == 2 {
                    coop::load(group, input, &mut values, options.oob_default(-7));
                } else {
                    coop::load(group, input, &mut values, options);
                }
            }
        }
        let before = values;
        coop::store(
            group,
            &mut *output,
            &mut values,
            coop::StoreOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset)
                .valid_items(count),
        );
        let after = values;
        coop::store(
            group,
            &mut *output,
            &mut values,
            coop::StoreOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset)
                .valid_items(0),
        );
        let twice = values;
        // Managed scratch reuse supplies matching completion before this read.
        coop::load(
            group,
            output,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset)
                .valid_items(count),
        );
        let mut matches = true;
        for i in 0..I {
            let first = width * i + lane;
            let second = width * (first % I) + first / I;
            let mut observation = 0;
            while observation < 3 {
                let (snapshot, index) = match observation {
                    0 => (before.get(i), lane * I + i),
                    1 => (after.get(i), first),
                    _ => (twice.get(i), second),
                };
                let initialized = if mode == 4 {
                    index < count || (index / I + index % I) % 3 == 1
                } else {
                    mode == 0 || mode == 2 || index < count
                };
                let expected = if initialized {
                    Some(if mode == 4 {
                        index as i32 + 17
                    } else if mode == 2 && index >= count {
                        -7
                    } else {
                        input[offset + q * width * I + index]
                    })
                } else {
                    None
                };
                matches &= snapshot == expected;
                observation += 1;
            }
            let index = lane * I + i;
            let expected = if index < count {
                Some(if mode == 4 {
                    index as i32 + 17
                } else {
                    input[offset + q * width * I + index]
                })
            } else {
                None
            };
            matches &= values.get(i) == expected;
        }
        assert!(matches);
    }

    #[inline(never)]
    fn selected<const W: usize, const I: usize>(
        input: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<coop::Transpose, ()>,
    {
        let group = coop::this_warp().group_by::<W>();
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * (32 / W)
            + group.rank_at::<coop::Warp>() as usize;
        let block = thread::blockIdx_x() as usize
            + thread::gridDim_x() as usize
                * (thread::blockIdx_y() as usize
                    + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
        exercise(
            group,
            input,
            output,
            W,
            q,
            group.rank() as usize,
            block * 64 * I,
            valid,
            mode,
        );
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_transpose<const W: usize, const I: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<coop::Transpose, ()>,
    {
        selected::<W, I>(input, &mut output, valid, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(128,1,1),dynamic_shared=inferred)]
    pub fn logical_transpose_b128(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        let group = coop::this_warp().group_by::<8>();
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * 4
            + group.rank_at::<coop::Warp>() as usize;
        exercise(
            group,
            input,
            &mut output,
            8,
            q,
            group.rank() as usize,
            thread::blockIdx_x() as usize * 512,
            17,
            4,
        );
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,2),dynamic_shared=inferred)]
    pub fn logical_transpose_xyz(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        selected::<8, 4>(input, &mut output, valid, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_transpose_prefix(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        // SAFETY: one whole fixed prefix, disjoint lane writes and Block completion.
        let lane = thread::threadIdx_x() as usize;
        let prefix =
            unsafe { cuda_device::DynamicSharedArray::<[i32; 64], 32>::get_fixed() as *mut i32 };
        unsafe {
            prefix.add(lane).write(lane as i32);
        }
        thread::sync_threads();
        selected::<8, 4>(input, &mut output, valid, mode);
        let neighbor = (lane + 1) % 64;
        assert!(unsafe { prefix.add(neighbor).read() } == neighbor as i32);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_transpose_containing(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        use cuda_device::cooperative_groups::{ThreadGroup, this_thread_block};
        let warp8 = coop::this_warp().group_by::<8>();
        let warp16 = coop::this_warp().group_by::<16>();
        let tile16 = this_thread_block().tiled_partition::<16>();
        let offset = thread::blockIdx_x() as usize * 256;
        let lane = thread::threadIdx_x() as usize;
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(
            warp8,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        coop::store(
            warp8,
            &mut output,
            &mut values,
            coop::StoreOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset)
                .valid_items(32),
        );
        tile16.sync();
        coop::load(
            warp16,
            &mut output,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let first = values;
        coop::load(
            warp16,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let second = values;
        tile16.sync();
        coop::load(
            warp8,
            &mut output,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let mut matches = true;
        for i in 0..4 {
            let expected = Some(input[offset + lane * 4 + i]);
            matches &=
                first.get(i) == expected && second.get(i) == expected && values.get(i) == expected;
        }
        assert!(matches);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_transpose_mixed(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        let block = coop::this_block();
        let warp = coop::this_warp().group_by::<8>();
        let offset = thread::blockIdx_x() as usize * 256;
        let lane = thread::threadIdx_x() as usize;
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let first = values;
        coop::load(
            warp,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let second = values;
        coop::load(
            warp,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        let third = values;
        // W8 reuse must leave Block scratch dirty, requiring a Block barrier here.
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .algorithm(coop::Transpose)
                .offset(offset),
        );
        coop::store(
            block,
            &mut output,
            &values,
            coop::StoreOptions::default().offset(offset),
        );
        let mut matches = true;
        for i in 0..4 {
            let expected = Some(input[offset + lane * 4 + i]);
            matches &= first.get(i) == expected
                && second.get(i) == expected
                && third.get(i) == expected
                && values.get(i) == expected;
        }
        assert!(matches);
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    init_device_contexts(0, 1)?;
    let stream = with_cuda_context(0, |context| context.default_stream())?;
    // SAFETY: the typed module and its launchers compile together.
    let module = unsafe { kernels::load_async(0)? };
    let mut cases = 0;
    macro_rules! exercise {
        ($w:literal,$i:literal,$scratch:literal,$prepare:expr,$launch:expr,$config:expr) => {{
            let plan = ($prepare)(($config)(3))?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, $scratch);
            let input: Vec<i32> = (0..3 * 64 * $i).map(|i| i as i32 + 31).collect();
            for mode in 0..5 {
                for valid in [0, 1, $w * $i - 1, $w * $i] {
                    let source = DeviceBuffer::from_host(&stream, &input)?;
                    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
                    let (_, target) = ($launch)(&plan, source, target, valid, mode).sync()?;
                    let mut result = vec![0; input.len()];
                    target.copy_to_host(&stream, &mut result)?;
                    stream.synchronize()?;
                    let mut expected = vec![-99; input.len()];
                    for block in 0..3 {
                        for q in 0..64 / $w {
                            let count = if mode == 0 {
                                $w * $i
                            } else if mode == 3 {
                                (valid + q).min($w * $i)
                            } else {
                                valid
                            };
                            for i in 0..count {
                                let index = block * 64 * $i + q * $w * $i + i;
                                expected[index] = if mode == 4 {
                                    i as i32 + 17
                                } else {
                                    input[index]
                                };
                            }
                        }
                    }
                    assert_eq!(result, expected);
                    cases += 1;
                }
            }
        }};
    }
    exercise!(
        1,
        1,
        512,
        |c| module.prepare_logical_transpose::<1, 1>(c),
        |p, s, t, v, m| module.logical_transpose_async_owned::<1, 1, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        8,
        4,
        1184,
        |c| module.prepare_logical_transpose::<8, 4>(c),
        |p, s, t, v, m| module.logical_transpose_async_owned::<8, 4, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        32,
        4,
        1064,
        |c| module.prepare_logical_transpose::<32, 4>(c),
        |p, s, t, v, m| module.logical_transpose_async_owned::<32, 4, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        8,
        4,
        1184,
        |c| module.prepare_logical_transpose_xyz(c),
        |p, s, t, v, m| module.logical_transpose_xyz_async_owned(p, s, t, v, m),
        |g| cuda_core::LaunchConfig3D::auto_shared((1, g, 1), (8, 4, 2))
    );
    exercise!(
        8,
        4,
        1440,
        |c| module.prepare_logical_transpose_prefix(c),
        |p, s, t, v, m| module.logical_transpose_prefix_async_owned(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    assert!(
        module
            .prepare_logical_transpose::<8, 4>(LaunchConfig1D::new(3, 64, 1183))
            .is_err()
    );
    let plan = module.prepare_logical_transpose_mixed(LaunchConfig1D::auto_shared(3, 64))?;
    assert_eq!(plan.__raw_config().shared_mem_bytes, 2080);
    let input: Vec<i32> = (0..768).map(|i| i + 53).collect();
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; 768])?;
    let (_, target) = module
        .logical_transpose_mixed_async_owned(&plan, source, target)
        .sync()?;
    let mut result = vec![0; 768];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert_eq!(result, input);
    let plan = module.prepare_logical_transpose_containing(LaunchConfig1D::auto_shared(3, 64))?;
    assert_eq!(plan.__raw_config().shared_mem_bytes, 2224);
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; 768])?;
    let (_, target) = module
        .logical_transpose_containing_async_owned(&plan, source, target)
        .sync()?;
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert_eq!(result, input);
    println!(
        "PASS {cases} logical Transpose cases plus mixed Block/W8 and W8/W16 pool reuse, containing synchronization and undersized launch rejection"
    );
    run_b128()?;
    Ok(())
}

/// One changed-replica-count contrast using the same value/validity helper.
fn run_b128() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::with_cuda_context;
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let stream = with_cuda_context(0, |context| context.default_stream())?;
    // SAFETY: the typed module and its launchers compile together.
    let module = unsafe { kernels::load_async(0)? };
    let plan = module.prepare_logical_transpose_b128(LaunchConfig1D::auto_shared(3, 128))?;
    assert_eq!(plan.__raw_config().shared_mem_bytes, 148 * 16);
    let input: Vec<i32> = (0..1536).map(|i| i + 31).collect();
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
    let (_, target) = module
        .logical_transpose_b128_async_owned(&plan, source, target)
        .sync()?;
    let mut result = vec![0; input.len()];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    let mut expected = vec![-99; input.len()];
    for block in 0..3 {
        for q in 0..16 {
            for i in 0..17 {
                expected[block * 512 + q * 32 + i] = i as i32 + 17;
            }
        }
    }
    assert_eq!(result, expected);
    println!("PASS B128/W8 Transpose: 16 replicas, 2368 bytes, observed values/validity");
    Ok(())
}
