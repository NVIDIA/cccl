// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Logical Warp Direct through inferred public calls and a generic helper.
use cuda_coop as coop;
use cuda_device::{cooperative_groups as cg, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;
    use cg::ThreadGroup as _;

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exercise<G: coop::LoadStoreGroup<coop::Direct, ()>, const I: usize, const N: u32>(
        group: coop::ThreadGroup<G>,
        tile: cg::WarpTile<N>,
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
                group,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
        } else if mode == 2 {
            coop::load(
                group,
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
                group,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
            coop::load(
                group,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(count),
            );
        }
        let before = values;
        coop::store(
            group,
            &mut *output,
            &values,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(count),
        );
        let after = values;
        tile.sync();
        coop::load(
            group,
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
                    input[offset + q * width * I + lane * I + i]
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

    #[inline(never)]
    fn selected<const W: usize, const I: usize, const N: u32>(
        input: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<coop::Direct, ()>,
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
            cg::this_thread_block().tiled_partition::<N>(),
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
    pub fn logical_direct<const W: usize, const I: usize, const N: u32>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<I>>,
        valid: usize,
        mode: u32,
    ) where
        coop::ThreadsWithinWarp<W>: coop::LoadStoreGroup<coop::Direct, ()>,
    {
        selected::<W, I, N>(input, &mut output, valid, mode);
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,2),dynamic_shared=inferred)]
    pub fn logical_xyz(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        selected::<8, 4, 8>(input, &mut output, valid, mode);
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_nonexhaustive(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        let group = coop::this_warp().group_by_non_exhaustive::<8>();
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * 4
            + group.rank_at::<coop::Warp>() as usize;
        exercise(
            group,
            cg::this_thread_block().tiled_partition::<8>(),
            input,
            &mut output,
            8,
            q,
            group.rank() as usize,
            thread::blockIdx_x() as usize * 256,
            valid,
            mode,
        );
    }
    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_grain(
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        adapt: bool,
    ) {
        let group = coop::this_warp().group_by::<8>();
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * 4
            + group.rank_at::<coop::Warp>() as usize;
        let offset = thread::blockIdx_x() as usize * 256 + if adapt { q * 24 } else { 0 };
        coop::store(
            group,
            &mut output,
            7_i32,
            coop::StoreOptions::default().offset(offset),
        );
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_cycle<const COMPLETE: bool>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        let group = coop::this_warp().group_by::<8>();
        let q = coop::this_block().rank_at::<coop::Warp>() as usize * 4
            + group.rank_at::<coop::Warp>() as usize;
        let block = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let base = block * 9 * 256;
        let mut values = coop::ThreadData::<_, 4>::new();
        let mut sums = [0_i32; 4];
        let mut matches = true;
        let mut iteration = 0;
        // Siblings execute zero through seven iterations over immutable input.
        while iteration < q {
            let offset = base + iteration * 256;
            coop::load(
                group,
                input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
            let mut i = 0;
            while i < 4 {
                matches &= values.get(i) == Some(input[offset + lane * 4 + i]);
                sums[i] += values.get(i).unwrap_or(0);
                i += 1;
            }
            iteration += 1;
        }
        if COMPLETE {
            cg::this_thread_block().tiled_partition::<32>().sync();
        }
        coop::load(
            coop::this_warp(),
            input,
            &mut values,
            coop::LoadOptions::default().offset(base + 7 * 256),
        );
        let mut i = 0;
        while i < 4 {
            matches &= values.get(i) == Some(input[base + 7 * 256 + lane * 4 + i]);
            sums[i] += 3 * values.get(i).unwrap_or(0);
            i += 1;
        }
        if COMPLETE {
            thread::sync_threads();
        }
        coop::load(
            coop::this_block(),
            input,
            &mut values,
            coop::LoadOptions::default().offset(base + 8 * 256),
        );
        let mut i = 0;
        while i < 4 {
            matches &= values.get(i) == Some(input[base + 8 * 256 + lane * 4 + i]);
            values.write(i, sums[i] + 5 * values.get(i).unwrap_or(0));
            i += 1;
        }
        coop::store(
            coop::this_block(),
            &mut output,
            &values,
            coop::StoreOptions::default().offset(block * 256),
        );
        assert!(matches);
    }

    #[kernel]
    #[launch_contract(domain=1,block=(64,1,1),dynamic_shared=inferred)]
    pub fn logical_mixed_block(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
    ) {
        let block = coop::this_block();
        let group = coop::this_warp().group_by::<8>();
        let offset = thread::blockIdx_x() as usize * 256;
        let lane = thread::threadIdx_x() as usize;
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .offset(offset)
                .algorithm(coop::Transpose),
        );
        coop::store(
            group,
            &mut output,
            &values,
            coop::StoreOptions::default().offset(offset),
        );
        cg::this_thread_block().tiled_partition::<8>().sync();
        coop::load(
            group,
            &mut output,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let mut matches = (values.get(0) == Some(input[offset + lane * 4]))
            & (values.get(1) == Some(input[offset + lane * 4 + 1]))
            & (values.get(2) == Some(input[offset + lane * 4 + 2]))
            & (values.get(3) == Some(input[offset + lane * 4 + 3]));
        // Tile completion leaves the Block scratch dirty. Managed reuse must
        // insert a full-block barrier before this second exchange.
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .offset(offset)
                .algorithm(coop::Transpose),
        );
        for i in 0..4 {
            matches &= values.get(i) == Some(input[offset + lane * 4 + i]);
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
    // SAFETY: this module and its typed launchers compile together.
    let module = unsafe { kernels::load_async(0)? };
    let mut cases = 0;
    macro_rules! exercise {
        ($width:literal,$items:literal,$prepare:expr,$launch:expr,$config:expr) => {{
            let groups = 3;
            let config = ($config)(groups);
            let plan = ($prepare)(config)?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
            let input: Vec<i32> = (0..groups as usize * 64 * $items)
                .map(|i| i as i32 + 31)
                .collect();
            for mode in 0..5 {
                for valid in [0, 1, $width * $items - 1, $width * $items] {
                    let source = DeviceBuffer::from_host(&stream, &input)?;
                    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
                    let (_, target) = ($launch)(&plan, source, target, valid, mode).sync()?;
                    let mut result = vec![0; input.len()];
                    target.copy_to_host(&stream, &mut result)?;
                    stream.synchronize()?;
                    let mut expected = vec![-99; input.len()];
                    for block in 0..groups as usize {
                        for q in 0..64 / $width {
                            let count = if mode == 0 {
                                $width * $items
                            } else if mode == 3 {
                                (valid + q).min($width * $items)
                            } else {
                                valid
                            };
                            for i in 0..count {
                                let index = block * 64 * $items + q * $width * $items + i;
                                expected[index] = if mode == 4 { -7 } else { input[index] };
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
        |c| module.prepare_logical_direct::<1, 1, 1>(c),
        |p, s, t, v, m| module.logical_direct_async_owned::<1, 1, 1, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        8,
        4,
        |c| module.prepare_logical_direct::<8, 4, 8>(c),
        |p, s, t, v, m| module.logical_direct_async_owned::<8, 4, 8, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        16,
        4,
        |c| module.prepare_logical_direct::<16, 4, 16>(c),
        |p, s, t, v, m| module.logical_direct_async_owned::<16, 4, 16, _, _>(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    exercise!(
        8,
        4,
        |c| module.prepare_logical_xyz(c),
        |p, s, t, v, m| module.logical_xyz_async_owned(p, s, t, v, m),
        |g| cuda_core::LaunchConfig3D::auto_shared((1, g, 1), (8, 4, 2))
    );
    exercise!(
        8,
        4,
        |c| module.prepare_logical_nonexhaustive(c),
        |p, s, t, v, m| module.logical_nonexhaustive_async_owned(p, s, t, v, m),
        |g| LaunchConfig1D::auto_shared(g, 64)
    );
    let grain_plan = module.prepare_logical_grain(LaunchConfig1D::auto_shared(2, 64))?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; 512])?;
    let adapt = !std::env::args().any(|arg| arg == "logical-grain-trap");
    let target = module
        .logical_grain_async_owned(&grain_plan, target, adapt)
        .sync()?;
    assert!(adapt, "expected fixed-grain trap completed successfully");
    let mut result = vec![0; 512];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    let mut expected = vec![-99; 512];
    for block in 0..2 {
        for q in 0..8 {
            expected[block * 256 + q * 32..block * 256 + q * 32 + 8].fill(7);
        }
    }
    assert_eq!(result, expected);
    println!("PASS {cases} logical Warp mask/default/scattered cases and fixed-grain offset24");

    let plan = module.prepare_logical_mixed_block(LaunchConfig1D::auto_shared(3, 64))?;
    assert_eq!(plan.__raw_config().shared_mem_bytes, 1024);
    let input: Vec<i32> = (0..768).map(|i| i + 53).collect();
    let source = DeviceBuffer::from_host(&stream, &input)?;
    let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; input.len()])?;
    let (_, target) = module
        .logical_mixed_block_async_owned(&plan, source, target)
        .sync()?;
    let mut result = vec![0; input.len()];
    target.copy_to_host(&stream, &mut result)?;
    stream.synchronize()?;
    assert_eq!(result, input);
    println!("PASS mixed Block Transpose/logical Warp Direct: 768 values");
    run_cycles()?;
    Ok(())
}

fn run_cycles() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::with_cuda_context;
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let stream = with_cuda_context(0, |context| context.default_stream())?;
    // SAFETY: this module and its typed launchers compile together.
    let module = unsafe { kernels::load_async(0)? };
    let input: Vec<i32> = (0..3 * 9 * 256).map(|i| i + 17).collect();
    let expected: Vec<i32> = (0..3 * 256)
        .map(|index| {
            let base = index / 256 * 9 * 256;
            let element = index % 256;
            let q = element / 4 / 8;
            (0..q)
                .map(|iteration| input[base + iteration * 256 + element])
                .sum::<i32>()
                + 3 * input[base + 7 * 256 + element]
                + 5 * input[base + 8 * 256 + element]
        })
        .collect();
    macro_rules! exercise {
        ($complete:literal) => {{
            let plan =
                module.prepare_logical_cycle::<$complete>(LaunchConfig1D::auto_shared(3, 64))?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, 0);
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-99_i32; expected.len()])?;
            let (_, target) = module
                .logical_cycle_async_owned::<$complete, _, _>(&plan, source, target)
                .sync()?;
            let mut result = vec![0; expected.len()];
            target.copy_to_host(&stream, &mut result)?;
            stream.synchronize()?;
            assert_eq!(result, expected);
        }};
    }
    exercise!(false);
    exercise!(true);
    println!("PASS finite sibling cycles then W32/Block: 1536 values, without/with barriers");
    Ok(())
}
