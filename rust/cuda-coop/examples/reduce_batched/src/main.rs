// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn batched_sum<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<
            i32,
            coop::CollectiveTiles<{ coop::batched_extent(ITEMS_PER_THREAD, 8) }>,
        >,
    ) where
        [(); coop::batched_extent(ITEMS_PER_THREAD, 8)]:,
    {
        let block = coop::this_block();
        let warp = coop::this_warp();
        let group = warp.group_by::<8>();
        let block_rank = block.rank_at::<coop::Grid>() as usize;
        let threads = block.count() as usize;
        let input_offset = block_rank * threads * ITEMS_PER_THREAD;
        let output_offset = block_rank * threads * coop::batched_extent(ITEMS_PER_THREAD, 8);
        let subgroup =
            warp.rank_at::<coop::Block>() as usize * 4 + group.rank_at::<coop::Warp>() as usize;
        let begin = input_offset + subgroup * 8 * ITEMS_PER_THREAD;
        let remaining = if begin <= input.len() {
            input.len() - begin
        } else {
            0
        };
        coop_reduce_batched_helper::batches::<_, ITEMS_PER_THREAD>(
            group,
            input,
            &mut output,
            input_offset,
            output_offset,
            remaining.min(8 * ITEMS_PER_THREAD),
            0_i32,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            const THREADS: usize = 64;
            const BLOCKS: usize = 3;
            const WIDTH: usize = 8;
            const K: usize = coop::batched_extent($items, WIDTH);
            let input: Vec<i32> = (0..BLOCKS * THREADS)
                .flat_map(|thread| {
                    (0..$items).map(move |batch| {
                        ((thread / THREADS + 1) * 1000 + (thread % THREADS) * 101 + batch * 7)
                            as i32
                    })
                })
                .take(BLOCKS * THREADS * $items - 7)
                .collect();
            let mut expected = vec![-1_i32; BLOCKS * THREADS * K];
            for group in 0..BLOCKS * THREADS / WIDTH {
                for batch in 0..$items {
                    expected[group * WIDTH * K + batch] = (0..WIDTH)
                        .map(|lane| {
                            input
                                .get((group * WIDTH + lane) * $items + batch)
                                .copied()
                                .unwrap_or(0)
                        })
                        .sum();
                }
            }
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; expected.len()])?;
            let plan = module.prepare_batched_sum::<$items>(LaunchConfig1D::auto_shared(
                BLOCKS as u32,
                THREADS as u32,
            ))?;
            let (_, target) = module
                .batched_sum_async_owned::<$items, _, _>(&plan, source, target)
                .sync()?;
            let output = target.to_host_vec(&stream)?;
            assert_eq!(output, expected);
            println!(
                "items/thread={}: {} batch sums and {} padding slots checked",
                $items,
                BLOCKS * THREADS / WIDTH * $items,
                expected.len() - BLOCKS * THREADS / WIDTH * $items
            );
        }};
    }
    run!(1);
    run!(4);
    run!(10);
    Ok(())
}
