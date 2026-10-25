// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn wrapping_add(left: i32, right: i32) -> i32 {
        left.wrapping_add(right)
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn block_callable<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut sums: coop::DisjointSlice<i32, thread::BlockIndex>,
        mut callable_sums: coop::DisjointSlice<i32, thread::BlockIndex>,
        mut maxima: coop::DisjointSlice<i32, thread::BlockIndex>,
        mut biased_sums: coop::DisjointSlice<i32, thread::BlockIndex>,
        bias: i32,
    ) {
        let block = coop::this_block();
        let offset = block.rank_at::<coop::Grid>() as usize * 128 * ITEMS_PER_THREAD;
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let before = values;
        let mut scratch = coop::TempStorage::new().auto_sync();
        let sum = coop_reduce_helper::reduce_values(block, &values, coop::Sum, &mut scratch);
        let callable_sum =
            coop_reduce_helper::reduce_values(block, &values, wrapping_add, &mut scratch);
        let maximum = coop_reduce_helper::reduce_values(
            block,
            &values,
            |left, right| if left > right { left } else { right },
            &mut scratch,
        );
        let biased_sum = coop_reduce_helper::reduce_values(
            block,
            &values,
            move |left, right| wrapping_add(wrapping_add(left, right), bias),
            &mut scratch,
        );
        for index in 0..ITEMS_PER_THREAD {
            assert!(values.get(index) == before.get(index));
        }
        // All four collectives finish before any root-only output branch.
        if let (Some(sum), Some(callable_sum), Some(maximum), Some(biased_sum)) =
            (sum, callable_sum, maximum, biased_sum)
        {
            if let Some(index) = thread::block_index()
                && let Some(slot) = sums.get_mut(index)
            {
                *slot = sum;
            }
            if let Some(index) = thread::block_index()
                && let Some(slot) = callable_sums.get_mut(index)
            {
                *slot = callable_sum;
            }
            if let Some(index) = thread::block_index()
                && let Some(slot) = maxima.get_mut(index)
            {
                *slot = maximum;
            }
            if let Some(index) = thread::block_index()
                && let Some(slot) = biased_sums.get_mut(index)
            {
                *slot = biased_sum;
            }
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn block_sum<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, thread::BlockIndex>,
    ) {
        let block = coop::this_block();
        let tile = block.count() as usize * ITEMS_PER_THREAD;
        let offset = block.rank_at::<coop::Grid>() as usize * tile;
        let remaining = if offset <= input.len() {
            input.len() - offset
        } else {
            0
        };
        let total = coop_reduce_helper::sum_tile::<_, ITEMS_PER_THREAD>(
            block,
            input,
            offset,
            remaining.min(tile),
            0_i32,
        );
        // Every thread reduces before this terminal output branch. The witness
        // proves ownership of one block's output; get_mut checks its bounds.
        if let Some(total) = total
            && let Some(index) = thread::block_index()
            && let Some(slot) = output.get_mut(index)
        {
            *slot = total;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 128 * $items;
            let input: Vec<i32> = (0..3 * tile - 7).map(|i| (i % 7) as i32).collect();
            let expected: Vec<i32> = input.chunks(tile).map(|chunk| chunk.iter().sum()).collect();
            let plan = module.prepare_block_sum::<$items>(LaunchConfig1D::auto_shared(3, 128))?;
            for output_len in [3, 2, 0] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; output_len])?;
                let (_, target) = module
                    .block_sum_async_owned::<$items, _, _>(&plan, source, target)
                    .sync()?;
                let output = if target.is_empty() {
                    stream.synchronize()?;
                    Vec::new()
                } else {
                    target.to_host_vec(&stream)?
                };
                assert_eq!(output, expected[..output_len]);
                println!(
                    "items/thread={}: {output_len} block sums checked (3 blocks launched)",
                    $items
                );
            }

            let input: Vec<i32> = (0..3 * tile).map(|i| (i % 29) as i32 - 14).collect();
            let expected_sums: Vec<i32> = input.chunks(tile).map(|v| v.iter().sum()).collect();
            let expected_maxima: Vec<i32> = input
                .chunks(tile)
                .map(|v| *v.iter().max().expect("nonempty block tile"))
                .collect();
            let plan =
                module.prepare_block_callable::<$items>(LaunchConfig1D::auto_shared(3, 128))?;
            // The same compiled specialization receives two different capture values.
            for bias in [-3_i32, 7] {
                let expected_biased: Vec<i32> = expected_sums
                    .iter()
                    .map(|&sum| sum + (tile as i32 - 1) * bias)
                    .collect();
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let sums = DeviceBuffer::from_host(&stream, &[-1_i32; 3])?;
                let callable_sums = DeviceBuffer::from_host(&stream, &[-1_i32; 3])?;
                let maxima = DeviceBuffer::from_host(&stream, &[-1_i32; 3])?;
                let biased_sums = DeviceBuffer::from_host(&stream, &[-1_i32; 3])?;
                let (source, sums, callable_sums, maxima, biased_sums) = module
                    .block_callable_async_owned::<$items, _, _, _, _, _>(
                        &plan,
                        source,
                        sums,
                        callable_sums,
                        maxima,
                        biased_sums,
                        bias,
                    )
                    .sync()?;
                assert_eq!(source.to_host_vec(&stream)?, input);
                assert_eq!(sums.to_host_vec(&stream)?, expected_sums);
                assert_eq!(callable_sums.to_host_vec(&stream)?, expected_sums);
                assert_eq!(maxima.to_host_vec(&stream)?, expected_maxima);
                assert_eq!(biased_sums.to_host_vec(&stream)?, expected_biased);
                println!(
                    "items/thread={}, bias={bias}: 12 built-in/callable results and {} preserved inputs checked",
                    $items,
                    input.len(),
                );
            }
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
