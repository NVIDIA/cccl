// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

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
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
