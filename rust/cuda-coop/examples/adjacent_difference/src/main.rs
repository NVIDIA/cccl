// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn difference_tile<const ITEMS_PER_THREAD: usize, const RIGHT: bool>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        if RIGHT {
            coop_adjacent_difference_helper::difference_tile::<_, _, ITEMS_PER_THREAD>(
                block,
                input,
                &mut output,
                offset,
                0_i32,
                coop::Right,
            );
        } else {
            coop_adjacent_difference_helper::difference_tile::<_, _, ITEMS_PER_THREAD>(
                block,
                input,
                &mut output,
                offset,
                0_i32,
                coop::Left,
            );
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
        ($items:literal, $right:literal) => {{
            let tile = 128 * $items;
            let input: Vec<i32> = (0..3 * tile - 7)
                .map(|i| (i * 17 % 97) as i32 - 48)
                .collect();
            let expected: Vec<i32> = input
                .chunks(tile)
                .flat_map(|values| {
                    values.iter().enumerate().map(|(i, &value)| {
                        if $right {
                            if i + 1 < values.len() {
                                value - values[i + 1]
                            } else {
                                value
                            }
                        } else if i > 0 {
                            value - values[i - 1]
                        } else {
                            value
                        }
                    })
                })
                .collect();
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; input.len()])?;
            let plan = module
                .prepare_difference_tile::<$items, $right>(LaunchConfig1D::auto_shared(3, 128))?;
            let (_, target) = module
                .difference_tile_async_owned::<$items, $right, _, _>(&plan, source, target)
                .sync()?;
            let mut output = vec![0_i32; input.len()];
            target.copy_to_host(&stream, &mut output)?;
            stream.synchronize()?;
            assert_eq!(output, expected);
            println!(
                "items/thread={}, right={}: {} differences checked",
                $items,
                $right,
                output.len()
            );
        }};
    }
    run!(1, false);
    run!(4, false);
    run!(1, true);
    run!(4, true);
    Ok(())
}
