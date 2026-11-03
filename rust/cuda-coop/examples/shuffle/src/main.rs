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
    pub fn shuffle_tile<const ITEMS_PER_THREAD: usize, const UP: bool>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        if UP {
            coop_shuffle_helper::shuffle_tile::<_, _, ITEMS_PER_THREAD>(
                block,
                input,
                &mut output,
                offset,
                0_i32,
                coop::ShuffleOptions::default()
                    .mode(coop::Up)
                    .distance::<1>(),
            );
        } else {
            coop_shuffle_helper::shuffle_tile::<_, _, ITEMS_PER_THREAD>(
                block,
                input,
                &mut output,
                offset,
                0_i32,
                Default::default(),
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
        ($items:literal, $up:literal) => {{
            let tile = 128 * $items;
            let input: Vec<i32> = (1..=3 * tile).map(|i| i as i32).collect();
            let expected: Vec<i32> = input
                .chunks_exact(tile)
                .flat_map(|values| {
                    if $up {
                        std::iter::once(0)
                            .chain(values[..tile - 1].iter().copied())
                            .collect::<Vec<_>>()
                    } else {
                        values[1..]
                            .iter()
                            .copied()
                            .chain(std::iter::once(0))
                            .collect::<Vec<_>>()
                    }
                })
                .collect();
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; input.len()])?;
            let plan =
                module.prepare_shuffle_tile::<$items, $up>(LaunchConfig1D::auto_shared(3, 128))?;
            let (_, target) = module
                .shuffle_tile_async_owned::<$items, $up, _, _>(&plan, source, target)
                .sync()?;
            let mut output = vec![0_i32; input.len()];
            target.copy_to_host(&stream, &mut output)?;
            stream.synchronize()?;
            assert_eq!(output, expected);
            println!(
                "items/thread={}, up={}: {} values checked",
                $items,
                $up,
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
