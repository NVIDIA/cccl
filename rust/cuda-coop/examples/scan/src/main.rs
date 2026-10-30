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
    pub fn prefix<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let tile = block.count() as usize * ITEMS_PER_THREAD;
        let offset = block.rank_at::<coop::Grid>() as usize * tile;
        let remaining = if offset <= input.len() {
            input.len() - offset
        } else {
            0
        };
        coop_scan_helper::prefix::<_, ITEMS_PER_THREAD>(
            block,
            input,
            &mut output,
            offset,
            remaining.min(tile),
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
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 128 * $items;
            let input: Vec<i32> = (0..3 * tile - 7).map(|i| (i % 7) as i32).collect();
            let mut expected = Vec::with_capacity(input.len());
            for chunk in input.chunks(tile) {
                let mut prefix = 0;
                for &value in chunk {
                    expected.push(prefix);
                    prefix += value;
                }
            }
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-1_i32; input.len()])?;
            let plan = module.prepare_prefix::<$items>(LaunchConfig1D::auto_shared(3, 128))?;
            let (_, target) = module
                .prefix_async_owned::<$items, _, _>(&plan, source, target)
                .sync()?;
            let mut output = vec![0_i32; input.len()];
            target.copy_to_host(&stream, &mut output)?;
            stream.synchronize()?;
            assert_eq!(output, expected);
            println!("items/thread={}: {} prefixes checked", $items, output.len());
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
