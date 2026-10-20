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
    pub fn copy<const ITEMS_PER_THREAD: usize>(
        input: &[i16],
        mut output: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let tile = block.count() as usize * ITEMS_PER_THREAD;
        let offset = block.rank_at::<coop::Grid>() as usize * tile;
        let remaining = if offset <= input.len() {
            input.len() - offset
        } else {
            0
        };
        let valid = remaining.min(tile);
        coop_load_store_helper::copy::<_, ITEMS_PER_THREAD>(
            block,
            input,
            &mut output,
            offset,
            valid,
            -7_i16,
        );
    }
    #[kernel]
    #[launch_contract(domain = 1, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn transpose<const ITEMS_PER_THREAD: usize>(
        input: &[i16],
        mut output: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let tile = block.count() as usize * ITEMS_PER_THREAD;
        let offset = block.rank_at::<coop::Grid>() as usize * tile;
        let remaining = if offset <= input.len() {
            input.len() - offset
        } else {
            0
        };
        let valid = remaining.min(tile);
        coop_load_store_helper::copy_with_algorithm::<_, ITEMS_PER_THREAD, _>(
            block,
            input,
            &mut output,
            offset,
            valid,
            -7_i16,
            coop::Transpose,
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
    // SAFETY: these typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal, $prepare:ident, $launch:ident) => {{
            let input: Vec<i16> = (0..3 * 128 * $items - 7)
                .map(|i| (i % 251) as i16)
                .collect();
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![-1_i16; input.len()])?;
            let plan = module.$prepare::<$items>(LaunchConfig1D::auto_shared(3, 128))?;
            let (_, target) = module
                .$launch::<$items, _, _>(&plan, source, target)
                .sync()?;
            let mut output = vec![0_i16; input.len()];
            target.copy_to_host(&stream, &mut output)?;
            stream.synchronize()?;
            assert_eq!(output, input);
            println!("items/thread={}: {} values copied", $items, output.len());
        }};
    }
    run!(1, prepare_copy, copy_async_owned);
    run!(4, prepare_copy, copy_async_owned);
    run!(1, prepare_transpose, transpose_async_owned);
    run!(4, prepare_transpose, transpose_async_owned);
    Ok(())
}
