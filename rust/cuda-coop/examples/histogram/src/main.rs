// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn count_tile<const ITEMS_PER_THREAD: usize>(
        input: &[u8],
        mut output: coop::DisjointSlice<u64, coop::CollectiveTiles<2>>,
        mut preserved: coop::DisjointSlice<u8, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let block_index = block.rank_at::<coop::Grid>() as usize;
        let input_offset = block_index * block.count() as usize * ITEMS_PER_THREAD;
        let output_offset = block_index * block.count() as usize * 2;
        let (counts, values) =
            coop_histogram_helper::count_tile::<_, ITEMS_PER_THREAD>(block, input, input_offset);
        coop::store(
            block,
            &mut output,
            &counts,
            coop::StoreOptions::default()
                .algorithm(coop::Striped)
                .offset(output_offset),
        );
        coop::store(
            block,
            &mut preserved,
            &values,
            coop::StoreOptions::default().offset(input_offset),
        );
        // Terminal safe reads also cover the initialized zero-padding slot.
        assert!(counts.get(0).is_some() && counts.get(1).is_some());
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
            let tile = 64 * $items;
            let input: Vec<u8> = (0..2 * tile)
                .map(|i| if i < tile { 64 } else { ((i - tile) % 65) as u8 })
                .collect();
            // Each Block owns all 128 slots: 65 real bins and 63 padding zeros.
            let mut expected = vec![0_u64; 2 * 128];
            for (block, samples) in input.chunks_exact(tile).enumerate() {
                for &sample in samples {
                    expected[block * 128 + usize::from(sample)] += 1;
                }
            }
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let target = DeviceBuffer::from_host(&stream, &vec![u64::MAX; expected.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![255_u8; input.len()])?;
            let plan = module.prepare_count_tile::<$items>(LaunchConfig1D::auto_shared(2, 64))?;
            let (source, target, preserved_target) = module
                .count_tile_async_owned::<$items, _, _, _>(
                    &plan, source, target, preserved_target,
                )
                .sync()?;
            let mut output = vec![0_u64; expected.len()];
            let mut preserved = vec![0_u8; input.len()];
            let mut source_after = vec![0_u8; input.len()];
            target.copy_to_host(&stream, &mut output)?;
            preserved_target.copy_to_host(&stream, &mut preserved)?;
            source.copy_to_host(&stream, &mut source_after)?;
            stream.synchronize()?;
            assert_eq!(output, expected);
            assert_eq!(preserved, input);
            assert_eq!(source_after, input);
            println!(
                "items/thread={}: {} counters (126 padding), {} preserved inputs and {} source values checked",
                $items, output.len(), preserved.len(), source_after.len()
            );
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
