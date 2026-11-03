// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn rank_tile<const ITEMS_PER_THREAD: usize>(
        keys: &[i64],
        mut ascending: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        begin: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (ascending_ranks, descending_ranks, input) =
            coop_radix_rank_keys_helper::rank_tile::<_, ITEMS_PER_THREAD, 32>(
                block, keys, offset, begin,
            );
        coop::store(
            block,
            &mut ascending,
            &ascending_ranks,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut descending,
            &descending_ranks,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved,
            &input,
            coop::StoreOptions::default().offset(offset),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};
    use std::cmp::Reverse;

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 64 * $items;
            let keys: Vec<i64> = (0..2 * tile).map(|i| {
                let local = i % tile;
                let block = i / tile;
                let digit = ((local * 5 + block * 3) % 8) as u64;
                // Distinct increasing low tags make stable digit ranking differ
                // from whole-key descending ranking on repeated high digits.
                ((digit << 61 | i as u64) ^ (1_u64 << 63)) as i64
            }).collect();
            let mut expected_ascending = vec![0_i32; keys.len()];
            let mut expected_descending = vec![0_i32; keys.len()];
            for (block, input) in keys.chunks_exact(tile).enumerate() {
                let digit = |index: &usize| ((input[*index] as u64) ^ (1_u64 << 63)) >> 61;
                let mut ascending: Vec<usize> = (0..tile).collect();
                let mut descending = ascending.clone();
                ascending.sort_by_key(digit);
                descending.sort_by_key(|index| Reverse(digit(index)));
                let mut full_key = (0..tile).collect::<Vec<_>>();
                full_key.sort_by_key(|index| Reverse(input[*index]));
                assert_ne!(full_key, descending);
                // Rank belongs to the original item, so invert each sorted
                // permutation rather than storing the sorted index sequence.
                for (rank, index) in ascending.into_iter().enumerate() {
                    expected_ascending[block * tile + index] = rank as i32;
                }
                for (rank, index) in descending.into_iter().enumerate() {
                    expected_descending[block * tile + index] = rank as i32;
                }
            }
            let source = DeviceBuffer::from_host(&stream, &keys)?;
            let ascending_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let descending_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![17_i64; keys.len()])?;
            let plan = module.prepare_rank_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)))?;
            let (source, ascending_target, descending_target, preserved_target) = module.rank_tile_async_owned::<$items, _, _, _, _>(&plan, source, ascending_target, descending_target, preserved_target, 61).sync()?;
            let mut ascending = vec![0_i32; keys.len()];
            let mut descending = vec![0_i32; keys.len()];
            let mut preserved = vec![0_i64; keys.len()];
            let mut source_after = vec![0_i64; keys.len()];
            ascending_target.copy_to_host(&stream, &mut ascending)?;
            descending_target.copy_to_host(&stream, &mut descending)?;
            preserved_target.copy_to_host(&stream, &mut preserved)?;
            source.copy_to_host(&stream, &mut source_after)?;
            stream.synchronize()?;
            assert_eq!(ascending, expected_ascending);
            assert_eq!(descending, expected_descending);
            assert_eq!(preserved, keys);
            assert_eq!(source_after, keys);
            println!("items/thread={}: {} rank observations, {} preserved keys and {} source keys checked", $items, 2 * keys.len(), keys.len(), keys.len());
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
