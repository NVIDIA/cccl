// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The kernel and generated launch wrapper expose separate typed result buffers.
#![allow(clippy::too_many_arguments)]

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn sort_tile<const ITEMS_PER_THREAD: usize>(
        keys: &[i32],
        values: &[f64],
        mut ascending_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut ascending_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated_keys: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated_values: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        begin: i64,
        end: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (mut ascending_k, mut ascending_v, descending_k, descending_v, input_k, input_v) =
            coop_radix_sort_pairs_helper::sort_tile::<_, _, ITEMS_PER_THREAD, 16, 32>(
                block, keys, values, offset, begin, end,
            );
        coop::store(
            block,
            &mut ascending_keys,
            &ascending_k,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut ascending_values,
            &ascending_v,
            coop::StoreOptions::default().offset(offset),
        );
        // Modify fresh outputs after observing the complete original sort.
        // The next Stores observe both independent results and both inputs.
        ascending_k.write(0, 123_456_789);
        ascending_v.write(0, -4096.5);
        coop::store(
            block,
            &mut descending_keys,
            &descending_k,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut descending_values,
            &descending_v,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved_keys,
            &input_k,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved_values,
            &input_v,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut mutated_keys,
            &ascending_k,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut mutated_values,
            &ascending_v,
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
            let keys: Vec<i32> = (0..2 * tile).map(|i| {
                let local = i % tile;
                let block = i / tile;
                let digit = ((local * 5 + block * 3) % 8) as u32;
                // Increasing distinct tags make stable selected-digit sorting
                // distinguishable from whole-key sorting in descending order.
                ((digit << 29 | i as u32) ^ (1_u32 << 31)) as i32
            }).collect();
            let values: Vec<f64> = (0..keys.len()).map(|i| i as f64 + 0.25).collect();
            let pairs: Vec<(i32, f64)> = keys.iter().copied().zip(values.iter().copied()).collect();
            let mut expected_ascending = pairs.clone();
            let mut expected_descending = pairs.clone();
            let digit = |pair: &(i32, f64)| ((pair.0 as u32) ^ (1_u32 << 31)) >> 29;
            for block in expected_ascending.chunks_exact_mut(tile) { block.sort_by_key(digit); }
            for block in expected_descending.chunks_exact_mut(tile) { block.sort_by_key(|pair| Reverse(digit(pair))); }
            let mut whole_key_descending = pairs;
            for block in whole_key_descending.chunks_exact_mut(tile) { block.sort_by_key(|pair| Reverse(pair.0)); }
            assert_ne!(whole_key_descending, expected_descending);
            let (expected_ascending_keys, expected_ascending_values): (Vec<_>, Vec<_>) = expected_ascending.into_iter().unzip();
            let (expected_descending_keys, expected_descending_values): (Vec<_>, Vec<_>) = expected_descending.into_iter().unzip();
            let mut expected_mutated_keys = expected_ascending_keys.clone();
            let mut expected_mutated_values = expected_ascending_values.clone();
            for index in (0..keys.len()).step_by($items) {
                expected_mutated_keys[index] = 123_456_789;
                expected_mutated_values[index] = -4096.5;
            }
            let source_keys = DeviceBuffer::from_host(&stream, &keys)?;
            let source_values = DeviceBuffer::from_host(&stream, &values)?;
            let ascending_keys_target = DeviceBuffer::from_host(&stream, &vec![17 as i32; keys.len()])?;
            let ascending_values_target = DeviceBuffer::from_host(&stream, &vec![17 as f64; keys.len()])?;
            let descending_keys_target = DeviceBuffer::from_host(&stream, &vec![17 as i32; keys.len()])?;
            let descending_values_target = DeviceBuffer::from_host(&stream, &vec![17 as f64; keys.len()])?;
            let preserved_keys_target = DeviceBuffer::from_host(&stream, &vec![17 as i32; keys.len()])?;
            let preserved_values_target = DeviceBuffer::from_host(&stream, &vec![17 as f64; keys.len()])?;
            let mutated_keys_target = DeviceBuffer::from_host(&stream, &vec![17 as i32; keys.len()])?;
            let mutated_values_target = DeviceBuffer::from_host(&stream, &vec![17 as f64; keys.len()])?;
            let plan = module.prepare_sort_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)))?;
            let (source_keys, source_values, ascending_keys_target, ascending_values_target, descending_keys_target, descending_values_target, preserved_keys_target, preserved_values_target, mutated_keys_target, mutated_values_target) = module.sort_tile_async_owned::<$items, _, _, _, _, _, _, _, _, _, _>(&plan, source_keys, source_values, ascending_keys_target, ascending_values_target, descending_keys_target, descending_values_target, preserved_keys_target, preserved_values_target, mutated_keys_target, mutated_values_target, 29, 32).sync()?;
            let mut ascending_keys = vec![0 as i32; keys.len()];
            ascending_keys_target.copy_to_host(&stream, &mut ascending_keys)?;
            let mut ascending_values = vec![0 as f64; keys.len()];
            ascending_values_target.copy_to_host(&stream, &mut ascending_values)?;
            let mut descending_keys = vec![0 as i32; keys.len()];
            descending_keys_target.copy_to_host(&stream, &mut descending_keys)?;
            let mut descending_values = vec![0 as f64; keys.len()];
            descending_values_target.copy_to_host(&stream, &mut descending_values)?;
            let mut preserved_keys = vec![0 as i32; keys.len()];
            preserved_keys_target.copy_to_host(&stream, &mut preserved_keys)?;
            let mut preserved_values = vec![0 as f64; keys.len()];
            preserved_values_target.copy_to_host(&stream, &mut preserved_values)?;
            let mut mutated_keys = vec![0 as i32; keys.len()];
            mutated_keys_target.copy_to_host(&stream, &mut mutated_keys)?;
            let mut mutated_values = vec![0 as f64; keys.len()];
            mutated_values_target.copy_to_host(&stream, &mut mutated_values)?;
            let mut source_keys_after = vec![0_i32; keys.len()];
            let mut source_values_after = vec![0_f64; values.len()];
            source_keys.copy_to_host(&stream, &mut source_keys_after)?;
            source_values.copy_to_host(&stream, &mut source_values_after)?;
            stream.synchronize()?;
            assert_eq!(ascending_keys, expected_ascending_keys);
            assert_eq!(ascending_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected_ascending_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(descending_keys, expected_descending_keys);
            assert_eq!(descending_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected_descending_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(preserved_keys, keys);
            assert_eq!(preserved_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(mutated_keys, expected_mutated_keys);
            assert_eq!(mutated_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected_mutated_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(source_keys_after, keys);
            assert_eq!(source_values_after.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            println!("items/thread={}: {} sorted key/value observations, {} preserved inputs, {} source values and {} mutated outputs checked", $items, 4 * keys.len(), 2 * keys.len(), 2 * keys.len(), 2 * keys.len());
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
