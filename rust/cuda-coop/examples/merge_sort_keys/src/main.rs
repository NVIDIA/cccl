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
    pub fn sort_tile<const ITEMS_PER_THREAD: usize>(
        keys: &[i64],
        mut ascending: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        valid_items: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (mut ascending_keys, descending_keys, input) =
            coop_merge_sort_keys_helper::sort_tile::<_, ITEMS_PER_THREAD, 32>(
                block,
                keys,
                offset,
                valid_items,
                i64::MIN,
            );
        let mut flags = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        for item in 0..ITEMS_PER_THREAD {
            flags.write(item, i32::from(descending_keys.get(item).is_some()));
        }
        coop::store(
            block,
            &mut ascending,
            &ascending_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut descending,
            &descending_keys,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(valid_items as usize),
        );
        // Observe a mutation of the fresh result separately from the original.
        ascending_keys.write(0, -9999);
        coop::store(
            block,
            &mut mutated,
            &ascending_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved,
            &input,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut validity,
            &flags,
            coop::StoreOptions::default().offset(offset),
        );
    }
    #[kernel]
    #[launch_contract(domain = 3, block = (128, 1, 1), dynamic_shared = inferred)]
    pub fn compare_tile<const ITEMS_PER_THREAD: usize>(
        keys: &[i32],
        mut ascending: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut even_first: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let mut input = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        coop::load(
            block,
            keys,
            &mut input,
            coop::LoadOptions::default().offset(offset),
        );
        let mut scratch = coop::TempStorage::new().auto_sync();
        let mut ascending_keys = coop_merge_sort_keys_helper::sort_values(
            block,
            &input,
            coop::MergeSortOptions::default().temp_storage(&mut scratch),
        );
        let even_keys = coop_merge_sort_keys_helper::sort_values(
            block,
            &input,
            coop::MergeSortOptions::default()
                .order(coop_merge_sort_keys_helper::even_first)
                .temp_storage(&mut scratch),
        );
        let descending_keys = coop_merge_sort_keys_helper::sort_values(
            block,
            &input,
            coop::MergeSortOptions::default().order(|left, right| left > right),
        );
        // All three fresh results survive explicit and implicit scratch use.
        let mut flags = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        for item in 0..ITEMS_PER_THREAD {
            flags.write(
                item,
                i32::from(ascending_keys.get(item).is_some())
                    | (i32::from(even_keys.get(item).is_some()) << 1)
                    | (i32::from(descending_keys.get(item).is_some()) << 2),
            );
        }
        coop::store(
            block,
            &mut ascending,
            &ascending_keys,
            coop::StoreOptions::default().offset(offset),
        );
        ascending_keys.write(0, -9999);
        coop::store(
            block,
            &mut mutated,
            &ascending_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut even_first,
            &even_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut descending,
            &descending_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut preserved,
            &input,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut validity,
            &flags,
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
            let count = tile - 1;
            let keys: Vec<i64> = (0..2 * tile).map(|i| {
                let local = i % tile;
                let block = i / tile;
                // Repeated signed keys exercise comparisons. The excluded key
                // would lead the descending result if the count were ignored.
                if local == count { 1000 + block as i64 }
                else { ((local * 17 + block * 11) % 23) as i64 - 11 }
            }).collect();
            let mut expected_ascending = Vec::with_capacity(keys.len());
            let mut expected_descending = Vec::with_capacity(keys.len());
            let mut expected_validity = Vec::with_capacity(keys.len());
            for input in keys.chunks_exact(tile) {
                let mut ascending = input.to_vec();
                ascending.sort_unstable();
                expected_ascending.extend(ascending);
                let mut descending = input[..count].to_vec();
                descending.sort_unstable_by_key(|value| Reverse(*value));
                expected_descending.extend(descending);
                expected_descending.push(-777);
                expected_validity.extend(std::iter::repeat_n(1_i32, count));
                expected_validity.push(0);
            }
            let mut expected_mutated = expected_ascending.clone();
            for lane in expected_mutated.chunks_exact_mut($items) { lane[0] = -9999; }
            let source = DeviceBuffer::from_host(&stream, &keys)?;
            let ascending_target = DeviceBuffer::from_host(&stream, &vec![-777_i64; keys.len()])?;
            let descending_target = DeviceBuffer::from_host(&stream, &vec![-777_i64; keys.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![17_i64; keys.len()])?;
            let validity_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let mutated_target = DeviceBuffer::from_host(&stream, &vec![17_i64; keys.len()])?;
            let plan = module.prepare_sort_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)))?;
            let (source, ascending_target, descending_target, preserved_target, validity_target, mutated_target) =
                module.sort_tile_async_owned::<$items, _, _, _, _, _, _>(
                    &plan, source, ascending_target, descending_target, preserved_target, validity_target, mutated_target, count as i64,
                ).sync()?;
            let mut ascending = vec![0_i64; keys.len()];
            let mut descending = vec![0_i64; keys.len()];
            let mut preserved = vec![0_i64; keys.len()];
            let mut source_after = vec![0_i64; keys.len()];
            let mut validity = vec![0_i32; keys.len()];
            let mut mutated = vec![0_i64; keys.len()];
            ascending_target.copy_to_host(&stream, &mut ascending)?;
            descending_target.copy_to_host(&stream, &mut descending)?;
            preserved_target.copy_to_host(&stream, &mut preserved)?;
            source.copy_to_host(&stream, &mut source_after)?;
            validity_target.copy_to_host(&stream, &mut validity)?;
            mutated_target.copy_to_host(&stream, &mut mutated)?;
            stream.synchronize()?;
            assert_eq!(ascending, expected_ascending);
            assert_eq!(descending, expected_descending);
            assert_eq!(preserved, keys);
            assert_eq!(source_after, keys);
            assert_eq!(validity, expected_validity);
            assert_eq!(mutated, expected_mutated);
            println!("items/thread={}: {} sorted values, {} preserved keys, {} source keys, {} validity flags and {} mutated values checked", $items, keys.len() + 2 * count, keys.len(), keys.len(), keys.len(), keys.len());
        }};
    }
    run!(1);
    run!(4);
    macro_rules! run_callable {
        ($items:literal) => {{
            let tile = 128 * $items;
            let keys: Vec<i32> = (0..3 * tile).map(|i| match i % tile {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => ((i * 17 + i / tile * 11) % 23) as i32 - 11,
            }).collect();
            let mut expected_ascending = Vec::new();
            let mut expected_even = Vec::new();
            let mut expected_descending = Vec::new();
            for input in keys.chunks_exact(tile) {
                let mut sorted = input.to_vec();
                sorted.sort_unstable();
                expected_ascending.extend_from_slice(&sorted);
                sorted.sort_unstable_by_key(|value| (value & 1, *value));
                expected_even.extend_from_slice(&sorted);
                sorted.sort_unstable_by_key(|value| Reverse(*value));
                expected_descending.extend(sorted);
            }
            let mut expected_mutated = expected_ascending.clone();
            for lane in expected_mutated.chunks_exact_mut($items) { lane[0] = -9999; }
            let source = DeviceBuffer::from_host(&stream, &keys)?;
            let ascending_target = DeviceBuffer::from_host(&stream, &vec![-777_i32; keys.len()])?;
            let even_target = DeviceBuffer::from_host(&stream, &vec![-777_i32; keys.len()])?;
            let descending_target = DeviceBuffer::from_host(&stream, &vec![-777_i32; keys.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![-777_i32; keys.len()])?;
            let validity_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let mutated_target = DeviceBuffer::from_host(&stream, &vec![-777_i32; keys.len()])?;
            let plan = module.prepare_compare_tile::<$items>(LaunchConfig3D::auto_shared((3, 1, 1), (128, 1, 1)))?;
            let (source, ascending_target, even_target, descending_target, preserved_target, validity_target, mutated_target) =
                module.compare_tile_async_owned::<$items, _, _, _, _, _, _, _>(
                    &plan, source, ascending_target, even_target, descending_target, preserved_target, validity_target, mutated_target,
                ).sync()?;
            for (target, expected) in [
                (&ascending_target, &expected_ascending),
                (&even_target, &expected_even),
                (&descending_target, &expected_descending),
                (&preserved_target, &keys),
                (&source, &keys),
                (&validity_target, &vec![7_i32; keys.len()]),
                (&mutated_target, &expected_mutated),
            ] {
                let mut actual = vec![0_i32; keys.len()];
                target.copy_to_host(&stream, &mut actual)?;
                stream.synchronize()?;
                assert_eq!(&actual, expected);
            }
            println!("callable items/thread={}: {} sorted values, {} preserved keys, {} source keys, {} validity flags and {} mutated values checked", $items, 3 * keys.len(), keys.len(), keys.len(), keys.len(), keys.len());
        }};
    }
    run_callable!(1);
    run_callable!(4);
    Ok(())
}
