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
        input: &[i64],
        mut full: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut selected: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut descending: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        begin: i64,
        end: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (full_keys, selected_keys, descending_keys, values) =
            coop_radix_sort_keys_helper::sort_tile::<_, ITEMS_PER_THREAD, 16>(
                block, input, offset, begin, end,
            );
        coop::store(
            block,
            &mut full,
            &full_keys,
            coop::StoreOptions::default().offset(offset),
        );
        coop::store(
            block,
            &mut selected,
            &selected_keys,
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
            &values,
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
            let input: Vec<i64> = (0..2 * tile)
                .map(|i| {
                    let local = i % tile;
                    let block = i / tile;
                    match local % 23 {
                        0 => i64::MIN,
                        1 => i64::MAX,
                        2 => -1,
                        3 => 0,
                        _ => {
                            let digit = ((local * 5 + block * 3) % 8) as u64;
                            let tag = (tile - local + block * tile) as u64;
                            ((digit << 61 | tag) ^ (1_u64 << 63)) as i64
                        }
                    }
                })
                .collect();
            let mut expected_full = input.clone();
            let mut expected_selected = input.clone();
            let mut expected_descending = input.clone();
            for values in expected_full.chunks_exact_mut(tile) {
                values.sort();
            }
            // Stable host sorting retains the original order of distinct keys
            // sharing selected digits. Signed normalization precedes selection.
            let digit = |key: &i64| ((*key as u64) ^ (1_u64 << 63)) >> 61;
            for values in expected_selected.chunks_exact_mut(tile) {
                values.sort_by_key(digit);
            }
            for values in expected_descending.chunks_exact_mut(tile) {
                values.sort_by_key(|key| Reverse(digit(key)));
            }
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let full_target = DeviceBuffer::from_host(&stream, &vec![17_i64; input.len()])?;
            let selected_target = DeviceBuffer::from_host(&stream, &vec![17_i64; input.len()])?;
            let descending_target = DeviceBuffer::from_host(&stream, &vec![17_i64; input.len()])?;
            let preserved_target = DeviceBuffer::from_host(&stream, &vec![17_i64; input.len()])?;
            let plan = module
                .prepare_sort_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)))?;
            let (source, full_target, selected_target, descending_target, preserved_target) =
                module
                    .sort_tile_async_owned::<$items, _, _, _, _, _>(
                        &plan,
                        source,
                        full_target,
                        selected_target,
                        descending_target,
                        preserved_target,
                        61,
                        64,
                    )
                    .sync()?;
            let mut full = vec![0_i64; input.len()];
            let mut selected = vec![0_i64; input.len()];
            let mut descending = vec![0_i64; input.len()];
            let mut preserved = vec![0_i64; input.len()];
            let mut source_after = vec![0_i64; input.len()];
            full_target.copy_to_host(&stream, &mut full)?;
            selected_target.copy_to_host(&stream, &mut selected)?;
            descending_target.copy_to_host(&stream, &mut descending)?;
            preserved_target.copy_to_host(&stream, &mut preserved)?;
            source.copy_to_host(&stream, &mut source_after)?;
            stream.synchronize()?;
            assert_eq!(full, expected_full);
            assert_eq!(selected, expected_selected);
            assert_eq!(descending, expected_descending);
            assert_eq!(preserved, input);
            assert_eq!(source_after, input);
            println!(
                "items/thread={}: {} sorted keys, {} preserved inputs and {} source values checked",
                $items,
                3 * input.len(),
                preserved.len(),
                source_after.len()
            );
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
