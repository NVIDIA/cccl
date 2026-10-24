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
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        valid_items: i64,
    ) {
        let block = coop::this_block();
        let offset =
            block.rank_at::<coop::Grid>() as usize * block.count() as usize * ITEMS_PER_THREAD;
        let (mut ascending_k, mut ascending_v, descending_k, descending_v, input_k, input_v) =
            coop_merge_sort_pairs_helper::sort_tile::<_, _, ITEMS_PER_THREAD, 16, 32>(
                block,
                keys,
                values,
                offset,
                valid_items,
                i32::MIN,
            );
        let mut flags = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        for item in 0..ITEMS_PER_THREAD {
            flags.write(
                item,
                i32::from(input_k.get(item).is_some())
                    | (i32::from(input_v.get(item).is_some()) << 1)
                    | (i32::from(descending_k.get(item).is_some()) << 2)
                    | (i32::from(descending_v.get(item).is_some()) << 3),
            );
        }
        coop::store(
            block,
            &mut validity,
            &flags,
            coop::StoreOptions::default().offset(offset),
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
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(valid_items as usize),
        );
        coop::store(
            block,
            &mut descending_values,
            &descending_v,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(valid_items as usize),
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

// CUB Merge Sort does not promise stable ties. Verify numeric key order and
// the exact associated (key, value bits) multiset, not a tie permutation.
fn check_pairs(
    input_k: &[i32],
    input_v: &[f64],
    output_k: &[i32],
    output_v: &[f64],
    descending: bool,
) {
    assert!(output_k.windows(2).all(|w| if descending {
        w[0] >= w[1]
    } else {
        w[0] <= w[1]
    }));
    let mut expected: Vec<_> = input_k
        .iter()
        .copied()
        .zip(input_v.iter().map(|v| v.to_bits()))
        .collect();
    let mut actual: Vec<_> = output_k
        .iter()
        .copied()
        .zip(output_v.iter().map(|v| v.to_bits()))
        .collect();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 64 * $items;
            let count = tile - 1;
            let keys: Vec<i32> = (0..2 * tile).map(|i| {
                let local = i % tile;
                let block = i / tile;
                // Repeated signed keys allow arbitrary tie order. The excluded
                // suffix would lead descending output if the count were ignored.
                if local == count { 1000 + block as i32 }
                else { ((local * 17 + block * 11) % 23) as i32 - 11 }
            }).collect();
            let values: Vec<f64> = (0..keys.len()).map(|i| {
                // Values are carried, never compared: preserve NaN payloads,
                // signed zeros, and distinct finite association tags bitwise.
                match i {
                    0 => 0.0,
                    1 => -0.0,
                    _ if i % 7 == 0 => f64::from_bits(0x7ff8_0000_0000_0000 | i as u64),
                    _ => i as f64 + 0.25,
                }
            }).collect();
            let mut expected_validity = Vec::with_capacity(keys.len());
            for _ in 0..2 {
                expected_validity.extend(std::iter::repeat_n(15_i32, count));
                expected_validity.push(3);
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
            let validity_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let plan = module.prepare_sort_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)))?;
            let (source_keys, source_values, ascending_keys_target, ascending_values_target, descending_keys_target, descending_values_target, preserved_keys_target, preserved_values_target, mutated_keys_target, mutated_values_target, validity_target) = module.sort_tile_async_owned::<$items, _, _, _, _, _, _, _, _, _, _, _>(&plan, source_keys, source_values, ascending_keys_target, ascending_values_target, descending_keys_target, descending_values_target, preserved_keys_target, preserved_values_target, mutated_keys_target, mutated_values_target, validity_target, count as i64).sync()?;
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
            let mut validity = vec![0_i32; keys.len()];
            validity_target.copy_to_host(&stream, &mut validity)?;
            stream.synchronize()?;
            for block in 0..2 {
                let base = block * tile;
                check_pairs(&keys[base..base+tile], &values[base..base+tile],
                    &ascending_keys[base..base+tile], &ascending_values[base..base+tile], false);
                check_pairs(&keys[base..base+count], &values[base..base+count],
                    &descending_keys[base..base+count], &descending_values[base..base+count], true);
                assert_eq!(descending_keys[base+count], 17);
                assert_eq!(descending_values[base+count].to_bits(), 17_f64.to_bits());
            }
            let mut expected_mutated_keys = ascending_keys.clone();
            let mut expected_mutated_values = ascending_values.clone();
            for index in (0..keys.len()).step_by($items) {
                expected_mutated_keys[index] = 123_456_789;
                expected_mutated_values[index] = -4096.5;
            }
            assert_eq!(validity, expected_validity);
            assert_eq!(preserved_keys, keys);
            assert_eq!(preserved_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(mutated_keys, expected_mutated_keys);
            assert_eq!(mutated_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected_mutated_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            assert_eq!(source_keys_after, keys);
            assert_eq!(source_values_after.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            println!("items/thread={}: {} sorted key/value observations, {} preserved inputs, {} source values, {} validity flags and {} mutated outputs checked", $items, 2 * keys.len() + 4 * count, 2 * keys.len(), 2 * keys.len(), keys.len(), 2 * keys.len());
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
