// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each independent result is a typed launch argument.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 3, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn decode_tile<const ITEMS_PER_THREAD: usize>(
        values: &[i64],
        lengths: &[u64],
        mut windows: coop::DisjointSlice<i64, coop::CollectiveTiles<3>>,
        mut decoded: coop::DisjointSlice<i64, coop::CollectiveTiles<7>>,
        mut preserved_values: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved_lengths: coop::DisjointSlice<u64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i64, coop::CollectiveTiles<3>>,
        mut totals: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<3>>,
    ) {
        let block = coop::this_block();
        let rank = block.rank_at::<coop::Grid>() as usize;
        let threads = block.count() as usize;
        let input_offset = rank * threads * ITEMS_PER_THREAD;
        let output_offset = rank * threads * 3;
        let (mut window, total, input_values, input_lengths) =
            coop_run_length_decode_helper::decode_tile::<ITEMS_PER_THREAD>(
                block,
                values,
                lengths,
                input_offset,
                &mut decoded,
                (rank * threads * 7 + 3) as u64,
            );
        let mut flags = coop::ThreadData::<_, 3>::new();
        for item in 0..3 {
            flags.write(item, i32::from(window.get(item).is_some()));
        }
        coop::store(
            block,
            &mut windows,
            &window,
            coop::StoreOptions::default().offset(output_offset),
        );
        coop::store(
            block,
            &mut validity,
            &flags,
            coop::StoreOptions::default().offset(output_offset),
        );
        // Mutate only the fresh window after the complete decode has reused
        // implicit scratch. Neither the saved window nor either input aliases it.
        window.write(0, 123_456_789);
        coop::store(
            block,
            &mut mutated,
            &window,
            coop::StoreOptions::default().offset(output_offset),
        );
        coop::store(
            block,
            &mut preserved_values,
            &input_values,
            coop::StoreOptions::default().offset(input_offset),
        );
        coop::store(
            block,
            &mut preserved_lengths,
            &input_lengths,
            coop::StoreOptions::default().offset(input_offset),
        );
        // Observe the numeric total in every lane, without asserting uniformity
        // or feeding it into another collective's control argument.
        coop::store(
            block,
            &mut totals,
            total,
            coop::StoreOptions::default().offset(rank * threads),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    macro_rules! run {
        ($items:literal) => {{
            let tile = 32 * $items;
            let values: Vec<i64> = (0..2 * tile).map(|i| {
                let base = 100 * (i / tile) as i64;
                match i % tile { 0 => 7 + base, 1 => -9 + base, 2 => 2 + base, _ => 1234 + base }
            }).collect();
            let lengths: Vec<u64> = (0..2 * tile).map(|i| match i % tile { 0 => 3, 1 => 194, 2 => 2, _ => 0 }).collect();
            let source_values = DeviceBuffer::from_host(&stream, &values)?;
            let source_lengths = DeviceBuffer::from_host(&stream, &lengths)?;
            let windows_target = DeviceBuffer::from_host(&stream, &vec![-777_777_i64; 192])?;
            let decoded_target = DeviceBuffer::from_host(&stream, &vec![-777_777_i64; 448])?;
            let preserved_values_target = DeviceBuffer::from_host(&stream, &vec![0_i64; values.len()])?;
            let preserved_lengths_target = DeviceBuffer::from_host(&stream, &vec![0_u64; lengths.len()])?;
            let mutated_target = DeviceBuffer::from_host(&stream, &vec![-777_777_i64; 192])?;
            let totals_target = DeviceBuffer::from_host(&stream, &vec![u32::MAX; 64])?;
            let validity_target = DeviceBuffer::from_host(&stream, &vec![-1_i32; 192])?;
            let plan = module.prepare_decode_tile::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (32, 1, 1)))?;
            let (source_values, source_lengths, windows_target, decoded_target, preserved_values_target,
                 preserved_lengths_target, mutated_target, totals_target, validity_target) =
                module.decode_tile_async_owned::<$items, _, _, _, _, _, _, _, _, _>(
                    &plan, source_values, source_lengths, windows_target, decoded_target,
                    preserved_values_target, preserved_lengths_target, mutated_target,
                    totals_target, validity_target,
                ).sync()?;
            let mut windows = vec![0_i64; 192];
            let mut decoded = vec![0_i64; 448];
            let mut preserved_values = vec![0_i64; values.len()];
            let mut preserved_lengths = vec![0_u64; lengths.len()];
            let mut mutated = vec![0_i64; 192];
            let mut totals = vec![0_u32; 64];
            let mut validity = vec![0_i32; 192];
            let mut source_values_after = vec![0_i64; values.len()];
            let mut source_lengths_after = vec![0_u64; lengths.len()];
            windows_target.copy_to_host(&stream, &mut windows)?;
            decoded_target.copy_to_host(&stream, &mut decoded)?;
            preserved_values_target.copy_to_host(&stream, &mut preserved_values)?;
            preserved_lengths_target.copy_to_host(&stream, &mut preserved_lengths)?;
            mutated_target.copy_to_host(&stream, &mut mutated)?;
            totals_target.copy_to_host(&stream, &mut totals)?;
            validity_target.copy_to_host(&stream, &mut validity)?;
            source_values.copy_to_host(&stream, &mut source_values_after)?;
            source_lengths.copy_to_host(&stream, &mut source_lengths_after)?;
            stream.synchronize()?;
            let mut expected_windows = vec![0_i64; 192];
            let mut expected_decoded = vec![-777_777_i64; 448];
            for block in 0..2 {
                let base = 100 * block as i64;
                let stream_values: Vec<i64> = [vec![7 + base; 3], vec![-9 + base; 194], vec![2 + base; 2]].concat();
                expected_windows[block*96..block*96+3].copy_from_slice(&stream_values[196..]);
                expected_decoded[block*224+3..block*224+202].copy_from_slice(&stream_values);
            }
            let mut expected_mutated = expected_windows.clone();
            for i in (0..192).step_by(3) { expected_mutated[i] = 123_456_789; }
            assert_eq!(windows, expected_windows);
            assert_eq!(decoded, expected_decoded);
            assert_eq!(mutated, expected_mutated);
            assert_eq!(totals, vec![199_u32; 64]);
            assert_eq!(validity, vec![1_i32; 192]);
            assert_eq!(preserved_values, values);
            assert_eq!(preserved_lengths, lengths);
            assert_eq!(source_values_after, values);
            assert_eq!(source_lengths_after, lengths);
            println!("items/thread={}: 590 decoded values, {} preserved inputs, {} source values, 64 totals, 192 validity flags, 192 mutated values and 50 untouched destination slots checked", $items, 4 * tile, 4 * tile);
        }};
    }
    run!(1);
    run!(4);
    Ok(())
}
