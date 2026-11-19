// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent outputs expose zero tails, destination margins and scratch reuse.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:literal, $d:literal, $b:literal, $cap:literal, $va:literal, $la:literal,
         $manual:literal, $scratch:ident, $values:ident, $lengths:ident,
         $windows:ident, $decoded:ident, $keep_values:ident, $keep_lengths:ident,
         $changed:ident, $totals:ident, $validity:ident,
         $window_offset:ident, $destination_offset:ident, $invalid:ident) => {{
            let block = coop::this_block();
            let rank = block.rank_at::<coop::Grid>() as usize;
            let input_offset = rank * $b * $n;
            let output_offset = rank * $b * $d;
            let destination_offset = rank as u64 * $b * $cap + $destination_offset;
            let mut values = coop::ThreadData::<_, $n, $va>::new();
            let mut lengths = coop::ThreadData::<_, $n, $la>::new();
            if $invalid == 0 {
                coop::load(
                    block,
                    $values,
                    &mut values,
                    coop::LoadOptions::default().offset(input_offset),
                );
                coop::load(
                    block,
                    $lengths,
                    &mut lengths,
                    coop::LoadOptions::default().offset(input_offset),
                );
            } else {
                let lane = thread::threadIdx_x() as usize;
                for item in 0..$n {
                    if !($invalid == 1 && lane == $b - 1 && item == $n - 1) {
                        values.write(item, $values[input_offset + lane * $n + item]);
                    }
                    if !($invalid == 2 && lane == $b - 1 && item == $n - 1) {
                        lengths.write(item, $lengths[input_offset + lane * $n + item]);
                    }
                }
                if $invalid == 1 {
                    let _ = coop::run_length_decode(
                        block,
                        &values,
                        &lengths,
                        coop::RunLengthDecodeOptions::<$d>::default().temp_storage(&mut $scratch),
                    );
                } else if $invalid == 7 {
                    let _ = coop::run_length_decode(
                        block,
                        &values,
                        &lengths,
                        coop::RunLengthDecodeOptions::<$d>::default()
                            .decoded_window_offset(-1_i64)
                            .temp_storage(&mut $scratch),
                    );
                } else {
                    let _ = coop::run_length_decode_into(
                        block,
                        &values,
                        &lengths,
                        &mut $decoded,
                        coop::RunLengthDecodeIntoOptions::<$d>::default()
                            .destination_offset(destination_offset)
                            .temp_storage(&mut $scratch),
                    );
                }
                thread::sync_threads();
                // Never read an invalid operation's result. Only an independent
                // report is written if native preflight unexpectedly returns.
                let mut report = coop::ThreadData::<_, $d>::new();
                for item in 0..$d {
                    report.write(item, 99 as _);
                }
                coop::store(
                    block,
                    &mut $changed,
                    &report,
                    coop::StoreOptions::default().offset(output_offset),
                );
                return;
            }
            let mut window = coop::run_length_decode(
                block,
                &values,
                &lengths,
                coop::RunLengthDecodeOptions::<$d>::default()
                    .decoded_window_offset($window_offset)
                    .temp_storage(&mut $scratch),
            );
            if $manual {
                thread::sync_threads();
            }
            let total = coop::run_length_decode_into(
                block,
                &values,
                &lengths,
                &mut $decoded,
                coop::RunLengthDecodeIntoOptions::<$d>::default()
                    .temp_storage(&mut $scratch)
                    .destination_offset(destination_offset),
            );
            if $manual {
                thread::sync_threads();
            }
            // Keep the complete fresh window across Into's reuse, then mutate
            // it independently of the decoded destination and both inputs.
            coop::store(
                block,
                &mut $windows,
                &window,
                coop::StoreOptions::default().offset(output_offset),
            );
            let mut flags = coop::ThreadData::<_, $d>::new();
            for item in 0..$d {
                flags.write(item, i32::from(window.get(item).is_some()));
            }
            window.write(0, 99 as _);
            coop::store(
                block,
                &mut $changed,
                &window,
                coop::StoreOptions::default().offset(output_offset),
            );
            coop::store(
                block,
                &mut $keep_values,
                &values,
                coop::StoreOptions::default().offset(input_offset),
            );
            coop::store(
                block,
                &mut $keep_lengths,
                &lengths,
                coop::StoreOptions::default().offset(input_offset),
            );
            coop::store(
                block,
                &mut $totals,
                total,
                coop::StoreOptions::default().offset(rank * $b),
            );
            coop::store(
                block,
                &mut $validity,
                &flags,
                coop::StoreOptions::default().offset(output_offset),
            );
        }};
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn f32_i32_manual(
        values: &[f32],
        lengths: &[i32],
        mut windows: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut decoded: coop::DisjointSlice<f32, coop::CollectiveTiles<10>>,
        mut keep_values: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut keep_lengths: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut changed: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut totals: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        window_offset: u64,
        destination_offset: u64,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new();
        exercise!(
            9,
            9,
            33,
            10,
            16,
            32,
            true,
            scratch,
            values,
            lengths,
            windows,
            decoded,
            keep_values,
            keep_lengths,
            changed,
            totals,
            validity,
            window_offset,
            destination_offset,
            invalid
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (16, 1, 1), dynamic_shared = inferred)]
    pub fn u8_u64_auto(
        values: &[u8],
        lengths: &[u64],
        mut windows: coop::DisjointSlice<u8, coop::CollectiveTiles<3>>,
        mut decoded: coop::DisjointSlice<u8, coop::CollectiveTiles<6>>,
        mut keep_values: coop::DisjointSlice<u8, coop::CollectiveTiles<4>>,
        mut keep_lengths: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut changed: coop::DisjointSlice<u8, coop::CollectiveTiles<3>>,
        mut totals: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<3>>,
        window_offset: u64,
        destination_offset: u64,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        exercise!(
            4,
            3,
            16,
            6,
            1,
            16,
            false,
            scratch,
            values,
            lengths,
            windows,
            decoded,
            keep_values,
            keep_lengths,
            changed,
            totals,
            validity,
            window_offset,
            destination_offset,
            invalid
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("missing-value-empty") => 1,
        Some("missing-length-empty") => 2,
        Some("negative-length") => 3,
        Some("interior-zero") => 4,
        Some("total-overflow") => 5,
        Some("empty-destination-offset") => 6,
        Some("negative-window-offset") => 7,
        Some(other) => panic!("unknown Run Length Decode mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (
        mut cases,
        mut decoded_count,
        mut preserved,
        mut sources,
        mut totals_checked,
        mut validity_checked,
        mut mutated,
        mut untouched,
    ) = (0, 0, 0, 0, 0, 0, 0, 0);
    macro_rules! run {
        ($t:ty, $l:ty, $n:literal, $d:literal, $b:literal, $cap:literal,
         $prepare:ident, $launch:ident, $values:expr, $lengths:expr,
         $window_offset:expr, $destination_offset:expr, $bits:expr) => {{
            let values: Vec<$t> = $values;
            let lengths: Vec<$l> = $lengths;
            let blocks = values.len() / ($b * $n);
            let window_offset: u64 = $window_offset;
            let destination_offset: u64 = $destination_offset;
            let bits = $bits;
            let bit_vec = |v: &[$t]| v.iter().copied().map(bits).collect::<Vec<_>>();
            assert_eq!(values.len(), lengths.len());
            let source_v = DeviceBuffer::from_host(&stream, &values)?;
            let source_l = DeviceBuffer::from_host(&stream, &lengths)?;
            let windows = DeviceBuffer::from_host(&stream, &vec![123 as $t; blocks * $b * $d])?;
            let destination =
                DeviceBuffer::from_host(&stream, &vec![123 as $t; blocks * $b * $cap])?;
            let keep_v = DeviceBuffer::from_host(&stream, &vec![123 as $t; values.len()])?;
            let keep_l = DeviceBuffer::from_host(&stream, &vec![123 as $l; lengths.len()])?;
            let changed = DeviceBuffer::from_host(&stream, &vec![123 as $t; blocks * $b * $d])?;
            let totals = DeviceBuffer::from_host(&stream, &vec![u32::MAX; blocks * $b])?;
            let flags = DeviceBuffer::from_host(&stream, &vec![-1_i32; blocks * $b * $d])?;
            stream.synchronize()?;
            let plan = module.$prepare(LaunchConfig1D::auto_shared(blocks as u32, $b))?;
            let (source_v, source_l, windows, destination, keep_v, keep_l, changed, totals, flags) =
                module
                    .$launch(
                        &plan,
                        source_v,
                        source_l,
                        windows,
                        destination,
                        keep_v,
                        keep_l,
                        changed,
                        totals,
                        flags,
                        window_offset,
                        destination_offset,
                        invalid,
                    )
                    .sync()?;
            stream.synchronize()?;
            if invalid != 0 {
                panic!("invalid Run Length Decode mode {invalid} unexpectedly completed");
            }
            let window = windows.to_host_vec(&stream)?;
            let destination = destination.to_host_vec(&stream)?;
            let totals = totals.to_host_vec(&stream)?;
            let flags = flags.to_host_vec(&stream)?;
            let mut expected_mutation = Vec::new();
            for block in 0..blocks {
                let base = block * $b * $n;
                let mut expanded = Vec::new();
                for i in base..base + $b * $n {
                    let count = usize::try_from(lengths[i]).expect("positive test length");
                    expanded.extend(std::iter::repeat_n(values[i], count));
                }
                let total = expanded.len();
                let expected_window: Vec<$t> = (0..$b * $d)
                    .map(|i| {
                        window_offset
                            .checked_add(i as u64)
                            .and_then(|p| usize::try_from(p).ok())
                            .and_then(|p| expanded.get(p).copied())
                            .unwrap_or(0 as $t)
                    })
                    .collect();
                let start = block * $b * $d;
                assert_eq!(
                    bit_vec(&window[start..start + $b * $d]),
                    bit_vec(&expected_window)
                );
                let mut expected_destination = vec![123 as $t; $b * $cap];
                let offset = usize::try_from(destination_offset).unwrap();
                expected_destination[offset..offset + total].copy_from_slice(&expanded);
                let dest_start = block * $b * $cap;
                assert_eq!(
                    bit_vec(&destination[dest_start..dest_start + $b * $cap]),
                    bit_vec(&expected_destination)
                );
                assert!(
                    totals[block * $b..(block + 1) * $b]
                        .iter()
                        .all(|&v| v == total as u32)
                );
                assert!(flags[start..start + $b * $d].iter().all(|&v| v == 1));
                let mut changed_window = expected_window;
                for item in (0..$b * $d).step_by($d) {
                    changed_window[item] = 99 as $t;
                }
                expected_mutation.extend(changed_window);
                decoded_count += $b * $d + total;
                untouched += $b * $cap - total;
            }
            assert_eq!(bit_vec(&keep_v.to_host_vec(&stream)?), bit_vec(&values));
            assert_eq!(keep_l.to_host_vec(&stream)?, lengths);
            assert_eq!(bit_vec(&source_v.to_host_vec(&stream)?), bit_vec(&values));
            assert_eq!(source_l.to_host_vec(&stream)?, lengths);
            assert_eq!(
                bit_vec(&changed.to_host_vec(&stream)?),
                bit_vec(&expected_mutation)
            );
            cases += 1;
            preserved += 2 * values.len();
            sources += 2 * values.len();
            totals_checked += blocks * $b;
            validity_checked += blocks * $b * $d;
            mutated += blocks * $b * $d;
        }};
    }
    if invalid == 0 || invalid == 3 || invalid == 4 {
        let empties: &[bool] = if invalid == 0 {
            &[false, true]
        } else {
            &[false]
        };
        for &empty in empties {
            let blocks = if invalid == 0 { 2 } else { 1 };
            let len = blocks * 33 * 9;
            let values = (0..len)
                .map(|i| match i % 297 {
                    0 => -0.0_f32,
                    1 => 1.5,
                    2 => -2.25,
                    _ => 77.0,
                })
                .collect();
            let mut lengths: Vec<i32> = (0..len)
                .map(|i| {
                    if empty {
                        0
                    } else {
                        match i % 297 {
                            0 => 2,
                            1 => 298,
                            2 => 3,
                            _ => 0,
                        }
                    }
                })
                .collect();
            if invalid == 3 {
                lengths[0] = -1;
            }
            if invalid == 4 {
                lengths[1] = 0;
            }
            run!(
                f32,
                i32,
                9,
                9,
                33,
                10,
                prepare_f32_i32_manual,
                f32_i32_manual_async_owned,
                values,
                lengths,
                0,
                5,
                |v: f32| v.to_bits()
            );
        }
    }
    if !matches!(invalid, 3 | 4) {
        let empties: &[bool] = if invalid == 0 {
            &[false, true]
        } else {
            &[true]
        };
        for &empty in empties {
            let blocks = if invalid == 0 { 2 } else { 1 };
            let len = blocks * 16 * 4;
            let values = (0..len)
                .map(|i| match i % 64 {
                    0 => 7_u8,
                    1 => 9,
                    2 => 2,
                    _ => 77,
                })
                .collect();
            let mut lengths: Vec<u64> = (0..len)
                .map(|i| {
                    if empty {
                        0
                    } else {
                        match i % 64 {
                            0 => 3,
                            1 => 50,
                            2 => 2,
                            _ => 0,
                        }
                    }
                })
                .collect();
            if invalid == 5 {
                lengths[0] = u32::MAX as u64;
                lengths[1] = 1;
            }
            let destination_offset = if invalid == 6 { u64::MAX } else { 3 };
            run!(
                u8,
                u64,
                4,
                3,
                16,
                6,
                prepare_u8_u64_auto,
                u8_u64_auto_async_owned,
                values,
                lengths,
                if empty { u64::MAX } else { 54 },
                destination_offset,
                |v: u8| v
            );
        }
    }
    assert_eq!(
        (
            cases,
            decoded_count,
            preserved,
            sources,
            totals_checked,
            validity_checked,
            mutated,
            untouched
        ),
        (4, 2096, 2888, 2888, 196, 1380, 1380, 988)
    );
    println!(
        "run_length_decode: {cases} cases, {decoded_count} decoded, {preserved} preserved, {sources} source, {totals_checked} totals, {validity_checked} validity, {mutated} mutated, {untouched} untouched slots passed"
    );
    Ok(())
}
