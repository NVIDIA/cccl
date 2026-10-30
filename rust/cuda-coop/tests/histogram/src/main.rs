// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const CHECKS: u32 = 7;

#[cuda_module]
mod kernels {
    use super::*;

    fn counts<T, C, A, S, const BINS: usize, const N: usize, const K: usize, const ALIGN: usize>(
        block: coop::ThreadGroup<coop::Block>,
        input: &coop::ThreadData<T, N, ALIGN>,
        options: coop::HistogramOptions<BINS, C, K, A, S>,
    ) -> coop::ThreadData<C, K>
    where
        T: coop::HistogramSample,
        C: coop::HistogramCounter,
        A: coop::HistogramAlgorithm,
        S: coop::TempStorageLike,
        coop::Alignment<ALIGN>: coop::SupportedAlignment,
    {
        coop::histogram(block, input, options)
    }

    fn observe<C: coop::HistogramCounter + PartialEq, const K: usize, const B: usize>(
        counts: &coop::ThreadData<C, K>,
        expected: &[C],
        offset: usize,
        marker: Option<C>,
    ) -> bool {
        let mut valid = true;
        let mut i = 0;
        while i < K {
            let value = if i == 0 {
                marker.unwrap_or(expected[offset])
            } else {
                expected[offset + i * B]
            };
            valid &= counts.get(i) == Some(value);
            i += 1;
        }
        valid
    }

    // Each root expands exactly one static algorithm/storage choice. Repeated
    // expressions create ordinary reborrows of an explicit descriptor.
    macro_rules! exercise {
        ($n:literal, $align:literal, $b:literal, $k:literal, $options:expr,
         $manual:literal, $filled:literal,
         $input:ident, $expected:ident, $output:ident, $preserved:ident,
         $changed_output:ident, $reports:ident, $replacement:expr, $marker:expr,
         $invalid:ident) => {{
            let block = coop::this_block();
            let block_index = block.rank_at::<coop::Grid>() as usize;
            let rank = thread::threadIdx_x() as usize;
            let offset = block_index * $b * $n;
            let mut values = coop::ThreadData::<_, $n, $align>::new();
            if $invalid != 0 {
                let mut i = 0;
                while i < $n {
                    if !($invalid == 6 && rank == $b - 1 && i == $n - 1) {
                        values.write(i, $input[offset + rank * $n + i]);
                    }
                    i += 1;
                }
                let _ = counts(block, &values, $options);
                if $manual {
                    thread::sync_threads();
                }
                // No result read/assertion may counterfeit the preflight trap.
                // A lost preflight must return to the host's unconditional panic.
                let mut report = coop::ThreadData::<_, 1>::new();
                report.write(0, 99_u32);
                coop::store(
                    block,
                    &mut $reports,
                    &report,
                    coop::StoreOptions::default().offset(block_index * $b),
                );
                return;
            }
            if $filled {
                let valid = ($input.len() - offset).min($b * $n);
                coop::load(
                    block,
                    $input,
                    &mut values,
                    coop::LoadOptions::default()
                        .offset(offset)
                        .valid_items(valid)
                        .oob_default($replacement),
                );
            } else {
                coop::load(
                    block,
                    $input,
                    &mut values,
                    coop::LoadOptions::default().offset(offset),
                );
            }
            let mut changed = values;
            changed.write(0, $replacement);
            let result_offset = block_index * $b * $k;
            let mut first = counts(block, &values, $options);
            let mut word =
                observe::<_, $k, $b>(&first, $expected, result_offset + rank, None) as u32;
            first.write(0, $marker);
            if $manual {
                thread::sync_threads();
            }
            let second = counts(block, &changed, $options);
            word |= (observe::<_, $k, $b>(
                &second,
                $expected,
                $expected.len() / 2 + result_offset + rank,
                None,
            ) as u32)
                << 1;
            word |= (observe::<_, $k, $b>(&first, $expected, result_offset + rank, Some($marker))
                as u32)
                << 2;
            if $manual {
                thread::sync_threads();
            }
            coop::store(
                block,
                &mut $output,
                &second,
                coop::StoreOptions::default()
                    .algorithm(coop::Striped)
                    .offset(result_offset),
            );
            coop::store(
                block,
                &mut $preserved,
                &values,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $changed_output,
                &changed,
                coop::StoreOptions::default().offset(offset),
            );
            let mut report = coop::ThreadData::<_, 1>::new();
            report.write(0, word);
            coop::store(
                block,
                &mut $reports,
                &report,
                coop::StoreOptions::default().offset(block_index * $b),
            );
        }};
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn default_u8(
        input: &[u8],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<u8, coop::CollectiveTiles<1>>,
        mut changed_output: coop::DisjointSlice<u8, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        exercise!(
            1,
            1,
            64,
            1,
            coop::HistogramOptions::<33>::default(),
            false,
            true,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_u8,
            101_i32,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn sort_i32(
        input: &[i32],
        expected: &[u32],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<2>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        exercise!(
            4,
            16,
            33,
            2,
            coop::HistogramOptions::<35>::default()
                .counter_type::<u32>()
                .bins_per_thread::<2>()
                .algorithm(coop::Sort),
            false,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_i32,
            169_u32,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn atomic_u32(
        input: &[u32],
        expected: &[i64],
        mut output: coop::DisjointSlice<i64, coop::CollectiveTiles<2>>,
        mut preserved: coop::DisjointSlice<u32, coop::CollectiveTiles<9>>,
        mut changed_output: coop::DisjointSlice<u32, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        exercise!(
            9,
            1,
            32,
            2,
            coop::HistogramOptions::<17>::default()
                .counter_type::<i64>()
                .bins_per_thread::<2>(),
            false,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_u32,
            325_i64,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn sort_i64(
        input: &[i64],
        expected: &[u64],
        mut output: coop::DisjointSlice<u64, coop::CollectiveTiles<3>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<i64, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        exercise!(
            4,
            1,
            32,
            3,
            coop::HistogramOptions::<65>::default()
                .counter_type::<u64>()
                .bins_per_thread::<3>()
                .algorithm(coop::Sort),
            false,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_i64,
            165_u64,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_u64(
        input: &[u64],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut preserved: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut changed_output: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        exercise!(
            9,
            1,
            1,
            9,
            coop::HistogramOptions::<5>::default().bins_per_thread::<9>(),
            false,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_u64,
            46_i32,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn manual_i32(
        input: &[i32],
        expected: &[u64],
        mut output: coop::DisjointSlice<u64, coop::CollectiveTiles<2>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new();
        exercise!(
            4,
            1,
            33,
            2,
            coop::HistogramOptions::<35>::default()
                .counter_type::<u64>()
                .bins_per_thread::<2>()
                .temp_storage(&mut scratch),
            true,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_i32,
            169_u64,
            invalid
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn auto_i32(
        input: &[i32],
        expected: &[u64],
        mut output: coop::DisjointSlice<u64, coop::CollectiveTiles<2>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        exercise!(
            4,
            1,
            33,
            2,
            coop::HistogramOptions::<35>::default()
                .counter_type::<u64>()
                .bins_per_thread::<2>()
                .algorithm(coop::Sort)
                .temp_storage(&mut scratch),
            false,
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            0_i32,
            169_u64,
            invalid
        );
    }
}

// Count each complete initialized tile independently of the device's stripe
// distribution. Padding is part of the expected output, not omitted capacity.
fn oracle(input: &[usize], threads: usize, items: usize, bins: usize, k: usize) -> Vec<u64> {
    let mut expected = vec![0_u64; input.len() / (threads * items) * threads * k];
    for (block, tile) in input.chunks_exact(threads * items).enumerate() {
        for &sample in tile {
            assert!(sample < bins);
            expected[block * threads * k + sample] += 1;
        }
    }
    expected
}

fn samples(threads: usize, items: usize, bins: usize) -> Vec<usize> {
    let tile = threads * items;
    (0..3 * tile)
        .map(|index| match index / tile {
            0 => bins - 1,
            1 => {
                if index % 2 == 0 {
                    0
                } else {
                    bins - 1
                }
            }
            _ => (7 * (index % tile) + 3) % bins,
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("sample-at-bins") => 1,
        Some("negative-i32") => 2,
        Some("negative-i64") => 3,
        Some("wide-u32") => 4,
        Some("wide-u64") => 5,
        Some("missing-input") => 6,
        Some(other) => panic!("unknown Histogram mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut counters, mut values, mut sources, mut reports) = (0, 0, 0, 0, 0);
    macro_rules! run {
        ($t:ty, $c:ty, $n:literal, $b:literal, $bins:literal, $k:literal,
         $prepare:ident, $launch:ident, $blocks:expr, $input:expr) => {{
            let blocks = $blocks;
            let input: Vec<$t> = $input;
            let mut initialized = input.clone();
            initialized.resize(blocks * $b * $n, 0);
            let mut changed = initialized.clone();
            for value in changed.iter_mut().step_by($n) {
                *value = 0;
            }
            let count_slots = blocks * $b * $k;
            let (first_expected, second_expected): (Vec<$c>, Vec<$c>) = if invalid == 0 {
                let first: Vec<usize> = initialized.iter().map(|&v| v as usize).collect();
                let next: Vec<usize> = changed.iter().map(|&v| v as usize).collect();
                (
                    oracle(&first, $b, $n, $bins, $k)
                        .into_iter()
                        .map(|v| v as $c)
                        .collect(),
                    oracle(&next, $b, $n, $bins, $k)
                        .into_iter()
                        .map(|v| v as $c)
                        .collect(),
                )
            } else {
                (vec![0; count_slots], vec![0; count_slots])
            };
            let expected: Vec<$c> = first_expected
                .iter()
                .chain(&second_expected)
                .copied()
                .collect();
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let expected_device = DeviceBuffer::from_host(&stream, &expected)?;
            let output = DeviceBuffer::from_host(&stream, &vec![<$c>::MAX; count_slots])?;
            let preserved = DeviceBuffer::from_host(&stream, &vec![<$t>::MAX; initialized.len()])?;
            let changed_output = DeviceBuffer::from_host(&stream, &vec![<$t>::MAX; changed.len()])?;
            let words = DeviceBuffer::from_host(&stream, &vec![0_u32; blocks * $b])?;
            let plan = module.$prepare(LaunchConfig1D::auto_shared(blocks as u32, $b))?;
            let (source, _, output, preserved, changed_output, words) = module
                .$launch(
                    &plan,
                    source,
                    expected_device,
                    output,
                    preserved,
                    changed_output,
                    words,
                    invalid,
                )
                .sync()?;
            stream.synchronize()?;
            if invalid != 0 {
                panic!("invalid Histogram mode {invalid} unexpectedly completed");
            }
            assert_eq!(
                output.to_host_vec(&stream)?,
                second_expected,
                stringify!($prepare)
            );
            assert_eq!(
                preserved.to_host_vec(&stream)?,
                initialized,
                stringify!($prepare)
            );
            assert_eq!(
                changed_output.to_host_vec(&stream)?,
                changed,
                stringify!($prepare)
            );
            assert_eq!(source.to_host_vec(&stream)?, input, stringify!($prepare));
            assert_eq!(
                words.to_host_vec(&stream)?,
                vec![CHECKS; blocks * $b],
                stringify!($prepare)
            );
            cases += 1;
            counters += 2 * count_slots;
            values += initialized.len() + changed.len();
            sources += input.len();
            reports += blocks * $b;
        }};
    }
    match invalid {
        1 => {
            let mut v = vec![0_u8; 64];
            v[63] = 33;
            run!(
                u8,
                i32,
                1,
                64,
                33,
                1,
                prepare_default_u8,
                default_u8_async_owned,
                1,
                v
            );
        }
        2 => {
            let mut v = vec![0_i32; 132];
            v[131] = -1;
            run!(
                i32,
                u32,
                4,
                33,
                35,
                2,
                prepare_sort_i32,
                sort_i32_async_owned,
                1,
                v
            );
        }
        3 => {
            let mut v = vec![0_i64; 128];
            v[127] = i64::MIN;
            run!(
                i64,
                u64,
                4,
                32,
                65,
                3,
                prepare_sort_i64,
                sort_i64_async_owned,
                1,
                v
            );
        }
        4 => {
            let mut v = vec![0_u32; 288];
            v[287] = u32::MAX;
            run!(
                u32,
                i64,
                9,
                32,
                17,
                2,
                prepare_atomic_u32,
                atomic_u32_async_owned,
                1,
                v
            );
        }
        5 => {
            let mut v = vec![0_u64; 9];
            v[8] = (1_u64 << 32) + 1;
            run!(
                u64,
                i32,
                9,
                1,
                5,
                9,
                prepare_single_u64,
                single_u64_async_owned,
                1,
                v
            );
        }
        6 => {
            run!(
                u32,
                i64,
                9,
                32,
                17,
                2,
                prepare_atomic_u32,
                atomic_u32_async_owned,
                1,
                vec![0_u32; 288]
            );
        }
        _ => {}
    }
    let mut short: Vec<u8> = samples(64, 1, 33).into_iter().map(|v| v as u8).collect();
    short.truncate(2 * 64 + 45);
    run!(
        u8,
        i32,
        1,
        64,
        33,
        1,
        prepare_default_u8,
        default_u8_async_owned,
        3,
        short
    );
    run!(
        u8,
        i32,
        1,
        64,
        33,
        1,
        prepare_default_u8,
        default_u8_async_owned,
        1,
        Vec::<u8>::new()
    );
    run!(
        i32,
        u32,
        4,
        33,
        35,
        2,
        prepare_sort_i32,
        sort_i32_async_owned,
        3,
        samples(33, 4, 35).into_iter().map(|v| v as i32).collect()
    );
    run!(
        u32,
        i64,
        9,
        32,
        17,
        2,
        prepare_atomic_u32,
        atomic_u32_async_owned,
        3,
        samples(32, 9, 17).into_iter().map(|v| v as u32).collect()
    );
    run!(
        i64,
        u64,
        4,
        32,
        65,
        3,
        prepare_sort_i64,
        sort_i64_async_owned,
        3,
        samples(32, 4, 65).into_iter().map(|v| v as i64).collect()
    );
    run!(
        u64,
        i32,
        9,
        1,
        5,
        9,
        prepare_single_u64,
        single_u64_async_owned,
        3,
        samples(1, 9, 5).into_iter().map(|v| v as u64).collect()
    );
    run!(
        i32,
        u64,
        4,
        33,
        35,
        2,
        prepare_manual_i32,
        manual_i32_async_owned,
        3,
        samples(33, 4, 35).into_iter().map(|v| v as i32).collect()
    );
    run!(
        i32,
        u64,
        4,
        33,
        35,
        2,
        prepare_auto_i32,
        auto_i32_async_owned,
        3,
        samples(33, 4, 35).into_iter().map(|v| v as i32).collect()
    );
    assert_eq!(cases, 8);
    println!(
        "histogram: {cases} cases, {counters} counters, {values} preserved values, {sources} source values, {reports} lane reports passed"
    );
    Ok(())
}
