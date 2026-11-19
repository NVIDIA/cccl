// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const CHECKS: u32 = 15;

#[cuda_module]
mod kernels {
    use super::*;

    fn same<T: coop::RadixKey + PartialEq, const N: usize, const A: usize>(
        left: &coop::ThreadData<T, N, A>,
        right: &coop::ThreadData<T, N, A>,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut equal = true;
        let mut i = 0;
        while i < N {
            equal &= left.get(i) == right.get(i);
            i += 1;
        }
        equal
    }

    fn observe<T: coop::RadixKey + PartialEq, const N: usize, const A: usize>(
        result: &coop::ThreadData<T, N, A>,
        expected: &[T],
        offset: usize,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut valid = true;
        let mut i = 0;
        while i < N {
            valid &= result.get(i) == Some(expected[offset + i]);
            i += 1;
        }
        valid
    }

    // Expressions retain the concrete bound and storage types of each root.
    macro_rules! exercise {
        ($n:literal, $align:literal, $b:literal, $rank:expr,
         $first_options:expr, $second_options:expr, $manual:literal,
         $input:ident, $expected:ident, $output:ident, $preserved:ident,
         $changed_output:ident, $reports:ident, $replacement:expr, $marker:expr) => {{
            let block = coop::this_block();
            let block_index = block.rank_at::<coop::Grid>() as usize;
            let rank = $rank as usize;
            let offset = block_index * $b * $n;
            let mut values = coop::ThreadData::<_, $n, $align>::new();
            coop::load(
                block,
                $input,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
            let before = values;
            let mut first = coop::radix_sort_keys(block, &values, $first_options);
            let first_matches = observe(&first, $expected, offset + rank * $n);
            let mut inputs_preserved = same(&values, &before);
            first.write(0, $marker);
            let first_modified = first;
            let mut changed = values;
            changed.write(0, $replacement);
            let changed_before = changed;
            if $manual {
                thread::sync_threads();
            }
            let second = coop::radix_sort_keys(block, &changed, $second_options);
            let second_matches =
                observe(&second, $expected, $expected.len() / 2 + offset + rank * $n);
            inputs_preserved &= same(&values, &before) && same(&changed, &changed_before);
            let independent = same(&first, &first_modified) && first.get(0) == Some($marker);
            if $manual {
                thread::sync_threads();
            }
            coop::store(
                block,
                &mut $output,
                &second,
                coop::StoreOptions::default().offset(offset),
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
            let word = first_matches as u32
                | ((second_matches as u32) << 1)
                | ((inputs_preserved as u32) << 2)
                | ((independent as u32) << 3);
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
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn i32_nonwarp(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut changed_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        end: i64,
    ) {
        exercise!(
            1,
            1,
            33,
            thread::threadIdx_x(),
            coop::RadixSortOptions::default(),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .begin_bit_const::<29>()
                .end_bit(end),
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            -7_i32,
            0x1357_9bdf_i32
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 3, block = (3, 5, 2), dynamic_shared = inferred)]
    pub fn u32_xyz(
        input: &[u32],
        expected: &[u32],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        begin: i64,
        end: i64,
    ) {
        exercise!(
            4,
            16,
            30,
            thread::threadIdx_x() + 3 * (thread::threadIdx_y() + 5 * thread::threadIdx_z()),
            coop::RadixSortOptions::default()
                .begin_bit_const::<4>()
                .end_bit_const::<7>(),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .begin_bit(begin)
                .end_bit(end),
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            7_u32,
            0x1357_9bdf_u32
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn i64_manual(
        input: &[i64],
        expected: &[i64],
        mut output: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut changed_output: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        begin: i64,
        end: i64,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new();
        if invalid != 0 {
            let block = coop::this_block();
            let rank = thread::threadIdx_x() as usize;
            let mut values = coop::ThreadData::<_, 9, 16>::new();
            let mut i = 0;
            while i < 9 {
                if !(invalid == 1 && rank == 63 && i == 8) {
                    values.write(i, input[rank * 9 + i]);
                }
                i += 1;
            }
            let _ = coop::radix_sort_keys(
                block,
                &values,
                coop::RadixSortOptions::default()
                    .begin_bit(begin)
                    .end_bit(end)
                    .temp_storage(&mut scratch),
            );
            thread::sync_threads();
            // A lost preflight must complete normally and fail on the host.
            // No result read or assertion may counterfeit the expected trap.
            let mut report = coop::ThreadData::<_, 1>::new();
            report.write(0, 99_u32);
            coop::store(block, &mut reports, &report, Default::default());
            return;
        }
        exercise!(
            9,
            16,
            64,
            thread::threadIdx_x(),
            coop::RadixSortOptions::default()
                .begin_bit(begin)
                .end_bit_const::<64>()
                .temp_storage(&mut scratch),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .begin_bit(begin)
                .end_bit(end)
                .temp_storage(&mut scratch),
            true,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            -7_i64,
            0x1357_9bdf_2468_ace0_i64
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn u64_auto(
        input: &[u64],
        expected: &[u64],
        mut output: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut changed_output: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        exercise!(
            4,
            1,
            33,
            thread::threadIdx_x(),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .temp_storage(&mut scratch),
            coop::RadixSortOptions::default()
                .begin_bit_const::<0>()
                .end_bit_const::<3>()
                .temp_storage(&mut scratch),
            false,
            input,
            expected,
            output,
            preserved,
            changed_output,
            reports,
            7_u64,
            0x1357_9bdf_2468_ace0_u64
        );
    }
}

// Sort original keys by selected ordered digits. Stable descending compares
// digits in reverse; reversing an ascending result would reverse equal digits.
fn oracle<T: Copy>(
    input: &[T],
    tile: usize,
    begin: u32,
    end: u32,
    descending: bool,
    ordered: impl Fn(T) -> u128,
) -> Vec<T> {
    let mask = (1_u128 << (end - begin)) - 1;
    let digit = |key| (ordered(key) >> begin) & mask;
    let mut result = input.to_vec();
    for block in result.chunks_exact_mut(tile) {
        block.sort_by(|&left, &right| {
            if descending {
                digit(right).cmp(&digit(left))
            } else {
                digit(left).cmp(&digit(right))
            }
        });
    }
    result
}

fn signed_keys(width: u32, tile: usize, blocks: usize) -> Vec<i64> {
    (0..blocks * tile)
        .map(|index| {
            let j = index % tile;
            let b = index / tile;
            let digit = ((5 * j + 3 * b) % 8) as u64;
            let bits = ((digit << (width - 3)) | (index as u64 + 1)) ^ (1_u64 << (width - 1));
            let key = if width == 32 {
                bits as i32 as i64
            } else {
                bits as i64
            };
            match j {
                0 => {
                    if width == 32 {
                        i32::MIN as i64
                    } else {
                        i64::MIN
                    }
                }
                1 => {
                    if width == 32 {
                        i32::MAX as i64
                    } else {
                        i64::MAX
                    }
                }
                2 => -1,
                3 => 0,
                _ => key,
            }
        })
        .collect()
}

fn unsigned_keys(width: u32, tile: usize, blocks: usize) -> Vec<u64> {
    (0..blocks * tile)
        .map(|index| {
            let j = index % tile;
            let b = index / tile;
            let digit = ((5 * j + 3 * b) % 8) as u64;
            let tag = index as u64 + 1;
            match j {
                0 => 0,
                1 => {
                    if width == 32 {
                        u32::MAX as u64
                    } else {
                        u64::MAX
                    }
                }
                _ if width == 32 => (tag << 7) | (digit << 4) | (tag & 15),
                _ => (((j & 1) as u64) << 63) | (tag << 3) | digit,
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    let (invalid, begin, end) = match std::env::args().nth(1).as_deref() {
        None => (0, 61_i64, 64_i64),
        Some("missing-input") => (1, 61, 64),
        Some("negative-begin") => (2, i64::MIN, 64),
        Some("empty") => (3, 61, 61),
        Some("reversed") => (4, 62, 61),
        Some("end-beyond-width") => (5, 0, 65),
        Some("wide-end") => (6, 0, (1_i64 << 32) + 64),
        Some(other) => panic!("unknown Radix Sort Keys mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut sorted, mut values, mut sources, mut reports) = (0, 0, 0, 0, 0);
    macro_rules! run {
        ($t:ty, $n:literal, $b:literal, $prepare:ident, $launch:ident,
         $config:expr, $input:expr, $replacement:expr, $first:expr, $second:expr,
         $ordered:expr $(, $argument:expr)*) => {{
            let input: Vec<$t> = $input;
            let blocks = input.len() / ($b * $n);
            let mut changed = input.clone();
            for value in changed.iter_mut().step_by($n) { *value = $replacement; }
            let (b1, e1, d1) = $first;
            let (b2, e2, d2) = $second;
            let first_expected = oracle(&input, $b * $n, b1, e1, d1, $ordered);
            let second_expected = oracle(&changed, $b * $n, b2, e2, d2, $ordered);
            let expected: Vec<$t> = first_expected.iter().chain(&second_expected).copied().collect();
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let expected_device = DeviceBuffer::from_host(&stream, &expected)?;
            let output = DeviceBuffer::from_host(&stream, &vec![<$t>::MAX; input.len()])?;
            let preserved = DeviceBuffer::from_host(&stream, &vec![<$t>::MAX; input.len()])?;
            let changed_output = DeviceBuffer::from_host(&stream, &vec![<$t>::MAX; input.len()])?;
            let words = DeviceBuffer::from_host(&stream, &vec![0_u32; blocks * $b])?;
            stream.synchronize()?;
            let plan = module.$prepare($config)?;
            let (source, _, output, preserved, changed_output, words) = module.$launch(
                &plan, source, expected_device, output, preserved, changed_output, words
                $(, $argument)*).sync()?;
            stream.synchronize()?;
            if invalid != 0 { panic!("invalid Radix Sort Keys mode {invalid} unexpectedly completed"); }
            assert_eq!(output.to_host_vec(&stream)?, second_expected, stringify!($prepare));
            assert_eq!(preserved.to_host_vec(&stream)?, input, stringify!($prepare));
            assert_eq!(changed_output.to_host_vec(&stream)?, changed, stringify!($prepare));
            assert_eq!(source.to_host_vec(&stream)?, input, stringify!($prepare));
            assert_eq!(words.to_host_vec(&stream)?, vec![CHECKS; blocks * $b], stringify!($prepare));
            cases += 1; sorted += 2 * input.len(); values += 2 * input.len();
            sources += input.len(); reports += blocks * $b;
        }};
    }
    if invalid != 0 {
        run!(
            i64,
            9,
            64,
            prepare_i64_manual,
            i64_manual_async_owned,
            LaunchConfig1D::auto_shared(1, 64),
            signed_keys(64, 576, 1),
            -7,
            (61, 64, false),
            (61, 64, true),
            |v: i64| ((v as u64) ^ (1_u64 << 63)) as u128,
            begin,
            end,
            invalid
        );
    } else {
        run!(
            i32,
            1,
            33,
            prepare_i32_nonwarp,
            i32_nonwarp_async_owned,
            LaunchConfig1D::auto_shared(2, 33),
            signed_keys(32, 33, 2)
                .into_iter()
                .map(|v| v as i32)
                .collect(),
            -7,
            (0, 32, false),
            (29, 32, true),
            |v: i32| ((v as u32) ^ (1_u32 << 31)) as u128,
            32_i64
        );
        run!(
            u32,
            4,
            30,
            prepare_u32_xyz,
            u32_xyz_async_owned,
            LaunchConfig3D::auto_shared((2, 1, 1), (3, 5, 2)),
            unsigned_keys(32, 120, 2)
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            7,
            (4, 7, false),
            (0, 32, true),
            |v: u32| v as u128,
            0_i64,
            32_i64
        );
        run!(
            i64,
            9,
            64,
            prepare_i64_manual,
            i64_manual_async_owned,
            LaunchConfig1D::auto_shared(2, 64),
            signed_keys(64, 576, 2),
            -7,
            (61, 64, false),
            (61, 64, true),
            |v: i64| ((v as u64) ^ (1_u64 << 63)) as u128,
            begin,
            end,
            invalid
        );
        run!(
            u64,
            4,
            33,
            prepare_u64_auto,
            u64_auto_async_owned,
            LaunchConfig1D::auto_shared(2, 33),
            unsigned_keys(64, 132, 2),
            7,
            (0, 64, true),
            (0, 3, false),
            |v: u64| v as u128
        );
    }
    assert_eq!(
        (cases, sorted, values, sources, reports),
        (4, 3444, 3444, 1722, 320)
    );
    println!(
        "radix_sort_keys: {cases} cases, {sorted} sorted keys, {values} preserved values, {sources} source values, {reports} lane reports passed"
    );
    Ok(())
}
