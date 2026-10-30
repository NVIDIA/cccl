// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent outputs expose association, validity and both surviving pairs.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:literal, $ka:literal, $va:literal, $b:literal, $group:expr,
         $first_options:expr, $second_options:expr, $manual:literal,
         $keys:ident, $values:ident, $first_keys:ident, $first_values:ident,
         $second_keys:ident, $second_values:ident, $keep_keys:ident,
         $keep_values:ident, $changed_keys:ident, $changed_values:ident,
         $validity:ident) => {{
            let block = coop::this_block();
            let offset = block.rank_at::<coop::Grid>() as usize * $b * $n;
            let mut keys = coop::ThreadData::<_, $n, $ka>::new();
            let mut values = coop::ThreadData::<_, $n, $va>::new();
            coop::load(
                block,
                $keys,
                &mut keys,
                coop::LoadOptions::default().offset(offset),
            );
            coop::load(
                block,
                $values,
                &mut values,
                coop::LoadOptions::default().offset(offset),
            );
            let (mut first_k, mut first_v) =
                coop::merge_sort_pairs($group, &keys, &values, $first_options);
            if $manual {
                thread::sync_threads();
            }
            coop::store(
                block,
                &mut $first_keys,
                &first_k,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $first_values,
                &first_v,
                coop::StoreOptions::default().offset(offset),
            );
            // Retain both fresh owners, including their mutations, over reuse.
            first_k.write(0, 111 as _);
            first_v.write(0, -7 as _);
            let (second_k, second_v) =
                coop::merge_sort_pairs($group, &keys, &values, $second_options);
            if $manual {
                thread::sync_threads();
            }
            let mut visible_k = coop::ThreadData::<_, $n>::new();
            let mut visible_v = coop::ThreadData::<_, $n>::new();
            let mut flags = coop::ThreadData::<_, $n>::new();
            for item in 0..$n {
                visible_k.write(item, second_k.get(item).unwrap_or(99 as _));
                visible_v.write(item, second_v.get(item).unwrap_or(99 as _));
                flags.write(
                    item,
                    i32::from(keys.get(item).is_some())
                        + 2 * i32::from(values.get(item).is_some())
                        + 4 * i32::from(second_k.get(item).is_some())
                        + 8 * i32::from(second_v.get(item).is_some()),
                );
            }
            coop::store(
                block,
                &mut $second_keys,
                &visible_k,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $second_values,
                &visible_v,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $keep_keys,
                &keys,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $keep_values,
                &values,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $changed_keys,
                &first_k,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $changed_values,
                &first_v,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $validity,
                &flags,
                coop::StoreOptions::default().offset(offset),
            );
        }};
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn f32_i8_warp(
        keys: &[f32],
        values: &[i8],
        mut first_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut first_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut second_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut second_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut keep_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut keep_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut changed_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut changed_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();

        if invalid != 0 {
            let rank = thread::threadIdx_x() as usize;
            let mut input_keys = coop::ThreadData::<_, 4, 32>::new();
            let mut input_values = coop::ThreadData::<_, 4, 1>::new();
            for item in 0..4 {
                input_keys.write(item, keys[rank * 4 + item]);
                if !(rank % 8 == 7 && item == 3) {
                    input_values.write(item, values[rank * 4 + item]);
                }
            }
            let _ = coop::merge_sort_pairs(
                coop::this_warp().group_by::<8>(),
                &input_keys,
                &input_values,
                coop::MergeSortOptions::default(),
            );
            thread::sync_threads();
            // Never read the invalid result. A lost preflight returns an
            // independent report; the host then reports unexpected success.
            let mut report = coop::ThreadData::<_, 4>::new();
            for item in 0..4 {
                report.write(item, 99_f32);
            }
            coop::store(block, &mut changed_keys, &report, Default::default());
            return;
        }
        exercise!(
            4,
            32,
            1,
            64,
            coop::this_warp().group_by::<8>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, f32::MIN)
                .order(coop::Descending),
            false,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            keep_keys,
            keep_values,
            changed_keys,
            changed_values,
            validity
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn i64_f64_manual(
        keys: &[i64],
        values: &[f64],
        mut first_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut first_values: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut second_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut second_values: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut keep_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut keep_values: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut changed_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut changed_values: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let mut scratch = coop::TempStorage::new();
        if invalid != 0 {
            let rank = thread::threadIdx_x() as usize;
            let mut input_keys = coop::ThreadData::<_, 9, 1>::new();
            let mut input_values = coop::ThreadData::<_, 9, 32>::new();
            for item in 0..9 {
                input_keys.write(item, keys[rank * 9 + item]);
                if !(rank == 63 && item == 8) {
                    input_values.write(item, values[rank * 9 + item]);
                }
            }
            let _ = coop::merge_sort_pairs(
                block,
                &input_keys,
                &input_values,
                coop::MergeSortOptions::default()
                    .order(coop::Descending)
                    .valid_items(count, i64::MIN)
                    .temp_storage(&mut scratch),
            );
            thread::sync_threads();
            // Never read the invalid result. A lost preflight returns an
            // independent report; the host then reports unexpected success.
            let mut report = coop::ThreadData::<_, 9>::new();
            for item in 0..9 {
                report.write(item, 99_i64);
            }
            coop::store(block, &mut changed_keys, &report, Default::default());
            return;
        }
        exercise!(
            9,
            1,
            32,
            64,
            block,
            coop::MergeSortOptions::default().temp_storage(&mut scratch),
            coop::MergeSortOptions::default()
                .order(coop::Descending)
                .valid_items(count, i64::MIN)
                .temp_storage(&mut scratch),
            true,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            keep_keys,
            keep_values,
            changed_keys,
            changed_values,
            validity
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn u32_i16_auto(
        keys: &[u32],
        values: &[i16],
        mut first_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut first_values: coop::DisjointSlice<i16, coop::CollectiveTiles<1>>,
        mut second_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut second_values: coop::DisjointSlice<i16, coop::CollectiveTiles<1>>,
        mut keep_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut keep_values: coop::DisjointSlice<i16, coop::CollectiveTiles<1>>,
        mut changed_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mut changed_values: coop::DisjointSlice<i16, coop::CollectiveTiles<1>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let mut scratch = coop::TempStorage::new().auto_sync();
        let _ = invalid;
        exercise!(
            1,
            32,
            16,
            32,
            block,
            coop::MergeSortOptions::default().temp_storage(&mut scratch),
            coop::MergeSortOptions::default()
                .valid_items(count, u32::MIN)
                .order(coop::Descending)
                .temp_storage(&mut scratch),
            false,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            keep_keys,
            keep_values,
            changed_keys,
            changed_values,
            validity
        );
    }
}

// Keys are ordered, but equal keys may permute. Bitwise pair multisets check
// association and preserve value NaN payloads/signed zeros without a tie rule.
fn check_pairs<K: Copy + PartialOrd, V: Copy>(
    keys: &[K],
    values: &[V],
    actual_keys: &[K],
    actual_values: &[V],
    descending: bool,
    key_bits: impl Fn(K) -> u64,
    value_bits: impl Fn(V) -> u64,
) {
    assert!(actual_keys.windows(2).all(|w| if descending {
        w[0] >= w[1]
    } else {
        w[0] <= w[1]
    }));
    let pairs = |k: &[K], v: &[V]| {
        let mut result: Vec<_> = k
            .iter()
            .copied()
            .zip(v.iter().copied())
            .map(|(k, v)| (key_bits(k), value_bits(v)))
            .collect();
        result.sort_unstable();
        result
    };
    assert_eq!(pairs(actual_keys, actual_values), pairs(keys, values));
}

fn f64_values(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| match i % 8 {
            0 => f64::from_bits(0x7ff8_0000_0000_0000 | (i as u64 + 1)),
            1 => f64::from_bits(0xfff8_0000_0000_0000 | (i as u64 + 1)),
            2 => 0.0,
            3 => -0.0,
            4 => f64::INFINITY,
            5 => f64::NEG_INFINITY,
            _ => i as f64 + 0.25,
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("missing-value-zero") => 1,
        Some("missing-value-warp") => 2,
        Some(other) => panic!("unknown Merge Sort Pairs mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut sorted, mut preserved, mut sources, mut flags_checked, mut mutated) =
        (0, 0, 0, 0, 0, 0);
    macro_rules! run {
        ($kt:ty, $vt:ty, $n:literal, $b:literal, $group:literal, $blocks:expr,
         $prepare:ident, $launch:ident, $keys:expr, $values:expr, $count:expr,
         $key_bits:expr, $value_bits:expr) => {{
            let keys: Vec<$kt> = $keys;
            let values: Vec<$vt> = $values;
            let count: usize = $count;
            let group_items = $group * $n;
            let key_bits = $key_bits;
            let value_bits = $value_bits;
            let kb = |v: &[$kt]| v.iter().copied().map(key_bits).collect::<Vec<_>>();
            let vb = |v: &[$vt]| v.iter().copied().map(value_bits).collect::<Vec<_>>();
            let source_k = DeviceBuffer::from_host(&stream, &keys)?;
            let source_v = DeviceBuffer::from_host(&stream, &values)?;
            let first_k = DeviceBuffer::from_host(&stream, &vec![17 as $kt; keys.len()])?;
            let first_v = DeviceBuffer::from_host(&stream, &vec![17 as $vt; keys.len()])?;
            let second_k = DeviceBuffer::from_host(&stream, &vec![17 as $kt; keys.len()])?;
            let second_v = DeviceBuffer::from_host(&stream, &vec![17 as $vt; keys.len()])?;
            let keep_k = DeviceBuffer::from_host(&stream, &vec![17 as $kt; keys.len()])?;
            let keep_v = DeviceBuffer::from_host(&stream, &vec![17 as $vt; keys.len()])?;
            let changed_k = DeviceBuffer::from_host(&stream, &vec![17 as $kt; keys.len()])?;
            let changed_v = DeviceBuffer::from_host(&stream, &vec![17 as $vt; keys.len()])?;
            let flags = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            stream.synchronize()?;
            let plan = module.$prepare(LaunchConfig1D::auto_shared($blocks, $b))?;
            let (
                source_k,
                source_v,
                first_k,
                first_v,
                second_k,
                second_v,
                keep_k,
                keep_v,
                changed_k,
                changed_v,
                flags,
            ) = module
                .$launch(
                    &plan,
                    source_k,
                    source_v,
                    first_k,
                    first_v,
                    second_k,
                    second_v,
                    keep_k,
                    keep_v,
                    changed_k,
                    changed_v,
                    flags,
                    count as i64,
                    invalid,
                )
                .sync()?;
            stream.synchronize()?;
            if invalid != 0 {
                panic!("invalid Merge Sort Pairs mode {invalid} unexpectedly completed");
            }
            let first_k = first_k.to_host_vec(&stream)?;
            let first_v = first_v.to_host_vec(&stream)?;
            let second_k = second_k.to_host_vec(&stream)?;
            let second_v = second_v.to_host_vec(&stream)?;
            let flags = flags.to_host_vec(&stream)?;
            for base in (0..keys.len()).step_by(group_items) {
                let end = base + group_items;
                let prefix = base + count;
                check_pairs(
                    &keys[base..end],
                    &values[base..end],
                    &first_k[base..end],
                    &first_v[base..end],
                    false,
                    key_bits,
                    value_bits,
                );
                check_pairs(
                    &keys[base..prefix],
                    &values[base..prefix],
                    &second_k[base..prefix],
                    &second_v[base..prefix],
                    true,
                    key_bits,
                    value_bits,
                );
                assert!(
                    second_k[prefix..end]
                        .iter()
                        .all(|&v| key_bits(v) == key_bits(99 as $kt))
                );
                assert!(
                    second_v[prefix..end]
                        .iter()
                        .all(|&v| value_bits(v) == value_bits(99 as $vt))
                );
                assert!(flags[base..prefix].iter().all(|&v| v == 15));
                assert!(flags[prefix..end].iter().all(|&v| v == 3));
            }
            assert_eq!(kb(&keep_k.to_host_vec(&stream)?), kb(&keys));
            assert_eq!(vb(&keep_v.to_host_vec(&stream)?), vb(&values));
            assert_eq!(kb(&source_k.to_host_vec(&stream)?), kb(&keys));
            assert_eq!(vb(&source_v.to_host_vec(&stream)?), vb(&values));
            let mut expected_k = first_k;
            let mut expected_v = first_v;
            for i in (0..keys.len()).step_by($n) {
                expected_k[i] = 111 as $kt;
                expected_v[i] = -7 as $vt;
            }
            assert_eq!(kb(&changed_k.to_host_vec(&stream)?), kb(&expected_k));
            assert_eq!(vb(&changed_v.to_host_vec(&stream)?), vb(&expected_v));
            cases += 1;
            sorted += 2 * keys.len() + 2 * keys.len() / group_items * count;
            preserved += 2 * keys.len();
            sources += 2 * keys.len();
            flags_checked += keys.len();
            mutated += 2 * keys.len();
        }};
    }
    let blocks = if invalid == 0 { 2 } else { 1 };
    if invalid == 0 || invalid == 2 {
        let len = blocks as usize * 64 * 4;
        run!(
            f32,
            i8,
            4,
            64,
            8,
            blocks,
            prepare_f32_i8_warp,
            f32_i8_warp_async_owned,
            (0..len)
                .map(|i| match i % 13 {
                    0 => -0.0,
                    1 => 0.0,
                    _ => ((i * 7) % 11) as f32 - 5.0,
                })
                .collect(),
            (0..len).map(|i| ((i * 17) % 101) as i8 - 50).collect(),
            31,
            |v: f32| v.to_bits() as u64,
            |v: i8| v as u8 as u64
        );
    }
    if invalid == 0 || invalid == 1 {
        let len = blocks as usize * 64 * 9;
        run!(
            i64,
            f64,
            9,
            64,
            64,
            blocks,
            prepare_i64_f64_manual,
            i64_f64_manual_async_owned,
            (0..len).map(|i| ((i * 11) % 23) as i64 - 11).collect(),
            f64_values(len),
            if invalid == 1 { 0 } else { 575 },
            |v: i64| v as u64,
            f64::to_bits
        );
    }
    if invalid == 0 {
        for count in [0, 32] {
            run!(
                u32,
                i16,
                1,
                32,
                32,
                2,
                prepare_u32_i16_auto,
                u32_i16_auto_async_owned,
                (0..64).map(|i| 1 + (i * 5 % 13) as u32).collect(),
                (0..64).map(|i| i as i16 - 32).collect(),
                count,
                |v: u32| v as u64,
                |v: i16| v as u16 as u64
            );
        }
    }
    assert_eq!(
        (cases, sorted, preserved, sources, flags_checked, mutated),
        (4, 7004, 3584, 3584, 1792, 3584)
    );
    println!(
        "merge_sort_pairs: {cases} cases, {sorted} sorted key/value observations, {preserved} preserved values, {sources} source values, {flags_checked} validity flags, {mutated} retained mutated values passed"
    );
    Ok(())
}
