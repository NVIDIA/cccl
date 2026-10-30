// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independent outputs expose association, prefix validity and surviving pairs.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:literal, $ka:literal, $va:literal, $b:literal,
         $first:ident, $second:ident, $first_options:expr, $second_options:expr,
         $manual:literal, $first_count:expr, $second_count:expr,
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
            let (mut first_k, mut first_v) = coop::$first(block, &keys, &values, $first_options);
            if $manual {
                thread::sync_threads();
            }
            coop::store(
                block,
                &mut $first_keys,
                &first_k,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($first_count),
            );
            coop::store(
                block,
                &mut $first_values,
                &first_v,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($first_count),
            );
            // Mutate only an already initialized item; keep invalid tails invalid.
            if (thread::threadIdx_x() as usize) * $n < $first_count {
                first_k.write(0, 111 as _);
                first_v.write(0, 7 as _);
            }
            let (second_k, second_v) = coop::$second(block, &keys, &values, $second_options);
            if $manual {
                thread::sync_threads();
            }
            let mut flags = coop::ThreadData::<_, $n>::new();
            for item in 0..$n {
                flags.write(
                    item,
                    i32::from(keys.get(item).is_some())
                        + 2 * i32::from(values.get(item).is_some())
                        + 4 * i32::from(first_k.get(item).is_some())
                        + 8 * i32::from(first_v.get(item).is_some())
                        + 16 * i32::from(second_k.get(item).is_some())
                        + 32 * i32::from(second_v.get(item).is_some()),
                );
            }
            coop::store(
                block,
                &mut $second_keys,
                &second_k,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($second_count),
            );
            coop::store(
                block,
                &mut $second_values,
                &second_v,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($second_count),
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
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($first_count),
            );
            coop::store(
                block,
                &mut $changed_values,
                &first_v,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items($first_count),
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
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn f32_u64_manual(
        keys: &[f32],
        values: &[u64],
        mut first_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut first_values: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut second_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut second_values: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut keep_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut keep_values: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut changed_keys: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut changed_values: coop::DisjointSlice<u64, coop::CollectiveTiles<9>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        k: i64,
        valid: i64,
        invalid: u32,
    ) {
        let _ = invalid;
        let mut scratch = coop::TempStorage::new();
        let selected = k.min(valid) as usize;
        exercise!(
            9,
            16,
            32,
            33,
            topk_min_pairs,
            topk_max_pairs,
            coop::TopKOptions::default()
                .k_const::<17>()
                .temp_storage(&mut scratch),
            coop::TopKOptions::default()
                .k(k)
                .valid_items(valid)
                .temp_storage(&mut scratch),
            true,
            17,
            selected,
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
    pub fn u64_i8_auto(
        keys: &[u64],
        values: &[i8],
        mut first_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut first_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut second_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut second_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut keep_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut keep_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut changed_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut changed_values: coop::DisjointSlice<i8, coop::CollectiveTiles<4>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        k: i64,
        valid: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let mut scratch = coop::TempStorage::new().auto_sync();
        if invalid != 0 {
            let rank = thread::threadIdx_x() as usize;
            let mut input_k = coop::ThreadData::<_, 4, 32>::new();
            let mut input_v = coop::ThreadData::<_, 4, 1>::new();
            for item in 0..4 {
                if !(invalid == 3 && rank == 63 && item == 3) {
                    input_k.write(item, keys[rank * 4 + item]);
                }
                if !(invalid == 4 && rank == 63 && item == 3) {
                    input_v.write(item, values[rank * 4 + item]);
                }
            }
            if invalid == 3 {
                let _ = coop::topk_min_pairs(
                    block,
                    &input_k,
                    &input_v,
                    coop::TopKOptions::default().k(k).temp_storage(&mut scratch),
                );
            } else {
                let _ = coop::topk_min_pairs(
                    block,
                    &input_k,
                    &input_v,
                    coop::TopKOptions::default()
                        .k(k)
                        .valid_items(valid)
                        .temp_storage(&mut scratch),
                );
            }
            thread::sync_threads();
            // Never inspect an invalid result. Only an independent report is
            // written if native preflight unexpectedly returns.
            let mut report = coop::ThreadData::<_, 4>::new();
            for item in 0..4 {
                report.write(item, 99_u64);
            }
            coop::store(block, &mut changed_keys, &report, Default::default());
            return;
        }
        let selected = k.min(valid) as usize;
        exercise!(
            4,
            32,
            1,
            64,
            topk_max_pairs,
            topk_min_pairs,
            coop::TopKOptions::default()
                .k_const::<256>()
                .temp_storage(&mut scratch),
            coop::TopKOptions::default()
                .valid_items(valid)
                .k(k)
                .temp_storage(&mut scratch),
            false,
            256,
            selected,
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

fn check_selected<K: Copy + PartialOrd, V: Copy>(
    keys: &[K],
    values: &[V],
    actual_k: &[K],
    actual_v: &[V],
    descending: bool,
    key_bits: impl Fn(K) -> u64,
    value_bits: impl Fn(V) -> u64,
) {
    assert_eq!(keys.len(), values.len());
    assert_eq!(actual_k.len(), actual_v.len());
    let compare = |a: &K, b: &K| a.partial_cmp(b).expect("finite ordered keys");
    let mut expected = keys.to_vec();
    expected.sort_by(compare);
    if descending {
        expected.reverse();
    }
    expected.truncate(actual_k.len());
    expected.sort_by(compare);
    let mut selected = actual_k.to_vec();
    selected.sort_by(compare);
    // Numeric equality permits either signed zero at a selection boundary.
    assert!(selected == expected);
    let mut available = std::collections::BTreeMap::new();
    for (&key, &value) in keys.iter().zip(values) {
        *available
            .entry((key_bits(key), value_bits(value)))
            .or_insert(0_usize) += 1;
    }
    for (&key, &value) in actual_k.iter().zip(actual_v) {
        let count = available
            .get_mut(&(key_bits(key), value_bits(value)))
            .expect("selected key/value association belongs to the input");
        assert!(*count > 0, "selected pair multiplicity exceeds input");
        *count -= 1;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("negative-k") => 1,
        Some("wide-valid") => 2,
        Some("missing-key-zero-k") => 3,
        Some("missing-value-zero-valid") => 4,
        Some(other) => panic!("unknown TopK mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (
        mut cases,
        mut selected,
        mut preserved,
        mut sources,
        mut retained,
        mut flags_checked,
        mut suffix,
    ) = (0, 0, 0, 0, 0, 0, 0);
    macro_rules! run {
        ($kt:ty, $vt:ty, $n:literal, $b:literal, $blocks:expr,
         $prepare:ident, $launch:ident, $keys:expr, $values:expr,
         $first_count:expr, $first_desc:literal, $second_desc:literal,
         $k:expr, $valid:expr, $key_bits:expr, $value_bits:expr) => {{
            let keys: Vec<$kt> = $keys;
            let values: Vec<$vt> = $values;
            let k: i64 = $k;
            let valid: i64 = $valid;
            let first_count: usize = $first_count;
            let key_bits = $key_bits;
            let value_bits = $value_bits;
            let kb = |v: &[$kt]| v.iter().copied().map(key_bits).collect::<Vec<_>>();
            let vb = |v: &[$vt]| v.iter().copied().map(value_bits).collect::<Vec<_>>();
            let source_k = DeviceBuffer::from_host(&stream, &keys)?;
            let source_v = DeviceBuffer::from_host(&stream, &values)?;
            let first_k = DeviceBuffer::from_host(&stream, &vec![12345 as $kt; keys.len()])?;
            let first_v = DeviceBuffer::from_host(&stream, &vec![127 as $vt; keys.len()])?;
            let second_k = DeviceBuffer::from_host(&stream, &vec![12345 as $kt; keys.len()])?;
            let second_v = DeviceBuffer::from_host(&stream, &vec![127 as $vt; keys.len()])?;
            let keep_k = DeviceBuffer::from_host(&stream, &vec![12345 as $kt; keys.len()])?;
            let keep_v = DeviceBuffer::from_host(&stream, &vec![127 as $vt; keys.len()])?;
            let changed_k = DeviceBuffer::from_host(&stream, &vec![12345 as $kt; keys.len()])?;
            let changed_v = DeviceBuffer::from_host(&stream, &vec![127 as $vt; keys.len()])?;
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
                    &plan, source_k, source_v, first_k, first_v, second_k, second_v, keep_k,
                    keep_v, changed_k, changed_v, flags, k, valid, invalid,
                )
                .sync()?;
            stream.synchronize()?;
            if invalid != 0 {
                panic!("invalid TopK mode {invalid} unexpectedly completed");
            }
            let second_count = k.min(valid) as usize;
            let first_k = first_k.to_host_vec(&stream)?;
            let first_v = first_v.to_host_vec(&stream)?;
            let second_k = second_k.to_host_vec(&stream)?;
            let second_v = second_v.to_host_vec(&stream)?;
            let flags = flags.to_host_vec(&stream)?;
            for base in (0..keys.len()).step_by($b * $n) {
                let end = base + $b * $n;
                let first_end = base + first_count;
                let second_end = base + second_count;
                check_selected(
                    &keys[base..end],
                    &values[base..end],
                    &first_k[base..first_end],
                    &first_v[base..first_end],
                    $first_desc,
                    key_bits,
                    value_bits,
                );
                check_selected(
                    &keys[base..base + valid as usize],
                    &values[base..base + valid as usize],
                    &second_k[base..second_end],
                    &second_v[base..second_end],
                    $second_desc,
                    key_bits,
                    value_bits,
                );
                for (out_k, out_v, start) in [
                    (&first_k, &first_v, first_end),
                    (&second_k, &second_v, second_end),
                ] {
                    assert!(
                        out_k[start..end]
                            .iter()
                            .all(|&v| key_bits(v) == key_bits(12345 as $kt))
                    );
                    assert!(
                        out_v[start..end]
                            .iter()
                            .all(|&v| value_bits(v) == value_bits(127 as $vt))
                    );
                }
                for i in 0..$b * $n {
                    assert_eq!(
                        flags[base + i],
                        3 + if i < first_count { 12 } else { 0 }
                            + if i < second_count { 48 } else { 0 }
                    );
                }
            }
            assert_eq!(kb(&keep_k.to_host_vec(&stream)?), kb(&keys));
            assert_eq!(vb(&keep_v.to_host_vec(&stream)?), vb(&values));
            assert_eq!(kb(&source_k.to_host_vec(&stream)?), kb(&keys));
            assert_eq!(vb(&source_v.to_host_vec(&stream)?), vb(&values));
            let mut expected_k = first_k;
            let mut expected_v = first_v;
            for base in (0..keys.len()).step_by($b * $n) {
                for i in (0..first_count).step_by($n) {
                    expected_k[base + i] = 111 as $kt;
                    expected_v[base + i] = 7 as $vt;
                }
            }
            assert_eq!(kb(&changed_k.to_host_vec(&stream)?), kb(&expected_k));
            assert_eq!(vb(&changed_v.to_host_vec(&stream)?), vb(&expected_v));
            let blocks = keys.len() / ($b * $n);
            cases += 1;
            selected += 2 * blocks * (first_count + second_count);
            preserved += 2 * keys.len();
            sources += 2 * keys.len();
            retained += 2 * blocks * first_count;
            flags_checked += keys.len();
            suffix += 2 * blocks * (2 * $b * $n - first_count - second_count);
        }};
    }
    if invalid == 0 {
        let len = 2 * 33 * 9;
        run!(
            f32,
            u64,
            9,
            33,
            2,
            prepare_f32_u64_manual,
            f32_u64_manual_async_owned,
            (0..len)
                .map(|i| {
                    let p = (5 * (i % 297) + 13 * (i / 297)) % 297;
                    if p < 16 {
                        -1.0
                    } else if p % 7 == 0 {
                        1.0
                    } else if p % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                })
                .collect(),
            (0..len)
                .map(|i| 0xd3a5_7000_0000_0000_u64 ^ i as u64)
                .collect(),
            17,
            false,
            true,
            128,
            73,
            |v: f32| v.to_bits() as u64,
            |v: u64| v
        );
    }
    let controls: &[(i64, i64)] = match invalid {
        0 => &[(0, 256), (256, 0)],
        1 => &[(i64::MIN, 256)],
        2 => &[(1, (1_i64 << 32) + 1)],
        3 => &[(0, 256)],
        4 => &[(1, 0)],
        _ => unreachable!(),
    };
    for &(k, valid) in controls {
        let blocks = if invalid == 0 { 2 } else { 1 };
        let len = blocks as usize * 64 * 4;
        run!(
            u64,
            i8,
            4,
            64,
            blocks,
            prepare_u64_i8_auto,
            u64_i8_auto_async_owned,
            (0..len).map(|i| 1 + (i * 11 % 31) as u64).collect(),
            (0..len).map(|i| (i * 17 % 101) as i8 - 50).collect(),
            256,
            true,
            false,
            k,
            valid,
            |v: u64| v,
            |v: i8| v as u8 as u64
        );
    }
    assert_eq!(
        (
            cases,
            selected,
            preserved,
            sources,
            retained,
            flags_checked,
            suffix
        ),
        (3, 2408, 3236, 3236, 2116, 1618, 4064)
    );
    println!(
        "topk: {cases} cases, {selected} selected values, {preserved} preserved, {sources} source, {retained} retained, {flags_checked} validity, {suffix} untouched suffix slots passed"
    );
    Ok(())
}
