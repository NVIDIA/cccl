// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernels expose independent typed key/value outputs and preserved inputs.
#![allow(clippy::too_many_arguments)]

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:literal, $ka:literal, $va:literal, $b:literal,
         $first_options:expr, $second_options:expr, $manual:literal,
         $keys:ident, $values:ident, $first_keys:ident, $first_values:ident,
         $second_keys:ident, $second_values:ident, $preserved_keys:ident,
         $preserved_values:ident $(, $mutated_keys:ident)?) => {{
            let block = coop::this_block();
            let offset = block.rank_at::<coop::Grid>() as usize * $b * $n;
            let mut keys = coop::ThreadData::<_, $n, $ka>::new();
            let mut values = coop::ThreadData::<_, $n, $va>::new();
            coop::load(block, $keys, &mut keys, coop::LoadOptions::default().offset(offset));
            coop::load(block, $values, &mut values, coop::LoadOptions::default().offset(offset));
            #[allow(unused_mut)]
            let (mut first_k, first_v) =
                coop::radix_sort_pairs(block, &keys, &values, $first_options);
            if $manual {
                thread::sync_threads();
                coop::store(block, &mut $first_keys, &first_k, coop::StoreOptions::default().offset(offset));
            }
            $(
                // Observe the original result before changing one fresh slot.
                // Both this payload and first_v remain live across scratch reuse.
                first_k.write(0, -7_i64);
                let _ = &$mutated_keys;
            )?
            let (second_k, second_v) =
                coop::radix_sort_pairs(block, &keys, &values, $second_options);
            if $manual {
                thread::sync_threads();
            }
            if !$manual {
                coop::store(block, &mut $first_keys, &first_k, coop::StoreOptions::default().offset(offset));
            }
            coop::store(block, &mut $first_values, &first_v, coop::StoreOptions::default().offset(offset));
            coop::store(block, &mut $second_keys, &second_k, coop::StoreOptions::default().offset(offset));
            coop::store(block, &mut $second_values, &second_v, coop::StoreOptions::default().offset(offset));
            coop::store(block, &mut $preserved_keys, &keys, coop::StoreOptions::default().offset(offset));
            coop::store(block, &mut $preserved_values, &values, coop::StoreOptions::default().offset(offset));
            $(coop::store(block, &mut $mutated_keys, &first_k, coop::StoreOptions::default().offset(offset));)?
        }};
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (3, 5, 2), dynamic_shared = inferred)]
    pub fn u32_f32_xyz(
        keys: &[u32],
        values: &[f32],
        mut first_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut first_values: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut second_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut second_values: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        mut preserved_keys: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut preserved_values: coop::DisjointSlice<f32, coop::CollectiveTiles<4>>,
        begin: i64,
        end: i64,
    ) {
        exercise!(
            4,
            16,
            1,
            30,
            coop::RadixSortOptions::default(),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .begin_bit(begin)
                .end_bit(end),
            false,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            preserved_keys,
            preserved_values
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
        mut preserved_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        mut preserved_values: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut mutated_keys: coop::DisjointSlice<i64, coop::CollectiveTiles<9>>,
        begin: i64,
        end: i64,
        invalid: u32,
    ) {
        let mut scratch = coop::TempStorage::new();
        if invalid != 0 {
            let block = coop::this_block();
            let rank = thread::threadIdx_x() as usize;
            let mut input_keys = coop::ThreadData::<_, 9, 1>::new();
            let mut input_values = coop::ThreadData::<_, 9, 32>::new();
            let mut i = 0;
            while i < 9 {
                if !(invalid == 1 && rank == 63 && i == 8) {
                    input_keys.write(i, keys[rank * 9 + i]);
                }
                if !(invalid == 2 && rank == 63 && i == 8) {
                    input_values.write(i, values[rank * 9 + i]);
                }
                i += 1;
            }
            let _ = coop::radix_sort_pairs(
                block,
                &input_keys,
                &input_values,
                coop::RadixSortOptions::default()
                    .begin_bit(begin)
                    .end_bit(end)
                    .temp_storage(&mut scratch),
            );
            thread::sync_threads();
            // If preflight is lost, return an independent report and fail on
            // the host. Never observe either invalid result to force a trap.
            let mut report = coop::ThreadData::<_, 9>::new();
            let mut i = 0;
            while i < 9 {
                report.write(i, 99_i64);
                i += 1;
            }
            coop::store(block, &mut mutated_keys, &report, Default::default());
            return;
        }
        exercise!(
            9,
            1,
            32,
            64,
            coop::RadixSortOptions::default()
                .begin_bit(begin)
                .temp_storage(&mut scratch),
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .begin_bit_const::<61>()
                .end_bit(end)
                .temp_storage(&mut scratch),
            true,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            preserved_keys,
            preserved_values,
            mutated_keys
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn u64_i8_auto(
        keys: &[u64],
        values: &[i8],
        mut first_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<1>>,
        mut first_values: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
        mut second_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<1>>,
        mut second_values: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
        mut preserved_keys: coop::DisjointSlice<u64, coop::CollectiveTiles<1>>,
        mut preserved_values: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        exercise!(
            1,
            32,
            1,
            33,
            coop::RadixSortOptions::default()
                .order(coop::Descending)
                .temp_storage(&mut scratch),
            coop::RadixSortOptions::default()
                .begin_bit_const::<0>()
                .end_bit_const::<3>()
                .temp_storage(&mut scratch),
            false,
            keys,
            values,
            first_keys,
            first_values,
            second_keys,
            second_values,
            preserved_keys,
            preserved_values
        );
    }
}

// Sorting original indices independently checks both stability and association.
fn oracle<K: Copy, V: Copy>(
    keys: &[K],
    values: &[V],
    tile: usize,
    begin: u32,
    end: u32,
    descending: bool,
    ordered: impl Fn(K) -> u128,
) -> (Vec<K>, Vec<V>) {
    let mask = (1_u128 << (end - begin)) - 1;
    let digit = |index: usize| (ordered(keys[index]) >> begin) & mask;
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    for block in indices.chunks_exact_mut(tile) {
        block.sort_by(|&left, &right| {
            if descending {
                digit(right).cmp(&digit(left))
            } else {
                digit(left).cmp(&digit(right))
            }
        });
    }
    (
        indices.iter().map(|&i| keys[i]).collect(),
        indices.iter().map(|&i| values[i]).collect(),
    )
}

fn signed_keys(tile: usize, blocks: usize) -> Vec<i64> {
    (0..blocks * tile)
        .map(|index| {
            let j = index % tile;
            let digit = ((5 * j + 3 * (index / tile)) % 8) as u64;
            match j {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => -1,
                3 => 0,
                _ => ((digit << 61 | (index as u64 + 1)) ^ (1_u64 << 63)) as i64,
            }
        })
        .collect()
}

fn unsigned_keys(width: u32, tile: usize, blocks: usize) -> Vec<u64> {
    (0..blocks * tile)
        .map(|index| {
            let j = index % tile;
            let digit = ((5 * j + 3 * (index / tile)) % 8) as u64;
            let tag = index as u64 + 1;
            match j {
                0 => 0,
                1 if width == 32 => u32::MAX as u64,
                1 => u64::MAX,
                _ if width == 32 => (tag << 7) | (digit << 4) | (tag & 15),
                _ => (((j & 1) as u64) << 63) | (tag << 3) | digit,
            }
        })
        .collect()
}

fn f32_values(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| match i % 8 {
            0 => f32::from_bits(0x7fc0_0000 | (i as u32 + 1)),
            1 => f32::from_bits(0xffc0_0000 | (i as u32 + 1)),
            2 => 0.0,
            3 => -0.0,
            4 => f32::INFINITY,
            5 => f32::NEG_INFINITY,
            _ => i as f32 + 0.25,
        })
        .collect()
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
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    let (invalid, begin, end) = match std::env::args().nth(1).as_deref() {
        None => (0, 61_i64, 64_i64),
        Some("missing-key") => (1, 61, 64),
        Some("missing-value") => (2, 61, 64),
        Some("negative-begin") => (3, i64::MIN, 64),
        Some("empty") => (4, 61, 61),
        Some("wide-end") => (5, 0, (1_i64 << 32) + 64),
        Some(other) => panic!("unknown Radix Sort Pairs mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut sorted, mut preserved, mut sources, mut mutated) = (0, 0, 0, 0, 0);
    macro_rules! run {
        ($kt:ty, $vt:ty, $n:literal, $b:literal, $prepare:ident, $launch:ident,
         $config:expr, $keys:expr, $values:expr, $first:expr, $second:expr,
         $ordered:expr, $bits:expr, [$($argument:expr),*] $(, $mutation:ident)?) => {{
            let keys: Vec<$kt> = $keys;
            let values: Vec<$vt> = $values;
            let (b1, e1, d1) = $first;
            let (b2, e2, d2) = $second;
            let (first_k, first_v) = oracle(&keys, &values, $b * $n, b1, e1, d1, $ordered);
            let (second_k, second_v) = oracle(&keys, &values, $b * $n, b2, e2, d2, $ordered);
            let (whole_k, whole_v) = oracle(&keys, &values, $b * $n, 0, <$kt>::BITS, d2, $ordered);
            let bits = $bits;
            let value_bits = |v: &[$vt]| v.iter().copied().map(bits).collect::<Vec<_>>();
            assert!(second_k != whole_k || value_bits(&second_v) != value_bits(&whole_v), "selected-digit oracle must distinguish whole-key order");
            let source_k = DeviceBuffer::from_host(&stream, &keys)?;
            let source_v = DeviceBuffer::from_host(&stream, &values)?;
            let out_k1 = DeviceBuffer::from_host(&stream, &vec![0 as $kt; keys.len()])?;
            let out_v1 = DeviceBuffer::from_host(&stream, &vec![0 as $vt; keys.len()])?;
            let out_k2 = DeviceBuffer::from_host(&stream, &vec![0 as $kt; keys.len()])?;
            let out_v2 = DeviceBuffer::from_host(&stream, &vec![0 as $vt; keys.len()])?;
            let keep_k = DeviceBuffer::from_host(&stream, &vec![0 as $kt; keys.len()])?;
            let keep_v = DeviceBuffer::from_host(&stream, &vec![0 as $vt; keys.len()])?;
            $(let $mutation = DeviceBuffer::from_host(&stream, &vec![0 as $kt; keys.len()])?;)?
            stream.synchronize()?;
            let plan = module.$prepare($config)?;
            let (source_k, source_v, out_k1, out_v1, out_k2, out_v2, keep_k, keep_v $(, $mutation)?) = module.$launch(
                &plan, source_k, source_v, out_k1, out_v1, out_k2, out_v2, keep_k, keep_v $(, $mutation)? $(, $argument)*).sync()?;
            stream.synchronize()?;
            if invalid != 0 { panic!("invalid Radix Sort Pairs mode {invalid} unexpectedly completed"); }
            assert_eq!(out_k1.to_host_vec(&stream)?, first_k, stringify!($prepare));
            assert_eq!(value_bits(&out_v1.to_host_vec(&stream)?), value_bits(&first_v), stringify!($prepare));
            assert_eq!(out_k2.to_host_vec(&stream)?, second_k, stringify!($prepare));
            assert_eq!(value_bits(&out_v2.to_host_vec(&stream)?), value_bits(&second_v), stringify!($prepare));
            assert_eq!(keep_k.to_host_vec(&stream)?, keys, stringify!($prepare));
            assert_eq!(value_bits(&keep_v.to_host_vec(&stream)?), value_bits(&values), stringify!($prepare));
            assert_eq!(source_k.to_host_vec(&stream)?, keys, stringify!($prepare));
            assert_eq!(value_bits(&source_v.to_host_vec(&stream)?), value_bits(&values), stringify!($prepare));
            $(let mut expected_mutated = first_k;
              for key in expected_mutated.iter_mut().step_by($n) { *key = -7_i64; }
              assert_eq!($mutation.to_host_vec(&stream)?, expected_mutated);
              mutated += keys.len();)?
            cases += 1; sorted += 4 * keys.len(); preserved += 2 * keys.len(); sources += 2 * keys.len();
        }};
    }
    if invalid != 0 {
        run!(
            i64,
            f64,
            9,
            64,
            prepare_i64_f64_manual,
            i64_f64_manual_async_owned,
            LaunchConfig1D::auto_shared(1, 64),
            signed_keys(576, 1),
            f64_values(576),
            (61, 64, false),
            (61, 64, true),
            |v: i64| ((v as u64) ^ (1_u64 << 63)) as u128,
            f64::to_bits,
            [begin, end, invalid],
            changed
        );
    } else {
        run!(
            u32,
            f32,
            4,
            30,
            prepare_u32_f32_xyz,
            u32_f32_xyz_async_owned,
            LaunchConfig3D::auto_shared((2, 1, 1), (3, 5, 2)),
            unsigned_keys(32, 120, 2)
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            f32_values(240),
            (0, 32, false),
            (4, 7, true),
            |v: u32| v as u128,
            f32::to_bits,
            [4_i64, 7_i64]
        );
        run!(
            i64,
            f64,
            9,
            64,
            prepare_i64_f64_manual,
            i64_f64_manual_async_owned,
            LaunchConfig1D::auto_shared(2, 64),
            signed_keys(576, 2),
            f64_values(1152),
            (61, 64, false),
            (61, 64, true),
            |v: i64| ((v as u64) ^ (1_u64 << 63)) as u128,
            f64::to_bits,
            [begin, end, invalid],
            changed
        );
        run!(
            u64,
            i8,
            1,
            33,
            prepare_u64_i8_auto,
            u64_i8_auto_async_owned,
            LaunchConfig1D::auto_shared(2, 33),
            unsigned_keys(64, 33, 2),
            (0..66).map(|i| i as i8 - 33).collect(),
            (0, 64, true),
            (0, 3, false),
            |v: u64| v as u128,
            |v: i8| v,
            []
        );
    }
    assert_eq!(
        (cases, sorted, preserved, sources, mutated),
        (3, 5832, 2916, 2916, 1152)
    );
    println!(
        "radix_sort_pairs: {cases} cases, {sorted} sorted key/value observations, {preserved} preserved values, {sources} source values, {mutated} retained mutated keys passed"
    );
    Ok(())
}
