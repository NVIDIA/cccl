// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernels expose separate ranks, preserved inputs and a retained fresh result.
#![allow(clippy::too_many_arguments)]

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:literal, $a:literal, $b:literal, $first:expr, $second:expr,
         $keys:ident, $first_output:ident, $second_output:ident, $preserved:ident
         $(, $mutated:ident)?) => {{
            let block = coop::this_block();
            let offset = block.rank_at::<coop::Grid>() as usize * $b * $n;
            let mut keys = coop::ThreadData::<_, $n, $a>::new();
            coop::load(block, $keys, &mut keys, coop::LoadOptions::default().offset(offset));
            #[allow(unused_mut)]
            let mut first: coop::ThreadData<i32, $n> =
                coop::radix_rank_keys(block, &keys, $first);
            coop::store(block, &mut $first_output, &first, coop::StoreOptions::default().offset(offset));
            $(
                first.write(0, -7);
                let _ = &$mutated;
            )?
            let second: coop::ThreadData<i32, $n> =
                coop::radix_rank_keys(block, &keys, $second);
            coop::store(block, &mut $second_output, &second, coop::StoreOptions::default().offset(offset));
            coop::store(block, &mut $preserved, &keys, coop::StoreOptions::default().offset(offset));
            $(coop::store(block, &mut $mutated, &first, coop::StoreOptions::default().offset(offset));)?
        }};
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn i32_nonwarp(
        keys: &[i32],
        mut first: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut second: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut mutated: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        begin: i64,
        invalid: u32,
    ) {
        if invalid != 0 {
            let block = coop::this_block();
            let rank = thread::threadIdx_x() as usize;
            let mut input = coop::ThreadData::<_, 9, 32>::new();
            for item in 0..9 {
                if !(invalid == 1 && rank == 32 && item == 8) {
                    input.write(item, keys[rank * 9 + item]);
                }
            }
            if invalid == 2 || invalid == 3 {
                let _ = coop::radix_rank_keys(
                    block,
                    &input,
                    coop::RadixRankOptions::default()
                        .radix_bits::<8>()
                        .begin_bit(begin)
                        .order(coop::Descending),
                );
            } else {
                let _ = coop::radix_rank_keys(
                    block,
                    &input,
                    coop::RadixRankOptions::default()
                        .radix_bits::<8>()
                        .begin_bit(begin)
                        .end_bit_const::<32>()
                        .order(coop::Descending),
                );
            }
            // A missing native preflight must fail on host completion. The
            // invalid Rank result is discarded, never read to force a trap.
            let mut report = coop::ThreadData::<_, 9>::new();
            for item in 0..9 {
                report.write(item, 99_i32);
            }
            coop::store(block, &mut mutated, &report, Default::default());
            return;
        }
        exercise!(
            9,
            32,
            33,
            coop::RadixRankOptions::default()
                .begin_bit_const::<24>()
                .end_bit_const::<32>(),
            coop::RadixRankOptions::default()
                .radix_bits::<8>()
                .begin_bit(begin)
                .end_bit_const::<32>()
                .order(coop::Descending),
            keys,
            first,
            second,
            preserved,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (3, 5, 2), dynamic_shared = inferred)]
    pub fn u32_xyz(
        keys: &[u32],
        mut first: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut second: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        begin: i64,
    ) {
        exercise!(
            4,
            16,
            30,
            coop::RadixRankOptions::default(),
            coop::RadixRankOptions::default()
                .radix_bits::<4>()
                .begin_bit(begin)
                .order(coop::Descending),
            keys,
            first,
            second,
            preserved
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn u64_one_bit(
        keys: &[u64],
        mut first: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut second: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<u64, coop::CollectiveTiles<1>>,
    ) {
        exercise!(
            1,
            1,
            64,
            coop::RadixRankOptions::default()
                .begin_bit_const::<63>()
                .radix_bits::<1>(),
            coop::RadixRankOptions::default()
                .begin_bit_const::<63>()
                .end_bit_const::<64>()
                .order(coop::Descending),
            keys,
            first,
            second,
            preserved
        );
    }
}

// Stable sorting of original indices, followed by permutation inversion, gives
// ranks in input-slot order and distinguishes stable ties from sorted digits.
fn oracle<K: Copy>(
    keys: &[K],
    tile: usize,
    begin: u32,
    end: u32,
    descending: bool,
    ordered: impl Fn(K) -> u128,
) -> Vec<i32> {
    let mask = (1_u128 << (end - begin)) - 1;
    let mut output = vec![-1; keys.len()];
    for (block, values) in keys.chunks_exact(tile).enumerate() {
        let digit = |index: usize| (ordered(values[index]) >> begin) & mask;
        let mut indices: Vec<usize> = (0..tile).collect();
        indices.sort_by(|&left, &right| {
            if descending {
                digit(right).cmp(&digit(left))
            } else {
                digit(left).cmp(&digit(right))
            }
        });
        for (rank, index) in indices.into_iter().enumerate() {
            output[block * tile + index] = rank as i32;
        }
        let mut permutation = output[block * tile..(block + 1) * tile].to_vec();
        permutation.sort_unstable();
        assert_eq!(permutation, (0..tile as i32).collect::<Vec<_>>());
    }
    output
}

fn signed_keys(blocks: usize) -> Vec<i32> {
    (0..blocks * 297)
        .map(|index| {
            let j = index % 297;
            let digit = (((5 * j + 3 * (index / 297)) % 16) * 17) as u32;
            match j {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                3 => 0,
                _ => ((digit << 24 | (index as u32 + 1)) ^ (1_u32 << 31)) as i32,
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    let (invalid, begin) = match std::env::args().nth(1).as_deref() {
        None => (0, 24_i64),
        Some("missing-input") => (1, 24),
        Some("negative-begin") => (2, i64::MIN),
        Some("wide-begin") => (3, (1_i64 << 32) + 24),
        Some("end-mismatch") => (4, 23),
        Some(other) => panic!("unknown Radix Rank mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut ranks, mut preserved, mut sources, mut mutated) = (0, 0, 0, 0, 0);
    macro_rules! run {
        ($t:ty, $n:literal, $b:literal, $prepare:ident, $launch:ident,
         $config:expr, $keys:expr, $first:expr, $second:expr, $ordered:expr,
         [$($argument:expr),*] $(, $mutation:ident)?) => {{
            let keys: Vec<$t> = $keys;
            let (b1, e1, d1) = $first;
            let (b2, e2, d2) = $second;
            let first = oracle(&keys, $b * $n, b1, e1, d1, $ordered);
            let second = oracle(&keys, $b * $n, b2, e2, d2, $ordered);
            let whole = oracle(&keys, $b * $n, 0, <$t>::BITS, d2, $ordered);
            assert_ne!(second, whole, "selected-digit ranks must distinguish whole-key order");
            let input = DeviceBuffer::from_host(&stream, &keys)?;
            let output1 = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let output2 = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
            let keep = DeviceBuffer::from_host(&stream, &vec![0 as $t; keys.len()])?;
            $(let $mutation = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;)?
            stream.synchronize()?;
            let plan = module.$prepare($config)?;
            let (input, output1, output2, keep $(, $mutation)?) = module.$launch(
                &plan, input, output1, output2, keep $(, $mutation)? $(, $argument)*).sync()?;
            stream.synchronize()?;
            if invalid != 0 { panic!("invalid Radix Rank mode {invalid} unexpectedly completed"); }
            assert_eq!(output1.to_host_vec(&stream)?, first, stringify!($prepare));
            assert_eq!(output2.to_host_vec(&stream)?, second, stringify!($prepare));
            assert_eq!(keep.to_host_vec(&stream)?, keys, stringify!($prepare));
            assert_eq!(input.to_host_vec(&stream)?, keys, stringify!($prepare));
            $(let mut expected = first;
              for value in expected.iter_mut().step_by($n) { *value = -7; }
              assert_eq!($mutation.to_host_vec(&stream)?, expected);
              mutated += keys.len();)?
            cases += 1; ranks += 2 * keys.len(); preserved += keys.len(); sources += keys.len();
        }};
    }
    run!(
        i32,
        9,
        33,
        prepare_i32_nonwarp,
        i32_nonwarp_async_owned,
        LaunchConfig1D::auto_shared(if invalid == 0 { 2 } else { 1 }, 33),
        signed_keys(if invalid == 0 { 2 } else { 1 }),
        (24, 32, false),
        (24, 32, true),
        |v: i32| ((v as u32) ^ (1_u32 << 31)) as u128,
        [begin, invalid],
        changed
    );
    run!(
        u32,
        4,
        30,
        prepare_u32_xyz,
        u32_xyz_async_owned,
        LaunchConfig3D::auto_shared((2, 1, 1), (3, 5, 2)),
        (0..240)
            .map(|i| match i % 120 {
                0 => 0,
                1 => u32::MAX,
                _ => (1_u32 << 31) | ((i + 1) << 16) | (((5 * i + 3) % 16) << 12) | ((3 * i) % 16),
            })
            .collect(),
        (0, 4, false),
        (12, 16, true),
        |v: u32| v as u128,
        [12_i64]
    );
    run!(
        u64,
        1,
        64,
        prepare_u64_one_bit,
        u64_one_bit_async_owned,
        LaunchConfig1D::auto_shared(2, 64),
        (0..128)
            .map(|i| match i % 64 {
                0 => 0,
                1 => u64::MAX,
                _ => (((i & 1) as u64) << 63) | (i as u64 + 1),
            })
            .collect(),
        (63, 64, false),
        (63, 64, true),
        |v: u64| v as u128,
        []
    );
    assert_eq!(
        (cases, ranks, preserved, sources, mutated),
        (3, 1924, 962, 962, 594)
    );
    println!(
        "radix_rank_keys: {cases} cases, {ranks} ranks, {preserved} preserved keys, {sources} source keys, {mutated} retained mutated ranks passed"
    );
    Ok(())
}
