// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each root observes fresh results, prefix validity and preserved input.
#![allow(clippy::too_many_arguments)]
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    macro_rules! exercise {
        ($n:expr, $alignment:literal, $threads:literal, $first_group:expr,
         $second_group:expr, $first_options:expr, $second_options:expr, $manual:literal,
         $input:ident, $first:ident, $second:ident, $keep:ident, $flags:ident, $changed:ident) => {{
            let block = coop::this_block();
            let offset = block.rank_at::<coop::Grid>() as usize * $threads * $n;
            let mut keys = coop::ThreadData::<_, $n, $alignment>::new();
            coop::load(
                block,
                $input,
                &mut keys,
                coop::LoadOptions::default().offset(offset),
            );
            let mut first = coop::merge_sort_keys($first_group, &keys, $first_options);
            if $manual {
                thread::sync_threads();
            }
            coop::store(
                block,
                &mut $first,
                &first,
                coop::StoreOptions::default().offset(offset),
            );
            first.write(0, 111 as _);
            let second = coop::merge_sort_keys($second_group, &keys, $second_options);
            if $manual {
                thread::sync_threads();
            }
            let mut visible = coop::ThreadData::<_, $n>::new();
            let mut flags = coop::ThreadData::<_, $n>::new();
            for item in 0..$n {
                // Observe validity safely, never reading an invalid numeric suffix.
                visible.write(item, second.get(item).unwrap_or(99 as _));
                flags.write(
                    item,
                    2 * i32::from(keys.get(item).is_some()) + i32::from(second.get(item).is_some()),
                );
            }
            coop::store(
                block,
                &mut $second,
                &visible,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $keep,
                &keys,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $flags,
                &flags,
                coop::StoreOptions::default().offset(offset),
            );
            coop::store(
                block,
                &mut $changed,
                &first,
                coop::StoreOptions::default().offset(offset),
            );
        }};
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn f32_manual<const ITEMS_PER_THREAD: usize>(
        input: &[f32],
        mut first_output: coop::DisjointSlice<f32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<f32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<f32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<f32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let mut scratch = coop::TempStorage::new();
        if invalid != 0 {
            let rank = (thread::threadIdx_x()
                + 4 * (thread::threadIdx_y() + 4 * thread::threadIdx_z()))
                as usize;
            let mut keys = coop::ThreadData::<_, ITEMS_PER_THREAD, 32>::new();
            for item in 0..ITEMS_PER_THREAD {
                if !((invalid == 1 || invalid == 4) && rank == 63 && item == 8) {
                    keys.write(item, input[rank * ITEMS_PER_THREAD + item]);
                }
            }
            let _ = coop::merge_sort_keys(
                block,
                &keys,
                coop::MergeSortOptions::default()
                    .order(coop::Descending)
                    .valid_items(count, f32::MIN)
                    .temp_storage(&mut scratch),
            );
            thread::sync_threads();
            // Independent report only: losing preflight must not be hidden by
            // an invalid-result read or a device assertion.
            let mut report = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
            for item in 0..ITEMS_PER_THREAD {
                report.write(item, 99_f32);
            }
            coop::store(block, &mut mutated, &report, Default::default());
            return;
        }
        exercise!(
            ITEMS_PER_THREAD,
            32,
            64,
            block,
            block,
            coop::MergeSortOptions::default().temp_storage(&mut scratch),
            coop::MergeSortOptions::default()
                .valid_items(count, f32::MIN)
                .order(coop::Descending)
                .temp_storage(&mut scratch),
            true,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn f64_auto<const ITEMS_PER_THREAD: usize>(
        input: &[f64],
        mut first_output: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<f64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let mut scratch = coop::TempStorage::new().auto_sync();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            16,
            64,
            block,
            block,
            coop::MergeSortOptions::default().temp_storage(&mut scratch),
            coop::MergeSortOptions::default()
                .valid_items(count, f64::MIN)
                .order(coop::Descending)
                .temp_storage(&mut scratch),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn warp1<const ITEMS_PER_THREAD: usize>(
        input: &[u8],
        mut first_output: coop::DisjointSlice<u8, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<u8, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<u8, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<u8, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let warp = coop::this_warp();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            1,
            64,
            warp.group_by::<1>(),
            warp.group_by::<1>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, u8::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn warp2<const ITEMS_PER_THREAD: usize>(
        input: &[i16],
        mut first_output: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let warp = coop::this_warp();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            2,
            64,
            warp.group_by::<2>(),
            warp.group_by::<2>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, i16::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn warp4<const ITEMS_PER_THREAD: usize>(
        input: &[u32],
        mut first_output: coop::DisjointSlice<u32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<u32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<u32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<u32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let warp = coop::this_warp();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            16,
            64,
            warp.group_by::<4>(),
            warp.group_by::<4>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, u32::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 6), dynamic_shared = inferred)]
    pub fn warp8<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut first_output: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let block = coop::this_block();
        let warp = coop::this_warp();
        if invalid != 0 {
            let rank = (thread::threadIdx_x()
                + 4 * (thread::threadIdx_y() + 4 * thread::threadIdx_z()))
                as usize;
            let mut keys = coop::ThreadData::<_, ITEMS_PER_THREAD, 32>::new();
            for item in 0..ITEMS_PER_THREAD {
                if !((invalid == 1 || invalid == 4) && rank == 95 && item == 8) {
                    keys.write(item, input[rank * ITEMS_PER_THREAD + item]);
                }
            }
            if invalid == 4 {
                let _ = coop::merge_sort_keys(warp.group_by::<8>(), &keys, Default::default());
            } else {
                let _ = coop::merge_sort_keys(
                    warp.group_by::<8>(),
                    &keys,
                    coop::MergeSortOptions::default()
                        .order(coop::Descending)
                        .valid_items(count, i32::MIN),
                );
            }
            // Independent report only: losing preflight must not be hidden by
            // an invalid-result read or a device assertion.
            let mut report = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
            for item in 0..ITEMS_PER_THREAD {
                report.write(item, 99_i32);
            }
            coop::store(block, &mut mutated, &report, Default::default());
            return;
        }
        exercise!(
            ITEMS_PER_THREAD,
            32,
            96,
            warp.group_by::<8>(),
            warp.group_by::<8>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, i32::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn warp16<const ITEMS_PER_THREAD: usize>(
        input: &[u64],
        mut first_output: coop::DisjointSlice<u64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<u64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<u64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<u64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let warp = coop::this_warp();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            16,
            64,
            warp.group_by::<16>(),
            warp.group_by::<16>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, u64::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn warp32<const ITEMS_PER_THREAD: usize>(
        input: &[i64],
        mut first_output: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut second_output: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut validity: coop::DisjointSlice<i32, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        mut mutated: coop::DisjointSlice<i64, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        count: i64,
        invalid: u32,
    ) {
        let warp = coop::this_warp();
        let _ = invalid;
        exercise!(
            ITEMS_PER_THREAD,
            32,
            64,
            warp,
            warp.group_by::<32>(),
            coop::MergeSortOptions::default(),
            coop::MergeSortOptions::default()
                .valid_items(count, i64::MIN)
                .order(coop::Descending),
            false,
            input,
            first_output,
            second_output,
            preserved,
            validity,
            mutated
        );
    }
}

fn check_sorted<T: Copy + PartialOrd + std::fmt::Debug, B: Ord + std::fmt::Debug>(
    input: &[T],
    output: &[T],
    group: usize,
    count: usize,
    descending: bool,
    bits: impl Fn(T) -> B,
) {
    for (keys, sorted) in input.chunks_exact(group).zip(output.chunks_exact(group)) {
        for pair in sorted[..count].windows(2) {
            assert!(if descending {
                pair[0] >= pair[1]
            } else {
                pair[0] <= pair[1]
            });
        }
        let mut expected: Vec<_> = keys[..count].iter().copied().map(&bits).collect();
        let mut observed: Vec<_> = sorted[..count].iter().copied().map(&bits).collect();
        expected.sort_unstable();
        observed.sort_unstable();
        // Equal floating keys can permute; their exact bit multiset must survive.
        assert_eq!(observed, expected);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("missing-zero-input") => 1,
        Some("negative-count") => 2,
        Some("wide-count") => 3,
        Some("missing-warp-input") => 4,
        Some(other) => panic!("unknown Merge Sort mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    let (mut cases, mut sorted, mut kept, mut source_checks, mut flags, mut changed) =
        (0, 0, 0, 0, 0, 0);
    macro_rules! run {
        ($t:ty, $n:literal, $threads:literal, $width:literal, $config:expr,
         $prepare:ident, $launch:ident, $count:expr, $selected:expr, $data:expr, $bits:expr) => {{
            if invalid == 0 || $selected {
                let blocks: usize = if invalid == 0 { 2 } else { 1 };
                let keys: Vec<$t> = (0..blocks * $threads * $n).map($data).collect();
                let bits = $bits;
                let count: i64 = $count;
                let input = DeviceBuffer::from_host(&stream, &keys)?;
                let first = DeviceBuffer::from_host(&stream, &vec![77 as $t; keys.len()])?;
                let second = DeviceBuffer::from_host(&stream, &vec![77 as $t; keys.len()])?;
                let keep = DeviceBuffer::from_host(&stream, &vec![77 as $t; keys.len()])?;
                let validity = DeviceBuffer::from_host(&stream, &vec![-1_i32; keys.len()])?;
                let mutated = DeviceBuffer::from_host(&stream, &vec![77 as $t; keys.len()])?;
                stream.synchronize()?;
                let plan = module.$prepare::<$n>(($config)(blocks))?;
                let (input, first, second, keep, validity, mutated) = module
                    .$launch::<$n, _, _, _, _, _, _>(
                        &plan, input, first, second, keep, validity, mutated, count, invalid,
                    )
                    .sync()?;
                stream.synchronize()?;
                if invalid != 0 {
                    panic!("invalid Merge Sort mode {invalid} unexpectedly completed");
                }
                let first = first.to_host_vec(&stream)?;
                let second = second.to_host_vec(&stream)?;
                let keep = keep.to_host_vec(&stream)?;
                let source = input.to_host_vec(&stream)?;
                let validity = validity.to_host_vec(&stream)?;
                let mutated = mutated.to_host_vec(&stream)?;
                check_sorted(&keys, &first, $width * $n, $width * $n, false, bits);
                check_sorted(&keys, &second, $width * $n, count as usize, true, bits);
                for (index, value) in second.iter().enumerate() {
                    #[allow(clippy::modulo_one)] // Width-one singleton groups are intentional.
                    let valid = index % ($width * $n) < count as usize;
                    assert_eq!(validity[index], 2 + i32::from(valid));
                    if !valid {
                        assert_eq!(bits(*value), bits(99 as $t));
                    }
                    assert_eq!(bits(keep[index]), bits(keys[index]));
                    assert_eq!(bits(source[index]), bits(keys[index]));
                    let expected = if index.is_multiple_of($n) {
                        111 as $t
                    } else {
                        first[index]
                    };
                    assert_eq!(bits(mutated[index]), bits(expected));
                }
                cases += 1;
                sorted += keys.len() + (keys.len() / ($width * $n)) * count as usize;
                kept += keys.len();
                source_checks += keys.len();
                flags += keys.len();
                changed += keys.len();
            }
        }};
    }
    run!(
        f32,
        9,
        64,
        64,
        |blocks| LaunchConfig3D::auto_shared((blocks as u32, 1, 1), (4, 4, 4)),
        prepare_f32_manual,
        f32_manual_async_owned,
        if invalid == 1 {
            0
        } else if invalid == 2 {
            i64::MIN
        } else {
            575
        },
        invalid == 1 || invalid == 2,
        |i| if i % 11 == 0 {
            -0.0_f32
        } else if i % 13 == 0 {
            0.0
        } else {
            ((i * 17 + i / 576 * 7) % 31) as f32 - 15.0
        },
        f32::to_bits
    );
    run!(
        f64,
        4,
        64,
        64,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_f64_auto,
        f64_auto_async_owned,
        0,
        false,
        |i| if i % 11 == 0 {
            -0.0_f64
        } else if i % 13 == 0 {
            0.0
        } else {
            ((i * 17 + i / 256 * 7) % 31) as f64 - 15.0
        },
        f64::to_bits
    );
    run!(
        u8,
        1,
        64,
        1,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_warp1,
        warp1_async_owned,
        1,
        false,
        |i| ((i * 17 + i * 7) % 31 + 1) as u8,
        |v: u8| v
    );
    run!(
        i16,
        4,
        64,
        2,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_warp2,
        warp2_async_owned,
        7,
        false,
        |i| ((i * 17 + i / 8 * 7) % 31) as i16 - 15,
        |v: i16| v
    );
    run!(
        u32,
        4,
        64,
        4,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_warp4,
        warp4_async_owned,
        16,
        false,
        |i| ((i * 17 + i / 16 * 7) % 31 + 1) as u32,
        |v: u32| v
    );
    run!(
        i32,
        9,
        96,
        8,
        |blocks| LaunchConfig3D::auto_shared((blocks as u32, 1, 1), (4, 4, 6)),
        prepare_warp8,
        warp8_async_owned,
        if invalid == 3 { (1_i64 << 32) + 1 } else { 71 },
        invalid == 3 || invalid == 4,
        |i| ((i * 17 + i / 72 * 7) % 31) as i32 - 15,
        |v: i32| v
    );
    run!(
        u64,
        4,
        64,
        16,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_warp16,
        warp16_async_owned,
        0,
        false,
        |i| ((i * 17 + i / 64 * 7) % 31 + 1) as u64,
        |v: u64| v
    );
    run!(
        i64,
        4,
        64,
        32,
        |blocks| LaunchConfig1D::auto_shared(blocks as u32, 64),
        prepare_warp32,
        warp32_async_owned,
        127,
        false,
        |i| ((i * 17 + i / 128 * 7) % 31) as i64 - 15,
        |v: i64| v
    );
    assert_eq!(
        (cases, sorted, kept, source_checks, flags, changed),
        (8, 10018, 5568, 5568, 5568, 5568)
    );
    println!(
        "merge_sort_keys: {cases} cases, {sorted} sorted keys, {kept} preserved keys, {source_checks} source keys, {flags} validity flags and {changed} retained mutated keys passed"
    );
    Ok(())
}
