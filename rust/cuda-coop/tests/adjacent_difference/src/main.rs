// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const CHECKS: u32 = 7;

#[cuda_module]
mod kernels {
    use super::*;

    fn same<T: coop::Numeric + PartialEq, const N: usize, const A: usize>(
        left: &coop::ThreadData<T, N, A>,
        right: &coop::ThreadData<T, N, A>,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut matches = left.len() == N;
        let mut i = 0;
        while i < N {
            matches &= left.get(i) == right.get(i);
            i += 1;
        }
        matches
    }

    fn observe<T: coop::Numeric + PartialEq, const N: usize, const A: usize>(
        input: &coop::ThreadData<T, N, A>,
        before: &coop::ThreadData<T, N, A>,
        result: &coop::ThreadData<T, N, A>,
        expected: &[T],
        tid: usize,
    ) -> u32
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut matches = result.len() == N;
        let mut i = 0;
        while i < N {
            // The inactive suffix is copied and initialized, never repaired.
            matches &= result.get(i) == Some(expected[tid * N + i]);
            i += 1;
        }
        matches as u32 | ((same(input, before) as u32) << 1)
    }

    fn invoke<T: coop::Numeric, const N: usize, const A: usize, const FORMS: u8>(
        block: coop::ThreadGroup<coop::Block>,
        input: &coop::ThreadData<T, N, A>,
        mode: u32,
        count: usize,
        outside: T,
    ) -> coop::ThreadData<T, N, A>
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        match FORMS {
            0 => match mode {
                0 => coop::adjacent_difference(block, input, Default::default()),
                1 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default().tile_predecessor_item(outside),
                ),
                2 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default().valid_items(count),
                ),
                3 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default()
                        .valid_items(count)
                        .tile_predecessor_item(outside),
                ),
                4 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default().direction(coop::Right),
                ),
                5 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default()
                        .direction(coop::Right)
                        .tile_successor_item(outside),
                ),
                6 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default()
                        .direction(coop::Right)
                        .valid_items(count),
                ),
                _ => panic!(),
            },
            1 => match mode {
                0 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default().direction(coop::Right),
                ),
                1 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default().tile_predecessor_item(outside),
                ),
                _ => panic!(),
            },
            2 => match mode {
                0 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default()
                        .valid_items(count)
                        .tile_predecessor_item(outside),
                ),
                1 => coop::adjacent_difference(
                    block,
                    input,
                    coop::AdjacentDifferenceOptions::default()
                        .direction(coop::Right)
                        .tile_successor_item(outside),
                ),
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    fn write_results<T: coop::Numeric, const N: usize, const A: usize, const B: usize>(
        block: coop::ThreadGroup<coop::Block>,
        output: &mut coop::DisjointSlice<T, coop::CollectiveTiles<N>>,
        reports: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        result: &coop::ThreadData<T, N, A>,
        word: u32,
        block_index: usize,
    ) where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        coop::store(
            block,
            output,
            result,
            coop::StoreOptions::default().offset(block_index * B * N),
        );
        let mut report = coop::ThreadData::<_, 1>::new();
        report.write(0, word);
        coop::store(
            block,
            reports,
            &report,
            coop::StoreOptions::default().offset(block_index * B),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn exercise<T, const N: usize, const A: usize, const B: usize, const FORMS: u8>(
        input: &[T],
        expected: &[T],
        output: &mut coop::DisjointSlice<T, coop::CollectiveTiles<N>>,
        reports: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        rank: usize,
        mode: u32,
        count: usize,
        outside: T,
        invalid: u32,
    ) where
        T: coop::Numeric + PartialEq + Default,
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let block = coop::this_block();
        let block_index = block.rank_at::<coop::Grid>() as usize;
        let tid = block_index * B + rank;
        let mut values = coop::ThreadData::<_, N, A>::new();
        if invalid == 0 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default().offset(block_index * B * N),
            );
        } else {
            let mut i = 0;
            while i < N {
                if !(invalid >= 3 && rank == B - 1 && i == N - 1) {
                    values.write(i, input[tid * N + i]);
                }
                i += 1;
            }
            let _result = coop::adjacent_difference(
                block,
                &values,
                coop::AdjacentDifferenceOptions::default().valid_items(count),
            );
            // Unexpected return must not inspect the result or trigger a
            // downstream result guard that could counterfeit the expected trap.
            let mut report = coop::ThreadData::<_, 1>::new();
            report.write(0, 99_u32);
            coop::store(
                block,
                reports,
                &report,
                coop::StoreOptions::default().offset(block_index * B),
            );
            return;
        }
        let before = values;
        let mut first = invoke::<_, N, A, FORMS>(block, &values, mode, count, outside);
        let mut word = observe(&values, &before, &first, expected, tid);
        first.write(0, T::default());
        let mutated = first;
        let second = invoke::<_, N, A, FORMS>(block, &values, mode, count, outside);
        word &= observe(&values, &before, &second, expected, tid);
        word |= ((same(&first, &mutated) && first.get(0) == Some(T::default())) as u32) << 2;
        write_results::<_, N, A, B>(block, output, reports, &second, word, block_index);
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn forms_i32(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        count: usize,
        outside: i32,
        invalid: u32,
    ) {
        exercise::<_, 4, 16, 33, 0>(
            input,
            expected,
            &mut output,
            &mut reports,
            thread::threadIdx_x() as usize,
            mode,
            count,
            outside,
            invalid,
        );
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (5, 3, 2), dynamic_shared = inferred)]
    pub fn xyz_n1(
        input: &[i8],
        expected: &[i8],
        mut output: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        count: usize,
        outside: i8,
    ) {
        let rank = thread::threadIdx_x() + 5 * (thread::threadIdx_y() + 3 * thread::threadIdx_z());
        exercise::<_, 1, 1, 30, 1>(
            input,
            expected,
            &mut output,
            &mut reports,
            rank as usize,
            mode,
            count,
            outside,
            0,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn tail_n9(
        input: &[f64],
        expected: &[f64],
        mut output: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        count: usize,
        outside: f64,
    ) {
        exercise::<_, 9, 1, 32, 2>(
            input,
            expected,
            &mut output,
            &mut reports,
            thread::threadIdx_x() as usize,
            mode,
            count,
            outside,
            0,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_n1(
        input: &[u16],
        expected: &[u16],
        mut output: coop::DisjointSlice<u16, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        count: usize,
        outside: u16,
    ) {
        exercise::<_, 1, 1, 1, 2>(
            input,
            expected,
            &mut output,
            &mut reports,
            thread::threadIdx_x() as usize,
            mode,
            count,
            outside,
            0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn with_storage<const AUTO: bool>(
        input: &[i32],
        expected: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        reports: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        count: usize,
        outside: i32,
        scratch: &mut coop::TempStorage<0, 0, AUTO, false>,
    ) {
        let block = coop::this_block();
        let block_index = block.rank_at::<coop::Grid>() as usize;
        let tid = block_index * 33 + thread::threadIdx_x() as usize;
        let mut values = coop::ThreadData::<_, 4, 16>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default().offset(block_index * 132),
        );
        let before = values;
        let mut first = coop::adjacent_difference(
            block,
            &values,
            coop::AdjacentDifferenceOptions::default()
                .valid_items(count)
                .tile_predecessor_item(outside)
                .temp_storage(scratch),
        );
        let mut word = observe(&values, &before, &first, expected, tid);
        first.write(0, 0_i32);
        let mutated = first;
        if !AUTO {
            thread::sync_threads();
        }
        let second = coop::adjacent_difference(
            block,
            &values,
            coop::AdjacentDifferenceOptions::default()
                .valid_items(count)
                .tile_predecessor_item(outside)
                .temp_storage(scratch),
        );
        word &= observe(&values, &before, &second, expected, tid);
        word |= ((same(&first, &mutated) && first.get(0) == Some(0_i32)) as u32) << 2;
        if !AUTO {
            thread::sync_threads();
        }
        write_results::<_, 4, 16, 33>(block, output, reports, &second, word, block_index);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn manual_storage(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        _mode: u32,
        count: usize,
        outside: i32,
    ) {
        let mut scratch = coop::TempStorage::new();
        with_storage(
            input,
            expected,
            &mut output,
            &mut reports,
            count,
            outside,
            &mut scratch,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn auto_storage(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        _mode: u32,
        count: usize,
        outside: i32,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        with_storage(
            input,
            expected,
            &mut output,
            &mut reports,
            count,
            outside,
            &mut scratch,
        );
    }
}

fn oracle<T: Copy + std::ops::Sub<Output = T>>(
    input: &[T],
    tile: usize,
    recipe: u32,
    count: usize,
    outside: T,
) -> Vec<T> {
    let active = if matches!(recipe, 2 | 3 | 6) {
        count
    } else {
        tile
    };
    let mut expected = input.to_vec();
    for (source, target) in input
        .chunks_exact(tile)
        .zip(expected.chunks_exact_mut(tile))
    {
        for i in 0..active {
            if recipe < 4 {
                if i != 0 {
                    target[i] = source[i] - source[i - 1];
                } else if matches!(recipe, 1 | 3) {
                    target[i] = source[i] - outside;
                }
            } else if i + 1 < active {
                target[i] = source[i] - source[i + 1];
            } else if recipe == 5 {
                target[i] = source[i] - outside;
            }
        }
    }
    expected
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};

    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("count-over-capacity") => 1,
        Some("count-max") => 2,
        Some("missing-suffix-zero") => 3,
        Some("missing-suffix-partial") => 4,
        Some(other) => panic!("unknown test mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    if invalid != 0 {
        let count = match invalid {
            1 => 133,
            2 => usize::MAX,
            3 => 0,
            _ => 5,
        };
        let source = DeviceBuffer::from_host(&stream, &[20_i32; 132])?;
        let expected = DeviceBuffer::from_host(&stream, &[0_i32; 132])?;
        let target = DeviceBuffer::from_host(&stream, &[-1_i32; 132])?;
        let reports = DeviceBuffer::from_host(&stream, &[0_u32; 33])?;
        let plan = module.prepare_forms_i32(LaunchConfig1D::auto_shared(1, 33))?;
        module
            .forms_i32_async_owned(
                &plan, source, expected, target, reports, 2, count, 0, invalid,
            )
            .sync()?;
        stream.synchronize()?;
        panic!("invalid Adjacent Difference mode {invalid} unexpectedly completed");
    }

    let mut cases = 0;
    let mut reports = 0;
    let mut values = 0;
    macro_rules! run {
        ($ty:ty, $n:literal, $b:literal, $scale:literal, $prepare:ident, $launch:ident, $config:expr, $cases:expr $(, $extra:expr)?) => {{
            let input: Vec<$ty> = (0..3 * $b * $n).map(|i| {
                let block = i / ($b * $n);
                let slot = i % ($b * $n);
                (20 + (7 * slot + 3 * block) % 31) as $ty / $scale as $ty
            }).collect();
            for (mode, recipe, count, outside) in $cases {
                let expected = oracle(&input, $b * $n, recipe, count, outside);
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected_device = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &vec![<$ty>::MAX; input.len()])?;
                let words = DeviceBuffer::from_host(&stream, &[0_u32; 3 * $b])?;
                let plan = module.$prepare($config)?;
                let (_, _, target, words) = module.$launch(&plan, source, expected_device, target, words,
                    mode, count, outside $(, $extra)?).sync()?;
                assert_eq!(target.to_host_vec(&stream)?, expected, "{} mode{mode} count{count}", stringify!($prepare));
                assert_eq!(words.to_host_vec(&stream)?, vec![CHECKS; 3 * $b], "{} mode{mode} count{count}", stringify!($prepare));
                cases += 1;
                reports += 3 * $b;
                values += input.len();
            }
        }};
    }
    run!(
        i32,
        4,
        33,
        1,
        prepare_forms_i32,
        forms_i32_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        [
            (0, 0, 0, 0),
            (1, 1, 0, -11),
            (2, 2, 0, 0),
            (2, 2, 1, 0),
            (2, 2, 5, 0),
            (2, 2, 132, 0),
            (3, 3, 0, -11),
            (3, 3, 5, -11),
            (4, 4, 0, 0),
            (5, 5, 0, 13),
            (6, 6, 0, 0),
            (6, 6, 1, 0),
            (6, 6, 131, 0),
            (6, 6, 132, 0)
        ],
        0
    );
    run!(
        i8,
        1,
        30,
        1,
        prepare_xyz_n1,
        xyz_n1_async_owned,
        LaunchConfig3D::auto_shared((3, 1, 1), (5, 3, 2)),
        [(0, 4, 0, 0_i8), (1, 1, 0, -7_i8)]
    );
    run!(
        f64,
        9,
        32,
        2,
        prepare_tail_n9,
        tail_n9_async_owned,
        LaunchConfig1D::auto_shared(3, 32),
        [(0, 3, 10, 0.5_f64), (1, 5, 0, -3.5_f64)]
    );
    run!(
        u16,
        1,
        1,
        1,
        prepare_single_n1,
        single_n1_async_owned,
        LaunchConfig1D::auto_shared(3, 1),
        [(0, 3, 0, 3_u16), (1, 5, 0, 3_u16)]
    );
    run!(
        i32,
        4,
        33,
        1,
        prepare_manual_storage,
        manual_storage_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        [(3, 3, 5, -11_i32)]
    );
    run!(
        i32,
        4,
        33,
        1,
        prepare_auto_storage,
        auto_storage_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        [(3, 3, 5, -11_i32)]
    );
    println!(
        "PASS: {cases} public Adjacent Difference cases, {values} values, {reports} lane reports"
    );
    Ok(())
}
