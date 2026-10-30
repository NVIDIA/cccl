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
        matches && left.get(N).is_none() && left.get(usize::MAX).is_none()
    }

    fn shifted<T: coop::Numeric + PartialEq, const N: usize, const A: usize>(
        values: &coop::ThreadData<T, N, A>,
        expected: &[T],
        tid: usize,
        rank: usize,
        threads: usize,
        up: bool,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut matches = values.len() == N;
        let mut i = 0;
        while i < N {
            let valid = if up {
                rank != 0 || i != 0
            } else {
                rank != threads - 1 || i != N - 1
            };
            matches &= if valid {
                values.get(i) == Some(expected[tid * N + i])
            } else {
                values.get(i).is_none()
            };
            i += 1;
        }
        matches && values.get(N).is_none() && values.get(usize::MAX).is_none()
    }

    fn invoke<T: coop::Numeric, const N: usize, const A: usize>(
        block: coop::ThreadGroup<coop::Block>,
        values: &coop::ThreadData<T, N, A>,
        up: bool,
    ) -> coop::ThreadData<T, N, A>
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        if up {
            coop::shuffle(
                block,
                values,
                coop::ShuffleOptions::default().mode(coop::Up),
            )
        } else {
            coop::shuffle(block, values, Default::default())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exercise<T, const N: usize, const A: usize, const B: usize>(
        input: &[T],
        expected: &[T],
        output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<N>>,
        reports: &mut coop::DisjointSlice<'_, u32, coop::CollectiveTiles<1>>,
        rank: usize,
        up: bool,
        invalid: u32,
    ) where
        T: coop::Numeric + PartialEq + Default,
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let block = coop::this_block();
        let block_index = block.rank_at::<coop::Grid>() as usize;
        let tid = block_index * B + rank;
        let mut values = coop::ThreadData::<_, N, A>::new();
        let mut i = 0;
        while i < N {
            // These discarded values are still required by full-input preflight.
            let missing = (invalid == 1 && rank == 0 && i == 0)
                || (invalid == 2 && rank == B - 1 && i == N - 1);
            if !missing {
                values.write(i, input[tid * N + i]);
            }
            i += 1;
        }
        let before = values;
        let mut first = invoke(block, &values, up);
        let mut values_match = shifted(&first, expected, tid, rank, B, up);
        let mut input_preserved = same(&values, &before);
        // All input values are nonzero. Keep this changed result live across
        // scratch reuse, including the entirely invalid Block1/N1 result.
        first.write(0, T::default());
        let mutated = first;
        let mut second = invoke(block, &values, up);
        values_match &= shifted(&second, expected, tid, rank, B, up);
        input_preserved &= same(&values, &before);
        let independent = same(&first, &mutated) && first.get(0) == Some(T::default());
        let word =
            values_match as u32 | ((input_preserved as u32) << 1) | ((independent as u32) << 2);
        let unrepaired = second;
        if invalid != 4 {
            let mut i = 0;
            while i < N {
                if second.get(i).is_none() {
                    second.write(i, T::default());
                }
                i += 1;
            }
        }
        coop::store(
            block,
            output,
            &second,
            coop::StoreOptions::default().offset(block_index * B * N),
        );
        let mut report = coop::ThreadData::<_, 1>::new();
        // If Index were to allow the invalid read, preserve its comparison in
        // global memory and complete normally; only the host then reports the
        // missing trap. A value-dependent assertion could falsely pass it.
        let observed = if invalid == 3 {
            word | (((unrepaired[0] == input[tid * N]) as u32) << 3)
        } else {
            word
        };
        report.write(0, observed);
        coop::store(
            block,
            reports,
            &report,
            coop::StoreOptions::default().offset(block_index * B),
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn block33(
        input: &[i8],
        expected: &[i8],
        mut output: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        up: bool,
        invalid: u32,
    ) {
        exercise::<_, 1, 1, 33>(
            input,
            expected,
            &mut output,
            &mut reports,
            thread::threadIdx_x() as usize,
            up,
            invalid,
        );
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 3, 2), dynamic_shared = inferred)]
    pub fn block_xyz(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        up: bool,
    ) {
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 3 * thread::threadIdx_z());
        exercise::<_, 4, 16, 24>(
            input,
            expected,
            &mut output,
            &mut reports,
            rank as usize,
            up,
            0,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn block64(
        input: &[f64],
        expected: &[f64],
        mut output: coop::DisjointSlice<f64, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        up: bool,
    ) {
        exercise::<_, 9, 1, 64>(
            input,
            expected,
            &mut output,
            &mut reports,
            thread::threadIdx_x() as usize,
            up,
            0,
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_item(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        up: bool,
    ) {
        exercise::<_, 1, 1, 1>(input, expected, &mut output, &mut reports, 0, up, 0);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_thread(
        input: &[u16],
        expected: &[u16],
        mut output: coop::DisjointSlice<u16, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        up: bool,
    ) {
        exercise::<_, 4, 1, 1>(input, expected, &mut output, &mut reports, 0, up, 0);
    }
}

fn oracle<T: Copy + Default>(input: &[T], tile: usize, up: bool) -> Vec<T> {
    input
        .chunks_exact(tile)
        .flat_map(|values| {
            if up {
                std::iter::once(T::default())
                    .chain(values[..tile - 1].iter().copied())
                    .collect::<Vec<_>>()
            } else {
                values[1..]
                    .iter()
                    .copied()
                    .chain(std::iter::once(T::default()))
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};

    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("missing-down-input") => 1,
        Some("missing-up-input") => 2,
        Some("invalid-edge-read") => 3,
        Some("unrepaired-store") => 4,
        Some(other) => panic!("unknown test mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    if invalid != 0 {
        let up = invalid == 2 || invalid == 3;
        let input: Vec<i8> = (1..=33).collect();
        let expected = oracle(&input, 33, up);
        let source = DeviceBuffer::from_host(&stream, &input)?;
        let expected = DeviceBuffer::from_host(&stream, &expected)?;
        let target = DeviceBuffer::from_host(&stream, &[-1_i8; 33])?;
        let reports = DeviceBuffer::from_host(&stream, &[0_u32; 33])?;
        let plan = module.prepare_block33(LaunchConfig1D::auto_shared(1, 33))?;
        module
            .block33_async_owned(&plan, source, expected, target, reports, up, invalid)
            .sync()?;
        stream.synchronize()?;
        panic!("invalid Shuffle mode {invalid} unexpectedly completed");
    }

    let mut cases = 0;
    let mut reports = 0;
    let mut values = 0;
    macro_rules! run {
        ($ty:ty, $items:literal, $threads:literal, $prepare:ident, $launch:ident, $config:expr $(, $extra:expr)?) => {{
            let input: Vec<$ty> = (1..=3 * $threads * $items).map(|i| i as $ty).collect();
            for up in [false, true] {
                let expected = oracle(&input, $threads * $items, up);
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected_device = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &vec![<$ty>::MAX; input.len()])?;
                let words = DeviceBuffer::from_host(&stream, &[0_u32; 3 * $threads])?;
                let plan = module.$prepare($config)?;
                let (_, _, target, words) = module.$launch(
                    &plan, source, expected_device, target, words, up $(, $extra)?
                ).sync()?;
                assert_eq!(target.to_host_vec(&stream)?, expected, "{} up{}", stringify!($prepare), up);
                assert_eq!(words.to_host_vec(&stream)?, vec![CHECKS; 3 * $threads], "{} up{}", stringify!($prepare), up);
                cases += 1;
                reports += 3 * $threads;
                values += input.len();
            }
        }};
    }
    run!(
        i8,
        1,
        33,
        prepare_block33,
        block33_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        0
    );
    run!(
        i32,
        4,
        24,
        prepare_block_xyz,
        block_xyz_async_owned,
        LaunchConfig3D::auto_shared((3, 1, 1), (4, 3, 2))
    );
    run!(
        f64,
        9,
        64,
        prepare_block64,
        block64_async_owned,
        LaunchConfig1D::auto_shared(3, 64)
    );
    run!(
        i32,
        1,
        1,
        prepare_single_item,
        single_item_async_owned,
        LaunchConfig1D::auto_shared(3, 1)
    );
    run!(
        u16,
        4,
        1,
        prepare_single_thread,
        single_thread_async_owned,
        LaunchConfig1D::auto_shared(3, 1)
    );
    println!(
        "PASS: {cases} public Shuffle cases, {values} repaired values, {reports} lane reports"
    );
    Ok(())
}
