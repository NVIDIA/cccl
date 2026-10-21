// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const SKIPPED: u32 = 1 << 31;
const CHECKS: u32 = 15;

#[cuda_module]
mod kernels {
    use super::*;

    fn preserved<T: coop::Numeric + PartialEq, const N: usize>(
        values: &coop::ThreadData<T, N>,
        before: &coop::ThreadData<T, N>,
    ) -> bool {
        let mut matches = values.len() == N;
        let mut i = 0;
        while i < N {
            matches &= values.get(i).is_some() && values.get(i) == before.get(i);
            i += 1;
        }
        matches && values.get(N).is_none() && values.get(usize::MAX).is_none()
    }

    fn observe<T: coop::Numeric + PartialEq, const N: usize>(
        values: &coop::ThreadData<T, N>,
        expected: &[T],
        offset: usize,
    ) -> bool {
        let mut matches = values.len() == N;
        let mut i = 0;
        while i < N {
            matches &= values.get(i) == Some(expected[offset + i]);
            i += 1;
        }
        matches && values.get(N).is_none() && values.get(usize::MAX).is_none()
    }

    fn invoke<T: coop::Numeric, G: coop::ExchangeGroup, const ITEMS_PER_THREAD: usize>(
        group: coop::ThreadGroup<G>,
        input: &coop::ThreadData<T, ITEMS_PER_THREAD>,
        to_striped: bool,
    ) -> coop::ThreadData<T, ITEMS_PER_THREAD> {
        if to_striped {
            coop::exchange(
                group,
                input,
                coop::ExchangeOptions::default().mode(coop::BlockedToStriped),
            )
        } else {
            coop::exchange(group, input, Default::default())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exercise<T, G, const ITEMS_PER_THREAD: usize>(
        group: coop::ThreadGroup<G>,
        input: &[T],
        expected: &[T],
        tid: usize,
        subgroup: usize,
        to_striped: bool,
        selective: bool,
        invalid: bool,
    ) -> u32
    where
        T: coop::Numeric + PartialEq + Default,
        G: coop::ExchangeGroup,
    {
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            if !(invalid && tid == 31 && i == 3) {
                values.write(i, input[tid * ITEMS_PER_THREAD + i]);
            }
            i += 1;
        }
        let before = values;
        let mut word = SKIPPED;
        if !selective || subgroup & 1 == 0 {
            let iterations = if selective {
                1 + ((subgroup / 2) & 1)
            } else {
                2
            };
            let mut values_match = true;
            let mut input_preserved = true;
            let mut independent = true;
            let mut roundtrip = true;
            let mut iteration = 0;
            while iteration < iterations {
                let mut first = invoke(group, &values, to_striped);
                values_match &= observe(&first, expected, tid * ITEMS_PER_THREAD);
                input_preserved &= preserved(&values, &before);
                let restored = invoke(group, &first, !to_striped);
                input_preserved &= observe(&first, expected, tid * ITEMS_PER_THREAD);
                roundtrip &= preserved(&restored, &before);
                // All host inputs are nonzero. Keep this altered result live
                // across subsequent exchanges to check fresh-result independence.
                first.write(0, T::default());
                let mutated = first;
                let second = invoke(group, &values, to_striped);
                values_match &= observe(&second, expected, tid * ITEMS_PER_THREAD);
                let restored = invoke(group, &second, !to_striped);
                input_preserved &= observe(&second, expected, tid * ITEMS_PER_THREAD);
                roundtrip &= preserved(&restored, &before);
                independent &= preserved(&first, &mutated) && first.get(0) == Some(T::default());
                input_preserved &= preserved(&values, &before);
                iteration += 1;
            }
            word = values_match as u32
                | ((input_preserved as u32) << 1)
                | ((independent as u32) << 2)
                | ((roundtrip as u32) << 3);
        }
        word
    }

    fn report<const BLOCK_THREADS: usize>(
        output: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        word: u32,
    ) {
        let block = coop::this_block();
        let mut value = coop::ThreadData::<_, 1>::new();
        value.write(0, word);
        coop::store(
            block,
            output,
            &value,
            coop::StoreOptions::default()
                .offset(block.rank_at::<coop::Grid>() as usize * BLOCK_THREADS),
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn block33(
        input: &[i8],
        expected: &[i8],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        to_striped: bool,
    ) {
        let tid = (thread::blockIdx_x() * 33 + thread::threadIdx_x()) as usize;
        let word = exercise::<_, _, 1>(
            coop::this_block(),
            input,
            expected,
            tid,
            0,
            to_striped,
            false,
            false,
        );
        report::<33>(&mut output, word);
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 3, 2), dynamic_shared = inferred)]
    pub fn block_xyz(
        input: &[i16],
        expected: &[i16],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        to_striped: bool,
    ) {
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 3 * thread::threadIdx_z());
        let tid = (thread::blockIdx_x() * 24 + rank) as usize;
        let word = exercise::<_, _, 4>(
            coop::this_block(),
            input,
            expected,
            tid,
            0,
            to_striped,
            false,
            false,
        );
        report::<24>(&mut output, word);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn block64(
        input: &[f64],
        expected: &[f64],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        to_striped: bool,
    ) {
        let tid = (thread::blockIdx_x() * 64 + thread::threadIdx_x()) as usize;
        let word = exercise::<_, _, 9>(
            coop::this_block(),
            input,
            expected,
            tid,
            0,
            to_striped,
            false,
            false,
        );
        report::<64>(&mut output, word);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn physical(
        input: &[f32],
        expected: &[f32],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        to_striped: bool,
        invalid: bool,
    ) {
        let tid = (thread::blockIdx_x() * 64 + thread::threadIdx_x()) as usize;
        let word = exercise::<_, _, 4>(
            coop::this_warp(),
            input,
            expected,
            tid,
            0,
            to_striped,
            false,
            invalid,
        );
        report::<64>(&mut output, word);
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn logical<T, const ITEMS_PER_THREAD: usize, const WIDTH: usize>(
        input: &[T],
        expected: &[T],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        to_striped: bool,
        selective: bool,
    ) where
        T: coop::Numeric + PartialEq + Default,
        coop::ThreadsWithinWarp<WIDTH>: coop::ExchangeGroup,
    {
        let group = coop::this_warp().group_by::<WIDTH>();
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 4 * thread::threadIdx_z());
        let tid = (thread::blockIdx_x() * 64 + rank) as usize;
        let subgroup = coop::this_warp().rank_at::<coop::Block>() as usize * (32 / WIDTH)
            + group.rank_at::<coop::Warp>() as usize;
        let word = exercise::<_, _, ITEMS_PER_THREAD>(
            group, input, expected, tid, subgroup, to_striped, selective, false,
        );
        // All selected Warp work has reconverged before this Block Store.
        report::<64>(&mut output, word);
    }
}

fn oracle<T: Copy>(input: &[T], items: usize, width: usize, to_striped: bool) -> Vec<T> {
    let mut expected = input.to_vec();
    for group in 0..input.len() / (width * items) {
        for lane in 0..width {
            for i in 0..items {
                let logical = if to_striped {
                    i * width + lane
                } else {
                    lane * items + i
                };
                let (source_lane, source_slot) = if to_striped {
                    (logical / items, logical % items)
                } else {
                    (logical % width, logical / width)
                };
                expected[(group * width + lane) * items + i] =
                    input[(group * width + source_lane) * items + source_slot];
            }
        }
    }
    expected
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    let invalid = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("invalid-input") => true,
        Some(other) => panic!("unknown test mode {other}"),
    };
    if invalid {
        let input = DeviceBuffer::from_host(&stream, &[1_f32; 128 * 4])?;
        let expected = DeviceBuffer::from_host(&stream, &[1_f32; 128 * 4])?;
        let output = DeviceBuffer::from_host(&stream, &[0_u32; 128])?;
        let plan = module.prepare_physical(LaunchConfig1D::auto_shared(2, 64))?;
        module
            .physical_async_owned(&plan, input, expected, output, false, true)
            .sync()?;
        stream.synchronize()?;
        panic!("unwritten Exchange input unexpectedly completed");
    }
    let mut cases = 0;
    let mut reports = 0;
    macro_rules! fixed {
        ($ty:ty, $items:literal, $threads:literal, $width:literal, $prepare:ident, $launch:ident, $config:expr $(, $extra:expr)?) => {{
            let input: Vec<$ty> = (1..=2 * $threads * $items).map(|i| i as $ty).collect();
            for to_striped in [false, true] {
                let expected = oracle(&input, $items, $width, to_striped);
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &[0_u32; 2 * $threads])?;
                let plan = module.$prepare($config)?;
                let (_, _, target) = module.$launch(&plan, source, expected, target, to_striped $(, $extra)?).sync()?;
                assert_eq!(target.to_host_vec(&stream)?, vec![CHECKS; 2 * $threads], "{} striped{}", stringify!($prepare), to_striped);
                cases += 1;
                reports += 2 * $threads;
            }
        }};
    }
    fixed!(
        i8,
        1,
        33,
        33,
        prepare_block33,
        block33_async_owned,
        LaunchConfig1D::auto_shared(2, 33)
    );
    fixed!(
        i16,
        4,
        24,
        24,
        prepare_block_xyz,
        block_xyz_async_owned,
        LaunchConfig3D::auto_shared((2, 1, 1), (4, 3, 2))
    );
    fixed!(
        f64,
        9,
        64,
        64,
        prepare_block64,
        block64_async_owned,
        LaunchConfig1D::auto_shared(2, 64)
    );
    fixed!(
        f32,
        4,
        64,
        32,
        prepare_physical,
        physical_async_owned,
        LaunchConfig1D::auto_shared(2, 64),
        false
    );
    macro_rules! logical {
        ($ty:ty, $items:literal, $width:literal, $selective:expr) => {{
            let input: Vec<$ty> = (1..=128 * $items).map(|i| i as $ty).collect();
            for to_striped in [false, true] {
                let expected = oracle(&input, $items, $width, to_striped);
                let words: Vec<u32> = (0..128)
                    .map(|tid| {
                        if $selective && (tid / $width) & 1 != 0 {
                            SKIPPED
                        } else {
                            CHECKS
                        }
                    })
                    .collect();
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &[0_u32; 128])?;
                let plan = module.prepare_logical::<$ty, $items, $width>(
                    LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)),
                )?;
                let (_, _, target) = module
                    .logical_async_owned::<$ty, $items, $width, _, _, _>(
                        &plan, source, expected, target, to_striped, $selective,
                    )
                    .sync()?;
                assert_eq!(
                    target.to_host_vec(&stream)?,
                    words,
                    "{} N{} W{} striped{} selective{}",
                    stringify!($ty),
                    $items,
                    $width,
                    to_striped,
                    $selective
                );
                cases += 1;
                reports += 128;
            }
        }};
    }
    logical!(i32, 8, 8, false);
    logical!(u16, 9, 1, false);
    logical!(i32, 4, 8, true);
    println!("PASS: {cases} public Exchange cases, {reports} lane reports");
    Ok(())
}
