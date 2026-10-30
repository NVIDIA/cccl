// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const SKIPPED: u32 = 1 << 31;
const CHECKS: u32 = 15 << 16;

#[cuda_module]
mod kernels {
    use super::*;

    fn preserved<T: coop::Numeric + PartialEq, const N: usize>(
        values: &coop::ThreadData<T, N>,
        before: &coop::ThreadData<T, N>,
    ) -> bool {
        let mut matches = true;
        let mut i = 0;
        while i < N {
            matches &= values.get(i) == before.get(i);
            i += 1;
        }
        matches && values.get(N).is_none() && values.get(usize::MAX).is_none()
    }

    fn observe<T: coop::Numeric + PartialEq, const K: usize>(
        values: &coop::ThreadData<T, K>,
        expected: &[T],
        offset: usize,
        lane: usize,
        width: usize,
        batches: usize,
        blocked: bool,
    ) -> (u32, bool) {
        let mut mask = 0;
        let mut matches = values.len() == K;
        let mut i = 0;
        while i < K {
            let batch = if blocked {
                lane * K + i
            } else {
                lane + i * width
            };
            let value = values.get(i);
            matches &= value.is_some() == (batch < batches);
            if let Some(value) = value {
                mask |= 1 << i;
                matches &= value == expected[offset + i];
            }
            i += 1;
        }
        (
            mask,
            matches && values.get(K).is_none() && values.get(usize::MAX).is_none(),
        )
    }

    fn invoke<T, G, Op, const ITEMS_PER_THREAD: usize>(
        group: coop::ThreadGroup<G>,
        input: &coop::ThreadData<T, ITEMS_PER_THREAD>,
        blocked: bool,
        op: Op,
    ) -> coop::ThreadData<T, { coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH) }>
    where
        T: coop::Numeric,
        G: coop::ReduceBatchedGroup,
        Op: coop::ReduceOperator<T>,
        [(); coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH)]:,
    {
        let options = coop::ReduceBatchedOptions::default().binary_op(op);
        if blocked {
            coop::reduce_batched(group, input, options.output_layout(coop::Blocked))
        } else {
            coop::reduce_batched(group, input, options)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exercise<T, G, Op, const ITEMS_PER_THREAD: usize>(
        group: coop::ThreadGroup<G>,
        input: &[T],
        expected: &[T],
        tid: usize,
        subgroup: usize,
        blocked: bool,
        selective: bool,
        invalid: bool,
        op: Op,
    ) -> u32
    where
        T: coop::Numeric + PartialEq,
        G: coop::ReduceBatchedGroup,
        Op: coop::ReduceOperator<T> + Copy,
        [(); coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH)]:,
    {
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            // The isolated trap leaves one non-output-owner's input unwritten.
            if !(invalid && tid == 31 && i == 3) {
                values.write(i, input[tid * ITEMS_PER_THREAD + i]);
            }
            i += 1;
        }
        let before = values;
        let mut word = SKIPPED;
        if !selective || subgroup & 1 == 0 {
            let capacity = coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH);
            let lane = tid % G::WIDTH;
            let iterations = if selective {
                1 + ((subgroup / 2) & 1)
            } else {
                2
            };
            let mut values_match = true;
            let mut input_preserved = true;
            let mut independent = true;
            let mut masks_match = true;
            let mut mask = 0;
            let mut iteration = 0;
            while iteration < iterations {
                let mut first = invoke(group, &values, blocked, op);
                let (first_mask, first_ok) = observe(
                    &first,
                    expected,
                    tid * capacity,
                    lane,
                    G::WIDTH,
                    ITEMS_PER_THREAD,
                    blocked,
                );
                values_match &= first_ok;
                input_preserved &= preserved(&values, &before);
                // This also initializes the last slot on lanes where it was absent.
                first.write(capacity - 1, input[tid * ITEMS_PER_THREAD]);
                let mutated = first;
                let second = invoke(group, &values, blocked, op);
                let (second_mask, second_ok) = observe(
                    &second,
                    expected,
                    tid * capacity,
                    lane,
                    G::WIDTH,
                    ITEMS_PER_THREAD,
                    blocked,
                );
                values_match &= second_ok;
                masks_match &= first_mask == second_mask;
                independent &= preserved(&first, &mutated);
                input_preserved &= preserved(&values, &before);
                mask = second_mask;
                iteration += 1;
            }
            word = mask
                | ((values_match as u32) << 16)
                | ((input_preserved as u32) << 17)
                | ((independent as u32) << 18)
                | ((masks_match as u32) << 19);
        }
        word
    }

    fn report(output: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>, word: u32) {
        let block = coop::this_block();
        let mut value = coop::ThreadData::<_, 1>::new();
        value.write(0, word);
        // All Warp branches have reconverged; Block Store owns the XYZ output.
        coop::store(
            block,
            output,
            &value,
            coop::StoreOptions::default().offset(block.rank_at::<coop::Grid>() as usize * 64),
        );
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn physical<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        expected: &[i32],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        blocked: bool,
        invalid: bool,
    ) where
        [(); coop::batched_extent(ITEMS_PER_THREAD, 32)]:,
    {
        let warp = coop::this_warp();
        let tid = (thread::blockIdx_x() * 64 + thread::threadIdx_x()) as usize;
        let word = exercise::<_, _, _, ITEMS_PER_THREAD>(
            warp,
            input,
            expected,
            tid,
            warp.rank_at::<coop::Block>() as usize,
            blocked,
            false,
            invalid,
            coop::Sum,
        );
        report(&mut output, word);
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn logical<T, Op, const ITEMS_PER_THREAD: usize, const WIDTH: usize>(
        input: &[T],
        expected: &[T],
        mut output: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        blocked: bool,
        selective: bool,
    ) where
        T: coop::Numeric + PartialEq,
        Op: coop::ReduceOperator<T> + Default + Copy,
        coop::ThreadsWithinWarp<WIDTH>: coop::ReduceBatchedGroup,
        [(); coop::batched_extent(
            ITEMS_PER_THREAD,
            <coop::ThreadsWithinWarp<WIDTH> as coop::ReduceBatchedGroup>::WIDTH,
        )]:,
    {
        let group = coop::this_warp().group_by::<WIDTH>();
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 4 * thread::threadIdx_z());
        let tid = (thread::blockIdx_x() * 64 + rank) as usize;
        let word = exercise::<_, _, _, ITEMS_PER_THREAD>(
            group,
            input,
            expected,
            tid,
            coop::this_warp().rank_at::<coop::Block>() as usize * (32 / WIDTH)
                + group.rank_at::<coop::Warp>() as usize,
            blocked,
            selective,
            false,
            Op::default(),
        );
        report(&mut output, word);
    }
}

fn oracle<T: Copy>(
    input: &[T],
    batches: usize,
    width: usize,
    blocked: bool,
    selective: bool,
    reduce: impl Fn(T, T) -> T,
) -> (Vec<T>, Vec<u32>) {
    let capacity = batches.div_ceil(width);
    let mut expected = vec![input[0]; 128 * capacity];
    let mut words = vec![CHECKS; 128];
    for tid in 0..128 {
        let lane = tid % width;
        let group = tid / width;
        if selective && group & 1 != 0 {
            words[tid] = SKIPPED;
        }
        for i in 0..capacity {
            let batch = if blocked {
                lane * capacity + i
            } else {
                lane + i * width
            };
            if batch < batches {
                expected[tid * capacity + i] = (0..width)
                    .map(|lane| input[(group * width + lane) * batches + batch])
                    .reduce(&reduce)
                    .unwrap();
                if words[tid] != SKIPPED {
                    words[tid] |= 1 << i;
                }
            }
        }
    }
    (expected, words)
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
        let input = DeviceBuffer::from_host(&stream, &[1_i32; 128 * 4])?;
        let expected = DeviceBuffer::from_host(&stream, &[32_i32; 128])?;
        let output = DeviceBuffer::from_host(&stream, &[0_u32; 128])?;
        let plan = module.prepare_physical::<4>(LaunchConfig1D::auto_shared(2, 64))?;
        module
            .physical_async_owned::<4, _, _, _>(&plan, input, expected, output, false, true)
            .sync()?;
        stream.synchronize()?;
        panic!("unwritten batched input unexpectedly completed");
    }
    let mut cases = 0;
    macro_rules! physical {
        ($items:literal) => {{
            let input: Vec<i32> = (0..128)
                .flat_map(|tid| {
                    (0..$items).map(move |batch| {
                        (10000 * (tid / 64) + 1000 * (tid / 32) + 17 * batch + tid % 32) as i32
                    })
                })
                .collect();
            for blocked in [false, true] {
                let (expected, words) = oracle(&input, $items, 32, blocked, false, |a, b| a + b);
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &[0_u32; 128])?;
                let plan = module.prepare_physical::<$items>(LaunchConfig1D::auto_shared(2, 64))?;
                let (_, _, target) = module
                    .physical_async_owned::<$items, _, _, _>(
                        &plan, source, expected, target, blocked, false,
                    )
                    .sync()?;
                assert_eq!(
                    target.to_host_vec(&stream)?,
                    words,
                    "physical B{} blocked{}",
                    $items,
                    blocked
                );
                cases += 1;
            }
        }};
    }
    physical!(1);
    physical!(4);
    macro_rules! logical {
        ($ty:ty, $op:ty, $items:literal, $width:literal, $value:expr, $reduce:expr, $selective:expr) => {{
            let value = $value;
            let input: Vec<$ty> = (0..128)
                .flat_map(|tid| (0..$items).map(move |batch| value(tid, batch)))
                .collect();
            for blocked in [false, true] {
                let (expected, words) =
                    oracle(&input, $items, $width, blocked, $selective, $reduce);
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let expected = DeviceBuffer::from_host(&stream, &expected)?;
                let target = DeviceBuffer::from_host(&stream, &[0_u32; 128])?;
                let plan = module.prepare_logical::<$ty, $op, $items, $width>(
                    LaunchConfig3D::auto_shared((2, 1, 1), (4, 4, 4)),
                )?;
                let (_, _, target) = module
                    .logical_async_owned::<$ty, $op, $items, $width, _, _, _>(
                        &plan, source, expected, target, blocked, $selective,
                    )
                    .sync()?;
                assert_eq!(
                    target.to_host_vec(&stream)?,
                    words,
                    "{} B{} W{} blocked{} selective{}",
                    stringify!($ty),
                    $items,
                    $width,
                    blocked,
                    $selective
                );
                cases += 1;
            }
        }};
    }
    for selective in [false, true] {
        logical!(
            i8,
            coop::Min,
            10,
            8,
            |tid: usize, batch: usize| -112_i8 + (6 * (tid / 8) + batch + tid % 8) as i8,
            i8::min,
            selective
        );
    }
    logical!(
        f64,
        coop::Max,
        10,
        8,
        |tid: usize, batch: usize| (1000 * (tid / 64) + 100 * (tid / 8) + 7 * batch) as f64
            + (tid % 8) as f64 / 8.0,
        f64::max,
        false
    );
    logical!(
        u16,
        coop::Sum,
        9,
        1,
        |tid: usize, batch: usize| (tid * 17 + batch + 1) as u16,
        |a, b| a + b,
        false
    );
    logical!(
        u32,
        coop::BitXor,
        65,
        8,
        |tid: usize, batch: usize| 1_u32 << ((batch + 3 * (tid % 8) + tid / 8) % 31),
        |a, b| a ^ b,
        false
    );
    println!(
        "PASS: {cases} public batched Reduce cases, {} lane reports",
        cases * 128
    );
    Ok(())
}
