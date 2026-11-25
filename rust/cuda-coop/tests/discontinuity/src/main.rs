// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

const CHECKS: u32 = 15;

#[cuda_module]
mod kernels {
    use super::*;

    fn observe<const N: usize, const A: usize>(
        flags: &coop::ThreadData<i32, N, A>,
        expected: &[i32],
        offset: usize,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut valid = flags.len() == N;
        let mut i = 0;
        while i < N {
            valid &= flags.get(i) == Some(expected[offset + i]);
            i += 1;
        }
        valid
    }

    fn retained<const N: usize, const A: usize>(
        flags: &coop::ThreadData<i32, N, A>,
        expected: &[i32],
        offset: usize,
        slot: usize,
        marker: i32,
    ) -> bool
    where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let mut valid = flags.len() == N;
        let mut i = 0;
        while i < N {
            let value = if i == slot {
                marker
            } else {
                expected[offset + i]
            };
            valid &= flags.get(i) == Some(value);
            i += 1;
        }
        valid
    }

    #[allow(clippy::too_many_arguments)]
    fn exercise<T: coop::Numeric, const N: usize, const A: usize, const B: usize, const FORM: u8>(
        input: &[T],
        expected: &[i32],
        heads_output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<N>>,
        tails_output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<N>>,
        preserved: &mut coop::DisjointSlice<T, coop::CollectiveTiles<N>>,
        reports: &mut coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        rank: usize,
        mode: u32,
        predecessor: T,
        successor: T,
        replacement: T,
        invalid: u32,
    ) where
        coop::Alignment<A>: coop::SupportedAlignment,
    {
        let block = coop::this_block();
        let block_index = block.rank_at::<coop::Grid>() as usize;
        let offset = block_index * B * N;
        let local = offset + rank * N;
        let len = input.len();
        let mut values = coop::ThreadData::<_, N, A>::new();
        if FORM == 8 && invalid != 0 {
            let mut i = 0;
            while i < N {
                if !(rank == B - 1 && i == N - 1) {
                    values.write(i, input[local + i]);
                }
                i += 1;
            }
            if invalid == 1 {
                let _ = coop::discontinuity(block, &values, Default::default());
            } else {
                let _ = coop::discontinuity(
                    block,
                    &values,
                    coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails),
                );
            }
            // A lost preflight must reach the host's unexpected-return panic.
            // No result read or value-dependent device assertion can mimic it.
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
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let mut changed = values;
        changed.write(0, replacement);
        let mut word: u32;
        macro_rules! single {
            ($options:expr, $plane:expr) => {{
                let plane = $plane * len;
                let mut first = coop::discontinuity(block, &values, $options);
                word = observe(&first, expected, plane + local) as u32;
                first.write(0, 7);
                let second = coop::discontinuity(block, &changed, $options);
                word |= (observe(&second, expected, 2 * len + plane + local) as u32) << 1;
                word |= (retained(&first, expected, plane + local, 0, 7) as u32) << 2;
                word |= 8;
                if $plane == 0 {
                    coop::store(
                        block,
                        heads_output,
                        &second,
                        coop::StoreOptions::default().offset(offset),
                    );
                } else {
                    coop::store(
                        block,
                        tails_output,
                        &second,
                        coop::StoreOptions::default().offset(offset),
                    );
                }
            }};
        }
        macro_rules! pair {
            ($options:expr, $manual:expr) => {{
                let (mut heads, mut tails) = coop::discontinuity(block, &values, $options);
                word = (observe(&heads, expected, local) && observe(&tails, expected, len + local))
                    as u32;
                heads.write(0, 7);
                let separate = observe(&tails, expected, len + local);
                tails.write(N - 1, 9);
                if $manual {
                    thread::sync_threads();
                }
                let (next_heads, next_tails) = coop::discontinuity(block, &changed, $options);
                word |= ((observe(&next_heads, expected, 2 * len + local)
                    && observe(&next_tails, expected, 3 * len + local)) as u32)
                    << 1;
                word |= ((retained(&heads, expected, local, 0, 7)
                    && retained(&tails, expected, len + local, N - 1, 9)) as u32)
                    << 2;
                word |= (separate as u32) << 3;
                if $manual {
                    thread::sync_threads();
                }
                coop::store(
                    block,
                    heads_output,
                    &next_heads,
                    coop::StoreOptions::default().offset(offset),
                );
                coop::store(
                    block,
                    tails_output,
                    &next_tails,
                    coop::StoreOptions::default().offset(offset),
                );
            }};
        }
        if FORM == 9 {
            let mut scratch = coop::TempStorage::new();
            pair!(
                coop::DiscontinuityOptions::default()
                    .mode(coop::HeadsAndTails)
                    .tile_predecessor_item(predecessor)
                    .tile_successor_item(successor)
                    .temp_storage(&mut scratch),
                true
            );
        } else if FORM == 10 {
            let mut scratch = coop::TempStorage::new().auto_sync();
            pair!(
                coop::DiscontinuityOptions::default()
                    .mode(coop::HeadsAndTails)
                    .tile_predecessor_item(predecessor)
                    .tile_successor_item(successor)
                    .temp_storage(&mut scratch),
                false
            );
        } else {
            let recipe = if FORM == 8 { mode } else { FORM as u32 };
            match recipe {
                0 => single!(Default::default(), 0),
                1 => single!(
                    coop::DiscontinuityOptions::default().tile_predecessor_item(predecessor),
                    0
                ),
                2 => single!(coop::DiscontinuityOptions::default().mode(coop::Tails), 1),
                3 => single!(
                    coop::DiscontinuityOptions::default()
                        .mode(coop::Tails)
                        .tile_successor_item(successor),
                    1
                ),
                4 => pair!(
                    coop::DiscontinuityOptions::default().mode(coop::HeadsAndTails),
                    false
                ),
                5 => pair!(
                    coop::DiscontinuityOptions::default()
                        .mode(coop::HeadsAndTails)
                        .tile_predecessor_item(predecessor),
                    false
                ),
                6 => pair!(
                    coop::DiscontinuityOptions::default()
                        .mode(coop::HeadsAndTails)
                        .tile_successor_item(successor),
                    false
                ),
                7 => pair!(
                    coop::DiscontinuityOptions::default()
                        .mode(coop::HeadsAndTails)
                        .tile_predecessor_item(predecessor)
                        .tile_successor_item(successor),
                    false
                ),
                _ => panic!(),
            }
        }
        // Store original numeric bytes, including NaN payloads and zero signs.
        coop::store(
            block,
            preserved,
            &values,
            coop::StoreOptions::default().offset(offset),
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
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn forms_i32(
        input: &[i32],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i32,
        successor: i32,
        replacement: i32,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 4, 16, 33, 8>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 3, block = (4, 3, 2), dynamic_shared = inferred)]
    pub fn xyz_f64(
        input: &[f64],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<f64, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: f64,
        successor: f64,
        replacement: f64,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 3 * thread::threadIdx_z());
        exercise::<_, 4, 1, 24, 7>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn tail_f32(
        input: &[f32],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut preserved: coop::DisjointSlice<f32, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: f32,
        successor: f32,
        replacement: f32,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 9, 1, 32, 4>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn heads_i8(
        input: &[i8],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<i8, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i8,
        successor: i8,
        replacement: i8,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 1, 1, 33, 1>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_u8(
        input: &[u8],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<u8, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: u8,
        successor: u8,
        replacement: u8,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 1, 1, 1, 4>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 3, block = (4, 3, 2), dynamic_shared = inferred)]
    pub fn xyz_i16(
        input: &[i16],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<i16, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i16,
        successor: i16,
        replacement: i16,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 3 * thread::threadIdx_z());
        exercise::<_, 4, 1, 24, 3>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn tail_u16(
        input: &[u16],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<9>>,
        mut preserved: coop::DisjointSlice<u16, coop::CollectiveTiles<9>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: u16,
        successor: u16,
        replacement: u16,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 9, 1, 32, 5>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (1, 1, 1), dynamic_shared = inferred)]
    pub fn single_u32(
        input: &[u32],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<u32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: u32,
        successor: u32,
        replacement: u32,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 4, 1, 1, 7>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn heads_i64(
        input: &[i64],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        mut preserved: coop::DisjointSlice<i64, coop::CollectiveTiles<1>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i64,
        successor: i64,
        replacement: i64,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 1, 1, 32, 0>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn tails_u64(
        input: &[u64],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<u64, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: u64,
        successor: u64,
        replacement: u64,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 4, 1, 33, 2>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn manual_storage(
        input: &[i32],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i32,
        successor: i32,
        replacement: i32,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 4, 16, 33, 9>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn auto_storage(
        input: &[i32],
        expected: &[i32],
        mut heads_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut tails_output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut preserved: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mut reports: coop::DisjointSlice<u32, coop::CollectiveTiles<1>>,
        mode: u32,
        predecessor: i32,
        successor: i32,
        replacement: i32,
        invalid: u32,
    ) {
        let rank = thread::threadIdx_x();
        exercise::<_, 4, 16, 33, 10>(
            input,
            expected,
            &mut heads_output,
            &mut tails_output,
            &mut preserved,
            &mut reports,
            rank as usize,
            mode,
            predecessor,
            successor,
            replacement,
            invalid,
        );
    }
}

fn oracle<T: Copy + PartialEq>(
    input: &[T],
    tile: usize,
    recipe: u32,
    pred: T,
    succ: T,
) -> Vec<i32> {
    let mut flags = vec![-17; 2 * input.len()];
    for (block, values) in input.chunks_exact(tile).enumerate() {
        for i in 0..tile {
            if !(2..4).contains(&recipe) {
                flags[block * tile + i] = if i != 0 {
                    values[i - 1] != values[i]
                } else if matches!(recipe, 1 | 5 | 7) {
                    pred != values[i]
                } else {
                    true
                } as i32;
            }
            if recipe >= 2 {
                flags[input.len() + block * tile + i] = if i + 1 != tile {
                    values[i] != values[i + 1]
                } else if matches!(recipe, 3 | 6 | 7) {
                    values[i] != succ
                } else {
                    true
                } as i32;
            }
        }
    }
    flags
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};
    let invalid = match std::env::args().nth(1).as_deref() {
        None => 0,
        Some("missing-head-input") => 1,
        Some("missing-pair-input") => 2,
        Some(other) => panic!("unknown test mode {other}"),
    };
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: typed declarations and embedded kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    if invalid != 0 {
        let source = DeviceBuffer::from_host(&stream, &[3_i32; 132])?;
        let expected = DeviceBuffer::from_host(&stream, &[0_i32; 528])?;
        let heads = DeviceBuffer::from_host(&stream, &[-17_i32; 132])?;
        let tails = DeviceBuffer::from_host(&stream, &[-17_i32; 132])?;
        let preserved = DeviceBuffer::from_host(&stream, &[0_i32; 132])?;
        let report = DeviceBuffer::from_host(&stream, &[0_u32; 33])?;
        let plan = module.prepare_forms_i32(LaunchConfig1D::auto_shared(1, 33))?;
        module
            .forms_i32_async_owned(
                &plan, source, expected, heads, tails, preserved, report, 0, 3, 3, 4, invalid,
            )
            .sync()?;
        stream.synchronize()?;
        panic!("invalid Discontinuity mode {invalid} unexpectedly completed");
    }
    let mut cases = 0;
    let mut flags = 0;
    let mut values = 0;
    let mut reports = 0;
    macro_rules! run {
        ($ty:ty, $n:literal, $b:literal, $prepare:ident, $launch:ident, $config:expr,
         $recipe:expr, $input:expr, $pred:expr, $succ:expr, $replacement:expr, $bits:expr) => {{
            let input: Vec<$ty> = $input;
            let recipe = $recipe;
            let (pred, succ, replacement): ($ty, $ty, $ty) = ($pred, $succ, $replacement);
            let mut changed = input.clone();
            for slot in changed.iter_mut().step_by($n) {
                *slot = replacement;
            }
            let mut expected = oracle(&input, $b * $n, recipe, pred, succ);
            let second = oracle(&changed, $b * $n, recipe, pred, succ);
            expected.extend_from_slice(&second);
            let source = DeviceBuffer::from_host(&stream, &input)?;
            let expected_device = DeviceBuffer::from_host(&stream, &expected)?;
            let heads = DeviceBuffer::from_host(&stream, &vec![-17_i32; input.len()])?;
            let tails = DeviceBuffer::from_host(&stream, &vec![-17_i32; input.len()])?;
            let preserved = DeviceBuffer::from_host(&stream, &vec![0 as $ty; input.len()])?;
            let words = DeviceBuffer::from_host(&stream, &[0_u32; 3 * $b])?;
            let plan = module.$prepare($config)?;
            let (source, _, heads, tails, preserved, words) = module
                .$launch(
                    &plan,
                    source,
                    expected_device,
                    heads,
                    tails,
                    preserved,
                    words,
                    recipe,
                    pred,
                    succ,
                    replacement,
                    0,
                )
                .sync()?;
            assert_eq!(
                [heads.to_host_vec(&stream)?, tails.to_host_vec(&stream)?].concat(),
                second,
                "{} recipe{recipe}",
                stringify!($prepare)
            );
            let bit_pattern = $bits;
            let original_bits: Vec<_> = input.iter().copied().map(bit_pattern).collect();
            assert_eq!(
                source
                    .to_host_vec(&stream)?
                    .into_iter()
                    .map(bit_pattern)
                    .collect::<Vec<_>>(),
                original_bits
            );
            assert_eq!(
                preserved
                    .to_host_vec(&stream)?
                    .into_iter()
                    .map(bit_pattern)
                    .collect::<Vec<_>>(),
                original_bits
            );
            assert_eq!(
                words.to_host_vec(&stream)?,
                vec![CHECKS; 3 * $b],
                "{} recipe{recipe}",
                stringify!($prepare)
            );
            cases += 1;
            flags += input.len() * if recipe < 4 { 1 } else { 2 };
            values += input.len();
            reports += 3 * $b;
        }};
    }
    for recipe in 0..8 {
        run!(
            i32,
            4,
            33,
            prepare_forms_i32,
            forms_i32_async_owned,
            LaunchConfig1D::auto_shared(3, 33),
            recipe,
            (0..3 * 33 * 4)
                .map(|i| (i / (33 * 4) + (i % (33 * 4)) / 3) % 5)
                .collect(),
            0,
            3,
            11,
            |x: i32| x
        );
    }
    run!(
        f64,
        4,
        24,
        prepare_xyz_f64,
        xyz_f64_async_owned,
        LaunchConfig3D::auto_shared((3, 1, 1), (4, 3, 2)),
        7,
        (0..288)
            .map(|i| 1.0 + ((i / 3) % 5) as f64 * 1.0e-10)
            .collect(),
        1.0,
        1.0,
        2.0,
        f64::to_bits
    );
    run!(
        f64,
        4,
        24,
        prepare_xyz_f64,
        xyz_f64_async_owned,
        LaunchConfig3D::auto_shared((3, 1, 1), (4, 3, 2)),
        7,
        (0..288)
            .map(|i| [
                f64::from_bits(0x7ff8_0000_0000_0042),
                f64::NAN,
                2.0,
                0.0,
                -0.0,
                2.0,
                2.0,
                3.0
            ][i % 8])
            .collect(),
        f64::NAN,
        -0.0,
        4.0,
        f64::to_bits
    );
    run!(
        f32,
        9,
        32,
        prepare_tail_f32,
        tail_f32_async_owned,
        LaunchConfig1D::auto_shared(3, 32),
        4,
        (0..864)
            .map(|i| [
                f32::from_bits(0x7fc0_0042),
                f32::NAN,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                2.0,
                0.0,
                -0.0
            ][i % 10])
            .collect(),
        0.0,
        0.0,
        4.0,
        f32::to_bits
    );
    run!(
        i8,
        1,
        33,
        prepare_heads_i8,
        heads_i8_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        1,
        (0..99).map(|i| -7 + ((i / 3) % 5) as i8).collect(),
        -7,
        -3,
        -11,
        |x: i8| x
    );
    run!(
        u8,
        1,
        1,
        prepare_single_u8,
        single_u8_async_owned,
        LaunchConfig1D::auto_shared(3, 1),
        4,
        vec![2, 3, 4],
        2,
        4,
        7,
        |x: u8| x
    );
    run!(
        i16,
        4,
        24,
        prepare_xyz_i16,
        xyz_i16_async_owned,
        LaunchConfig3D::auto_shared((3, 1, 1), (4, 3, 2)),
        3,
        (0..3 * 24 * 4)
            .map(|i| ((i / (24 * 4) + (i % (24 * 4)) / 3) % 5) as i16)
            .collect(),
        0,
        1,
        17,
        |x: i16| x
    );
    run!(
        u16,
        9,
        32,
        prepare_tail_u16,
        tail_u16_async_owned,
        LaunchConfig1D::auto_shared(3, 32),
        5,
        (0..3 * 32 * 9)
            .map(|i| ((i / (32 * 9) + (i % (32 * 9)) / 3) % 5) as u16)
            .collect(),
        0,
        0,
        17,
        |x: u16| x
    );
    run!(
        u32,
        4,
        1,
        prepare_single_u32,
        single_u32_async_owned,
        LaunchConfig1D::auto_shared(3, 1),
        7,
        vec![2, 2, 3, 3, 2, 4, 4, 3, 5, 5, 5, 5],
        2,
        3,
        7,
        |x: u32| x
    );
    run!(
        i64,
        1,
        32,
        prepare_heads_i64,
        heads_i64_async_owned,
        LaunchConfig1D::auto_shared(3, 32),
        0,
        (0..96)
            .map(|i| (1_i64 << 40) + ((i / 3) % 3) as i64 * (1_i64 << 32))
            .collect(),
        0,
        0,
        -1,
        |x: i64| x
    );
    run!(
        u64,
        4,
        33,
        prepare_tails_u64,
        tails_u64_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        2,
        (0..396)
            .map(|i| (1_u64 << 40) + ((i / 3) % 3) as u64 * (1_u64 << 32))
            .collect(),
        0,
        0,
        1,
        |x: u64| x
    );
    run!(
        i32,
        4,
        33,
        prepare_manual_storage,
        manual_storage_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        7,
        (0..3 * 33 * 4)
            .map(|i| (i / (33 * 4) + (i % (33 * 4)) / 3) % 5)
            .collect(),
        0,
        3,
        11,
        |x: i32| x
    );
    run!(
        i32,
        4,
        33,
        prepare_auto_storage,
        auto_storage_async_owned,
        LaunchConfig1D::auto_shared(3, 33),
        7,
        (0..3 * 33 * 4)
            .map(|i| (i / (33 * 4) + (i % (33 * 4)) / 3) % 5)
            .collect(),
        0,
        3,
        11,
        |x: i32| x
    );
    assert_eq!(cases, 20);
    println!(
        "discontinuity: {cases} cases, {flags} flags, {values} preserved values, {reports} lane reports passed"
    );
    Ok(())
}
