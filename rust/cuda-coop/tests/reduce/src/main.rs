// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use cuda_coop as coop;
use cuda_device::{kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn preserved<T: coop::Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        values: &coop::ThreadData<T, ITEMS_PER_THREAD>,
        before: &coop::ThreadData<T, ITEMS_PER_THREAD>,
    ) -> bool {
        let mut matches = true;
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            matches &= values.get(i) == before.get(i);
            i += 1;
        }
        matches && values.get(ITEMS_PER_THREAD).is_none()
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn payload_sum<T: coop::Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        input: &[T],
        mut output: coop::DisjointSlice<T>,
        valid: usize,
        zero: T,
        invalid: bool,
    ) {
        let block = coop::this_block();
        let offset = block.rank_at::<coop::Grid>() as usize * 33 * ITEMS_PER_THREAD;
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        let options = coop::LoadOptions::default()
            .offset(offset)
            .valid_items(valid);
        if invalid {
            coop::load(block, input, &mut values, options);
        } else {
            coop::load(block, input, &mut values, options.oob_default(zero));
        }
        let before = values;
        let total = coop::sum(block, &values, Default::default());
        assert!(preserved(&values, &before));
        assert!(total.is_some() == (thread::threadIdx_x() == 0));
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = total.unwrap_or(zero);
        }
    }

    fn with_operator<Op: coop::ReduceOperator<i32>>(
        block: coop::ThreadGroup<coop::Block>,
        values: &coop::ThreadData<i32, 4>,
        op: Op,
    ) -> Option<i32> {
        coop::reduce(block, values, coop::ReduceOptions::default().binary_op(op))
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 2), dynamic_shared = inferred)]
    pub fn operators(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, thread::BlockIndex>,
        mode: u32,
    ) {
        let block = coop::this_block();
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(block, input, &mut values, Default::default());
        let before = values;
        let total = match mode {
            0 => coop::reduce(block, &mut values, Default::default()),
            1 => with_operator(block, &values, coop::Multiplies),
            2 => with_operator(block, &values, coop::Min),
            3 => with_operator(block, &values, coop::Max),
            4 => with_operator(block, &values, coop::BitAnd),
            5 => with_operator(block, &values, coop::BitOr),
            6 => with_operator(block, &values, coop::BitXor),
            7 => coop::sum(
                block,
                &values,
                coop::SumOptions::default().algorithm(coop::RakingCommutativeOnly),
            ),
            8 => coop::sum(
                block,
                &values,
                coop::SumOptions::default().algorithm(coop::Raking),
            ),
            _ => coop::sum(
                block,
                &values,
                coop::SumOptions::default().algorithm(coop::WarpReductions),
            ),
        };
        let leader =
            thread::threadIdx_x() == 0 && thread::threadIdx_y() == 0 && thread::threadIdx_z() == 0;
        assert!(preserved(&values, &before) && total.is_some() == leader);
        if let Some(total) = total
            && let Some(index) = thread::block_index()
            && let Some(slot) = output.get_mut(index)
        {
            *slot = total;
        }
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 2), dynamic_shared = inferred)]
    pub fn scalar_count(
        mut output: coop::DisjointSlice<i32, thread::BlockIndex>,
        valid: usize,
        counted: bool,
    ) {
        let block = coop::this_block();
        let rank = thread::threadIdx_x() + 4 * (thread::threadIdx_y() + 4 * thread::threadIdx_z());
        let value = rank as i32 + 1;
        let total = if counted {
            coop::sum(block, value, coop::SumOptions::default().valid_items(valid))
        } else {
            coop::reduce(block, value, Default::default())
        };
        assert!(total.is_some() == (rank == 0));
        if let Some(total) = total
            && let Some(index) = thread::block_index()
            && let Some(slot) = output.get_mut(index)
        {
            *slot = total;
        }
    }

    fn stored<const AUTO: bool>(
        input: &[i32],
        scratch: &mut coop::TempStorage<0, 0, AUTO, false>,
    ) -> Option<i32> {
        let block = coop::this_block();
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(block, input, &mut values, Default::default());
        let before = values;
        let mut matches = true;
        let mut last = None;
        let mut iteration = 0;
        while iteration < 2 {
            let total = coop::sum(
                block,
                &values,
                coop::SumOptions::default().temp_storage(scratch),
            );
            matches &=
                preserved(&values, &before) && total.is_some() == (thread::threadIdx_x() == 0);
            // Mutating a copied result cannot change the borrowed input.
            let mut copied = total;
            if let Some(value) = &mut copied {
                *value += 1;
            }
            matches &= preserved(&values, &before);
            if !AUTO {
                thread::sync_threads();
            }
            last = copied;
            iteration += 1;
        }
        assert!(matches);
        last
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (32, 1, 1), dynamic_shared = inferred)]
    pub fn storage(input: &[i32], mut output: coop::DisjointSlice<i32>, mode: u32) {
        let total = match mode {
            0 => {
                let block = coop::this_block();
                let mut values = coop::ThreadData::<_, 4>::new();
                coop::load(block, input, &mut values, Default::default());
                let before = values;
                let first = coop::sum(block, &values, Default::default());
                let second = coop::sum(block, &values, Default::default());
                assert!(first == second && preserved(&values, &before));
                second
            }
            1 => stored(input, &mut coop::TempStorage::new()),
            _ => stored(input, &mut coop::TempStorage::new().auto_sync()),
        };
        assert!(total.is_some() == (thread::threadIdx_x() == 0));
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = total.unwrap_or(-1);
        }
    }

    fn warp_payload<G: coop::ReduceGroup<(), ()>, const ITEMS_PER_THREAD: usize>(
        group: coop::ThreadGroup<G>,
        value: i32,
    ) -> Option<i32> {
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            values.write(i, value * (i as i32 + 1));
            i += 1;
        }
        let before = values;
        let total = coop::sum(group, &values, Default::default());
        assert!(preserved(&values, &before));
        total
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn logical<const ITEMS_PER_THREAD: usize>(mut output: coop::DisjointSlice<i32>) {
        let group = coop::this_warp().group_by::<8>();
        let tid = thread::threadIdx_x();
        let total = warp_payload::<_, ITEMS_PER_THREAD>(group, (tid / 8) as i32 + 1);
        assert!(total.is_some() == (tid.is_multiple_of(8)));
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = total.unwrap_or(-1);
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn physical(mut output: coop::DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let value = (tid / 32) as i32 + 1;
        let total = coop::sum(coop::this_warp(), value, Default::default());
        assert!(total.is_some() == (tid.is_multiple_of(32)));
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = total.unwrap_or(-1);
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn prefix17(mut output: coop::DisjointSlice<i32>) {
        let group = coop::this_warp().group_by_non_exhaustive::<17>();
        let tid = thread::threadIdx_x();
        let mut observed = -2;
        if group.is_member() == 1 {
            let total = warp_payload::<_, 1>(group, (tid / 32) as i32 + 1);
            assert!(total.is_some() == (tid.is_multiple_of(32)));
            observed = total.unwrap_or(-1);
        }
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = observed;
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn prefix23(mut output: coop::DisjointSlice<i32>) {
        let group = coop::this_warp().group_by_non_exhaustive::<23>();
        let tid = thread::threadIdx_x();
        let mut observed = -2;
        if group.is_member() == 1 {
            let total = warp_payload::<_, 4>(group, (tid / 32) as i32 + 1);
            assert!(total.is_some() == (tid.is_multiple_of(32)));
            observed = total.unwrap_or(-1);
        }
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = observed;
        }
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn prefix31(mut output: coop::DisjointSlice<i32>, valid: usize) {
        let group = coop::this_warp().group_by_non_exhaustive::<31>();
        let tid = thread::threadIdx_x();
        let mut observed = -2;
        if group.is_member() == 1 {
            let value = (tid / 32 + 1) as i32 * (tid % 32 + 1) as i32;
            let total = coop::sum(group, value, coop::SumOptions::default().valid_items(valid));
            assert!(total.is_some() == (tid.is_multiple_of(32)));
            observed = total.unwrap_or(-1);
        }
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = observed;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D, LaunchConfig3D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and embedded kernels compile together.
    let module = unsafe { kernels::load_async(0)? };
    let mode = std::env::args().nth(1);
    if mode.as_deref() == Some("invalid-input") {
        let plan = module.prepare_payload_sum::<i32, 4>(LaunchConfig1D::auto_shared(1, 33))?;
        let input = DeviceBuffer::from_host(&stream, &[1_i32; 132])?;
        let output = DeviceBuffer::from_host(&stream, &[0_i32; 33])?;
        module
            .payload_sum_async_owned::<i32, 4, _, _>(&plan, input, output, 131, 0, true)
            .sync()?;
        stream.synchronize()?;
        panic!("uninitialized Reduce input unexpectedly completed");
    }
    if let Some(count) = match mode.as_deref() {
        Some("count-zero") => Some(0),
        Some("count-high") => Some(33),
        Some("count-max") => Some(usize::MAX),
        _ => None,
    } {
        let plan =
            module.prepare_scalar_count(LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 2)))?;
        let output = DeviceBuffer::from_host(&stream, &[0_i32])?;
        module
            .scalar_count_async_owned(&plan, output, count, true)
            .sync()?;
        stream.synchronize()?;
        panic!("invalid scalar Reduce count unexpectedly completed");
    }
    let mut cases = 0;
    macro_rules! typed {
        ($ty:ty, $items:literal) => {{
            let tile: usize = 33 * $items;
            let input: Vec<$ty> = (0..2 * tile)
                .map(|i| {
                    (i.is_multiple_of($items) && (i / tile == 0 || (i / $items).is_multiple_of(2)))
                        as u8 as $ty
                })
                .collect();
            let plan =
                module.prepare_payload_sum::<$ty, $items>(LaunchConfig1D::auto_shared(2, 33))?;
            for valid in [tile - 1, tile] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &[0 as $ty; 66])?;
                let (_, target) = module
                    .payload_sum_async_owned::<$ty, $items, _, _>(
                        &plan, source, target, valid, 0 as $ty, false,
                    )
                    .sync()?;
                let actual = target.to_host_vec(&stream)?;
                for (block, chunk) in input.chunks(tile).enumerate() {
                    let mut total = 0 as $ty;
                    for value in &chunk[..valid] {
                        total += *value;
                    }
                    for lane in 0..33 {
                        assert_eq!(
                            actual[block * 33 + lane],
                            if lane == 0 { total } else { 0 as $ty },
                            "{} N{} valid{} block{} lane{}",
                            stringify!($ty),
                            $items,
                            valid,
                            block,
                            lane
                        );
                    }
                }
                cases += 1;
            }
        }};
    }
    typed!(i8, 1);
    typed!(i8, 4);
    typed!(u8, 1);
    typed!(u8, 4);
    typed!(i16, 1);
    typed!(i16, 4);
    typed!(u16, 1);
    typed!(u16, 4);
    typed!(i32, 1);
    typed!(i32, 4);
    typed!(u32, 1);
    typed!(u32, 4);
    typed!(i64, 1);
    typed!(i64, 4);
    typed!(u64, 1);
    typed!(u64, 4);
    typed!(f32, 1);
    typed!(f32, 4);
    typed!(f64, 1);
    typed!(f64, 4);
    typed!(i32, 9);
    // Empty input is legal for default-filled Load at offset zero; every
    // payload slot then contains zero, unlike an invalid zero scalar count.
    let plan = module.prepare_payload_sum::<i32, 4>(LaunchConfig1D::auto_shared(1, 33))?;
    let source = DeviceBuffer::from_host(&stream, &[] as &[i32])?;
    let target = DeviceBuffer::from_host(&stream, &[-1_i32; 33])?;
    let (_, target) = module
        .payload_sum_async_owned::<i32, 4, _, _>(&plan, source, target, 0, 0, false)
        .sync()?;
    assert_eq!(target.to_host_vec(&stream)?, [0; 33]);
    cases += 1;

    let plan = module.prepare_operators(LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 2)))?;
    for mode in 0..10 {
        let input: Vec<i32> = (0..128)
            .map(|i| {
                if mode == 1 {
                    if i == 0 { 2 } else { 1 }
                } else {
                    i % 7 + 1
                }
            })
            .collect();
        let expected = input
            .iter()
            .copied()
            .reduce(|a, b| match mode {
                1 => a * b,
                2 => a.min(b),
                3 => a.max(b),
                4 => a & b,
                5 => a | b,
                6 => a ^ b,
                _ => a + b,
            })
            .unwrap();
        let source = DeviceBuffer::from_host(&stream, &input)?;
        let target = DeviceBuffer::from_host(&stream, &[-1_i32])?;
        let (_, target) = module
            .operators_async_owned(&plan, source, target, mode)
            .sync()?;
        assert_eq!(
            target.to_host_vec(&stream)?,
            [expected],
            "operator mode {mode}"
        );
        cases += 1;
    }
    let plan = module.prepare_scalar_count(LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 2)))?;
    for (valid, counted) in [(1, true), (31, true), (32, true), (32, false)] {
        let target = DeviceBuffer::from_host(&stream, &[-1_i32])?;
        let target = module
            .scalar_count_async_owned(&plan, target, valid, counted)
            .sync()?;
        assert_eq!(
            target.to_host_vec(&stream)?,
            [(valid * (valid + 1) / 2) as i32]
        );
        cases += 1;
    }
    let plan = module.prepare_storage(LaunchConfig1D::auto_shared(1, 32))?;
    for mode in 0..3 {
        let source = DeviceBuffer::from_host(&stream, &[1_i32; 128])?;
        let target = DeviceBuffer::from_host(&stream, &[-2_i32; 32])?;
        let (_, target) = module
            .storage_async_owned(&plan, source, target, mode)
            .sync()?;
        let expected: Vec<_> = (0..32)
            .map(|lane| {
                if lane == 0 {
                    if mode == 0 { 128 } else { 129 }
                } else {
                    -1
                }
            })
            .collect();
        assert_eq!(
            target.to_host_vec(&stream)?,
            expected,
            "storage mode {mode}"
        );
        cases += 1;
    }
    macro_rules! logical {
        ($items:literal) => {{
            let plan = module.prepare_logical::<$items>(LaunchConfig1D::auto_shared(1, 64))?;
            let target = DeviceBuffer::from_host(&stream, &[-3_i32; 64])?;
            let target = module
                .logical_async_owned::<$items, _>(&plan, target)
                .sync()?;
            let expected: Vec<_> = (0_i32..64)
                .map(|tid| {
                    if tid % 8 == 0 {
                        8 * (tid / 8 + 1) * $items * ($items + 1) / 2
                    } else {
                        -1
                    }
                })
                .collect();
            assert_eq!(target.to_host_vec(&stream)?, expected);
            cases += 1;
        }};
    }
    logical!(1);
    logical!(4);
    macro_rules! warp {
        ($prepare:ident, $launch:ident, $width:literal, $items:literal) => {{
            let plan = module.$prepare(LaunchConfig1D::auto_shared(1, 64))?;
            let target = DeviceBuffer::from_host(&stream, &[-3_i32; 64])?;
            let target = module.$launch(&plan, target).sync()?;
            let expected: Vec<_> = (0_i32..64)
                .map(|tid| {
                    if tid % 32 >= $width {
                        -2
                    } else if tid % 32 == 0 {
                        $width * (tid / 32 + 1) * $items * ($items + 1) / 2
                    } else {
                        -1
                    }
                })
                .collect();
            assert_eq!(target.to_host_vec(&stream)?, expected, stringify!($launch));
            cases += 1;
        }};
    }
    warp!(prepare_physical, physical_async_owned, 32, 1);
    warp!(prepare_prefix17, prefix17_async_owned, 17, 1);
    warp!(prepare_prefix23, prefix23_async_owned, 23, 4);
    let plan = module.prepare_prefix31(LaunchConfig1D::auto_shared(1, 64))?;
    for valid in [1, 30, 31] {
        let target = DeviceBuffer::from_host(&stream, &[-3_i32; 64])?;
        let target = module.prefix31_async_owned(&plan, target, valid).sync()?;
        let expected: Vec<_> = (0_i32..64)
            .map(|tid| {
                if tid % 32 == 31 {
                    -2
                } else if tid % 32 == 0 {
                    (tid / 32 + 1) * (valid * (valid + 1) / 2) as i32
                } else {
                    -1
                }
            })
            .collect();
        assert_eq!(target.to_host_vec(&stream)?, expected);
        cases += 1;
    }
    println!("PASS: {cases} public Reduce cases");
    Ok(())
}
