// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Public Scan values, validity, preservation, groups, and scratch composition.
use cuda_coop as coop;
use cuda_device::{kernel, launch_contract};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn observe<T: coop::Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        input: &coop::ThreadData<T, ITEMS_PER_THREAD>,
        before: &coop::ThreadData<T, ITEMS_PER_THREAD>,
        result: &coop::ThreadData<T, ITEMS_PER_THREAD>,
    ) -> bool {
        let mut matches = true;
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            matches &= input.get(i) == before.get(i) && result.get(i).is_some();
            i += 1;
        }
        matches && result.get(ITEMS_PER_THREAD).is_none()
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn scan_sum<T: coop::Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        input: &[T],
        mut output: coop::DisjointSlice<T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        valid: usize,
        zero: T,
        invalid: bool,
    ) {
        let block = coop::this_block();
        let offset = block.rank_at::<coop::Grid>() as usize * 64 * ITEMS_PER_THREAD;
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
        let mut result = coop::scan(block, &values, Default::default());
        let matches = observe(&values, &before, &result);
        // Store every result, including prefixes whose input was default-filled.
        coop::store(
            block,
            &mut output,
            &result,
            coop::StoreOptions::default().offset(offset),
        );
        result.write(0, zero);
        assert!(matches && observe(&values, &before, &result));
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_seeded<T: coop::Numeric + PartialEq, const ITEMS_PER_THREAD: usize>(
        input: &[T],
        mut output: coop::DisjointSlice<T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
        valid: usize,
        zero: T,
        seed: T,
    ) {
        let block = coop::this_block();
        let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .valid_items(valid)
                .oob_default(zero),
        );
        let before = values;
        let result = coop::exclusive_scan(
            block,
            &mut values,
            coop::ScanOptions::default().initial_value(seed),
        );
        let matches = observe(&values, &before, &result);
        coop::store(block, &mut output, &result, Default::default());
        assert!(matches && observe(&values, &before, &result));
    }

    #[kernel]
    #[launch_contract(domain = 3, block = (4, 4, 4), dynamic_shared = inferred)]
    pub fn scan_operators(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        mode: u32,
    ) {
        let block = coop::this_block();
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(block, input, &mut values, Default::default());
        let before = values;
        let result = match mode {
            0 => coop::exclusive_sum(block, &values, Default::default()),
            1 => coop::inclusive_sum(block, &values, Default::default()),
            2 => coop::exclusive_scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .scan_op(coop::Multiplies)
                    .initial_value(1),
            ),
            3 => coop::inclusive_scan(
                block,
                &values,
                coop::ScanOptions::default().scan_op(coop::Min),
            ),
            4 => coop::scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .mode(coop::Inclusive)
                    .scan_op(coop::Max),
            ),
            5 => coop::exclusive_scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .scan_op(coop::BitAnd)
                    .initial_value(255),
            ),
            6 => coop::inclusive_scan(
                block,
                &values,
                coop::ScanOptions::default().scan_op(coop::BitOr),
            ),
            7 => coop::inclusive_scan(
                block,
                &values,
                coop::ScanOptions::default().scan_op(coop::BitXor),
            ),
            8 => coop::exclusive_scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .initial_value(3)
                    .algorithm(coop::RakingMemoize),
            ),
            _ => coop::exclusive_scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .initial_value(3)
                    .algorithm(coop::WarpScans),
            ),
        };
        let matches = observe(&values, &before, &result);
        coop::store(block, &mut output, &result, Default::default());
        assert!(matches && observe(&values, &before, &result));
    }

    #[inline(never)]
    fn with_storage<const AUTO: bool>(
        input: &[i32],
        output: &mut coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
        scratch: &mut coop::TempStorage<0, 0, AUTO, false>,
    ) {
        let block = coop::this_block();
        let mut values = coop::ThreadData::<_, 4>::new();
        coop::load(
            block,
            input,
            &mut values,
            coop::LoadOptions::default()
                .valid_items(valid)
                .oob_default(0)
                .algorithm(coop::Transpose)
                .temp_storage(scratch),
        );
        if !AUTO {
            cuda_device::thread::sync_threads();
        }
        let before = values;
        let mut matches = true;
        let mut iteration = 0;
        while iteration < 2 {
            let result = coop::exclusive_scan(
                block,
                &values,
                coop::ScanOptions::default()
                    .initial_value(5)
                    .temp_storage(scratch),
            );
            matches &= observe(&values, &before, &result);
            if !AUTO {
                cuda_device::thread::sync_threads();
            }
            coop::store(block, &mut *output, &result, Default::default());
            // Complete the global Store before repeating the same output region.
            cuda_device::thread::sync_threads();
            iteration += 1;
        }
        assert!(matches);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_manual(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
    ) {
        let mut scratch = coop::TempStorage::new();
        with_storage(input, &mut output, valid, &mut scratch);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_auto(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<4>>,
        valid: usize,
    ) {
        let mut scratch = coop::TempStorage::new().auto_sync();
        with_storage(input, &mut output, valid, &mut scratch);
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_warp(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
        seeded: bool,
    ) {
        let group = coop::this_warp().group_by::<8>();
        let block = coop::this_block();
        let offset = block.rank_at::<coop::Grid>() as usize * 64;
        let mut values = coop::ThreadData::<_, 1>::new();
        coop::load(
            group,
            input,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let value = values.get(0).unwrap_or(0);
        let q = block.rank_at::<coop::Warp>() as i32 * 4 + group.rank_at::<coop::Warp>() as i32;
        let result = if seeded {
            coop::exclusive_scan(
                group,
                value,
                coop::ScanOptions::default().initial_value(q + 5),
            )
        } else {
            coop::exclusive_sum(group, value, Default::default())
        };
        let matches = values.get(0) == Some(value);
        coop::store(
            group,
            &mut output,
            result,
            coop::StoreOptions::default().offset(offset),
        );
        assert!(matches && values.get(0) == Some(value));
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_physical_warp(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
    ) {
        let group = coop::this_warp();
        let offset = coop::this_block().rank_at::<coop::Grid>() as usize * 64;
        let mut values = coop::ThreadData::<_, 1>::new();
        coop::load(
            group,
            input,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let value = values.get(0).unwrap_or(0);
        let result = coop::exclusive_sum(group, value, Default::default());
        coop::store(
            group,
            &mut output,
            result,
            coop::StoreOptions::default().offset(offset),
        );
        assert!(values.get(0) == Some(value));
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1), dynamic_shared = inferred)]
    pub fn scan_block_scalar(
        input: &[i32],
        mut output: coop::DisjointSlice<i32, coop::CollectiveTiles<1>>,
    ) {
        let group = coop::this_block();
        let offset = group.rank_at::<coop::Grid>() as usize * 64;
        let mut values = coop::ThreadData::<_, 1>::new();
        coop::load(
            group,
            input,
            &mut values,
            coop::LoadOptions::default().offset(offset),
        );
        let value = values.get(0).unwrap_or(0);
        let result = coop::scan(group, value, Default::default());
        coop::store(
            group,
            &mut output,
            result,
            coop::StoreOptions::default().offset(offset),
        );
        assert!(values.get(0) == Some(value));
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
    if std::env::args().any(|arg| arg == "invalid-input") {
        let source = DeviceBuffer::from_host(&stream, &[1_i32; 256])?;
        let target = DeviceBuffer::from_host(&stream, &[0_i32; 256])?;
        let plan =
            module.prepare_scan_sum::<i32, 4>(LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 4)))?;
        module
            .scan_sum_async_owned::<i32, 4, _, _>(&plan, source, target, 1, 0, true)
            .sync()?;
        stream.synchronize()?;
        panic!("unwritten Scan input unexpectedly completed");
    }
    let mut cases = 0;
    macro_rules! sum_cases {
        ($ty:ty, $items:literal) => {{
            let tile = 64 * $items;
            let input: Vec<$ty> = (0..2 * tile).map(|i| (i % 8 == 0) as u8 as $ty).collect();
            let plan = module.prepare_scan_sum::<$ty, $items>(LaunchConfig3D::auto_shared(
                (2, 1, 1),
                (4, 4, 4),
            ))?;
            for valid in [0, 1, tile - 1, tile] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![0 as $ty; input.len()])?;
                let (_, target) = module
                    .scan_sum_async_owned::<$ty, $items, _, _>(
                        &plan, source, target, valid, 0 as $ty, false,
                    )
                    .sync()?;
                let mut actual = vec![0 as $ty; input.len()];
                target.copy_to_host(&stream, &mut actual)?;
                stream.synchronize()?;
                let mut expected = Vec::with_capacity(input.len());
                for chunk in input.chunks(tile) {
                    let mut prefix = 0 as $ty;
                    for (i, &value) in chunk.iter().enumerate() {
                        expected.push(prefix);
                        if i < valid {
                            prefix += value;
                        }
                    }
                }
                assert_eq!(
                    actual,
                    expected,
                    "{} N{} valid{}",
                    stringify!($ty),
                    $items,
                    valid
                );
                cases += 1;
            }
        }};
    }
    sum_cases!(i8, 1);
    sum_cases!(i8, 4);
    sum_cases!(u8, 1);
    sum_cases!(u8, 4);
    sum_cases!(i16, 1);
    sum_cases!(i16, 4);
    sum_cases!(u16, 1);
    sum_cases!(u16, 4);
    sum_cases!(i32, 1);
    sum_cases!(i32, 4);
    sum_cases!(u32, 1);
    sum_cases!(u32, 4);
    sum_cases!(i64, 1);
    sum_cases!(i64, 4);
    sum_cases!(u64, 1);
    sum_cases!(u64, 4);
    sum_cases!(f32, 1);
    sum_cases!(f32, 4);
    sum_cases!(f64, 1);
    sum_cases!(f64, 4);
    sum_cases!(i32, 9);
    macro_rules! seeded_cases {
        ($ty:ty, $items:literal, $seed:expr) => {{
            let tile = 64 * $items;
            let input: Vec<$ty> = (0..tile).map(|i| (i % 8 == 0) as u8 as $ty).collect();
            let plan =
                module.prepare_scan_seeded::<$ty, $items>(LaunchConfig1D::auto_shared(1, 64))?;
            for valid in [tile - 1, tile] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![0 as $ty; tile])?;
                let (_, target) = module
                    .scan_seeded_async_owned::<$ty, $items, _, _>(
                        &plan, source, target, valid, 0 as $ty, $seed,
                    )
                    .sync()?;
                let mut actual = vec![0 as $ty; tile];
                target.copy_to_host(&stream, &mut actual)?;
                stream.synchronize()?;
                let mut prefix: $ty = $seed;
                for (i, (&value, &got)) in input.iter().zip(&actual).enumerate() {
                    assert_eq!(got, prefix);
                    if i < valid {
                        prefix += value;
                    }
                }
                cases += 1;
            }
        }};
    }
    seeded_cases!(i8, 4, -3_i8);
    seeded_cases!(u16, 1, 7_u16);
    let plan = module.prepare_scan_operators(LaunchConfig3D::auto_shared((1, 1, 1), (4, 4, 4)))?;
    for mode in 0..10 {
        let input: Vec<i32> = (0..256)
            .map(|i| {
                if mode == 2 {
                    if i == 0 { 2 } else { 1 }
                } else {
                    i % 7 + 1
                }
            })
            .collect();
        let source = DeviceBuffer::from_host(&stream, &input)?;
        let target = DeviceBuffer::from_host(&stream, &[0_i32; 256])?;
        let (_, target) = module
            .scan_operators_async_owned(&plan, source, target, mode)
            .sync()?;
        let mut actual = vec![0_i32; 256];
        target.copy_to_host(&stream, &mut actual)?;
        stream.synchronize()?;
        let inclusive = matches!(mode, 1 | 3 | 4 | 6 | 7);
        let mut prefix = match mode {
            2 => 1,
            3 => i32::MAX,
            4 => i32::MIN,
            5 => 255,
            8 | 9 => 3,
            _ => 0,
        };
        for (&value, &got) in input.iter().zip(&actual) {
            let next = match mode {
                2 => prefix * value,
                3 => prefix.min(value),
                4 => prefix.max(value),
                5 => prefix & value,
                6 => prefix | value,
                7 => prefix ^ value,
                _ => prefix + value,
            };
            assert_eq!(got, if inclusive { next } else { prefix }, "mode {mode}");
            prefix = next;
        }
        cases += 1;
    }
    macro_rules! storage_cases {
        ($prepare:ident, $launch:ident) => {{
            let input: Vec<i32> = (0..256).map(|i| (i % 8 == 0) as i32).collect();
            let plan = module.$prepare(LaunchConfig1D::auto_shared(1, 64))?;
            for valid in [255, 256] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &[0_i32; 256])?;
                let (_, target) = module.$launch(&plan, source, target, valid).sync()?;
                let mut actual = vec![0_i32; 256];
                target.copy_to_host(&stream, &mut actual)?;
                stream.synchronize()?;
                let mut prefix = 5;
                for (i, (&value, &got)) in input.iter().zip(&actual).enumerate() {
                    assert_eq!(got, prefix);
                    if i < valid {
                        prefix += value;
                    }
                }
                cases += 1;
            }
        }};
    }
    storage_cases!(prepare_scan_manual, scan_manual_async_owned);
    storage_cases!(prepare_scan_auto, scan_auto_async_owned);
    let plan = module.prepare_scan_warp(LaunchConfig1D::auto_shared(3, 64))?;
    for seeded in [false, true] {
        let source = DeviceBuffer::from_host(&stream, &[1_i32; 192])?;
        let target = DeviceBuffer::from_host(&stream, &[0_i32; 192])?;
        let (_, target) = module
            .scan_warp_async_owned(&plan, source, target, seeded)
            .sync()?;
        let mut actual = vec![0_i32; 192];
        target.copy_to_host(&stream, &mut actual)?;
        stream.synchronize()?;
        for (i, &got) in actual.iter().enumerate() {
            let seed = if seeded { (i % 64 / 8) as i32 + 5 } else { 0 };
            assert_eq!(got, seed + (i % 8) as i32);
        }
        cases += 1;
    }
    macro_rules! scalar_case {
        ($prepare:ident, $launch:ident, $width:literal) => {{
            let plan = module.$prepare(LaunchConfig1D::auto_shared(3, 64))?;
            let source = DeviceBuffer::from_host(&stream, &[1_i32; 192])?;
            let target = DeviceBuffer::from_host(&stream, &[0_i32; 192])?;
            let (_, target) = module.$launch(&plan, source, target).sync()?;
            let mut actual = vec![0_i32; 192];
            target.copy_to_host(&stream, &mut actual)?;
            stream.synchronize()?;
            for (i, &got) in actual.iter().enumerate() {
                assert_eq!(got, (i % $width) as i32);
            }
            cases += 1;
        }};
    }
    scalar_case!(
        prepare_scan_physical_warp,
        scan_physical_warp_async_owned,
        32
    );
    scalar_case!(prepare_scan_block_scalar, scan_block_scalar_async_owned, 64);
    println!("PASS: {cases} public Scan cases");
    Ok(())
}
