// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Full-block warp exchange observes every payload slot before and after Store.
use cuda_coop as coop;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn expected<T: coop::Numeric>(
        input: &[T],
        offset: usize,
        index: usize,
        items: usize,
        valid: usize,
        default: T,
        mode: u32,
    ) -> Option<T> {
        if mode == 1 || mode == 5 || mode == 6 {
            if index < valid {
                Some(input[offset + index])
            } else {
                None
            }
        } else if mode == 2 || mode == 4 {
            if index < valid {
                Some(input[offset + index])
            } else {
                Some(default)
            }
        } else if mode == 3 {
            let lane = index / items;
            let item = index % items;
            if index < valid || (lane & 1 == 0 && item & 1 == 1) || (lane & 1 == 1 && item & 1 == 0)
            {
                Some(input[offset + index])
            } else {
                None
            }
        } else {
            Some(input[offset + index])
        }
    }

    #[inline(never)]
    fn exercise<
        T: coop::Numeric + PartialEq,
        const ITEMS: usize,
        const CAPACITY: usize,
        A: coop::Algorithm,
    >(
        input: &[T],
        output: &mut DisjointSlice<T, CollectiveTiles<CAPACITY>>,
        valid: usize,
        default: T,
        mode: u32,
        algorithm: A,
    ) {
        let block = coop::this_block();
        let x = thread::blockDim_x() as usize;
        let y = thread::blockDim_y() as usize;
        let threads = x * y * thread::blockDim_z() as usize;
        let lane = thread::threadIdx_x() as usize
            + x * (thread::threadIdx_y() as usize + y * thread::threadIdx_z() as usize);
        let group = thread::blockIdx_x() as usize
            + thread::gridDim_x() as usize
                * (thread::blockIdx_y() as usize
                    + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
        let offset = group * threads * CAPACITY;
        let mut values = coop::ThreadData::<_, ITEMS>::new();
        if mode != 9 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .algorithm(algorithm),
            );
        }
        if mode == 1 || mode == 5 || mode == 6 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .algorithm(algorithm),
            );
        } else if mode == 2 || mode == 4 {
            coop::load(
                block,
                input,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default)
                    .algorithm(algorithm),
            );
        } else if mode == 3 {
            let incoming = values;
            values = coop::ThreadData::new();
            let mut item = 0;
            while item < ITEMS {
                let index = lane * ITEMS + item;
                let initialize = index < valid
                    || (lane & 1 == 0 && item & 1 == 1)
                    || (lane & 1 == 1 && item & 1 == 0);
                if let (true, Some(value)) = (initialize, incoming.get(item)) {
                    values.write(item, value);
                }
                item += 1;
            }
        }
        let mut correct = true;
        let mut item = 0;
        while item < ITEMS {
            correct &= values.get(item)
                == expected(
                    input,
                    offset,
                    lane * ITEMS + item,
                    ITEMS,
                    valid,
                    default,
                    mode,
                );
            item += 1;
        }
        if mode == 0 || mode == 2 {
            coop::store(
                block,
                &mut *output,
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .algorithm(algorithm),
            );
        } else {
            let written = if mode == 5 {
                valid + 1
            } else if mode == 6 {
                valid / 2
            } else {
                valid
            };
            coop::store(
                block,
                &mut *output,
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(written)
                    .algorithm(algorithm),
            );
        }
        item = 0;
        while item < ITEMS {
            let source = (lane / 32) * (32 * ITEMS) + 32 * item + lane % 32;
            correct &=
                values.get(item) == expected(input, offset, source, ITEMS, valid, default, mode);
            item += 1;
        }
        // Managed scratch reuse inserts completion before this second operation.
        // A zero count still permutes all values and bits, with no global writes.
        coop::store(
            block,
            &mut *output,
            &mut values,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(0)
                .algorithm(algorithm),
        );
        item = 0;
        while item < ITEMS {
            let first = (lane / 32) * (32 * ITEMS) + 32 * item + lane % 32;
            let source =
                ((first / ITEMS) / 32) * (32 * ITEMS) + 32 * (first % ITEMS) + (first / ITEMS) % 32;
            correct &=
                values.get(item) == expected(input, offset, source, ITEMS, valid, default, mode);
            item += 1;
        }
        if mode == 4 {
            thread::sync_threads();
            coop::load(
                block,
                &mut *output,
                &mut values,
                coop::LoadOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .oob_default(default)
                    .algorithm(algorithm),
            );
            coop::store(
                block,
                &mut *output,
                &mut values,
                coop::StoreOptions::default()
                    .offset(offset)
                    .valid_items(valid)
                    .algorithm(algorithm),
            );
            item = 0;
            while item < ITEMS {
                correct &= values.get(item)
                    == expected(
                        input,
                        offset,
                        (lane / 32) * (32 * ITEMS) + 32 * item + lane % 32,
                        ITEMS,
                        valid,
                        default,
                        mode,
                    );
                item += 1;
            }
        }
        assert!(correct);
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,2),dynamic_shared=inferred)]
    pub fn block_warp<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4, _>(
            input,
            &mut output,
            valid,
            -17,
            mode,
            coop::WarpTranspose,
        );
    }
    #[kernel]
    #[launch_contract(domain=3,block=(8,4,2),dynamic_shared=inferred)]
    pub fn block_warp_timesliced<const ITEMS_PER_THREAD: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
        mode: u32,
    ) {
        exercise::<_, ITEMS_PER_THREAD, 4, _>(
            input,
            &mut output,
            valid,
            -17,
            mode,
            coop::WarpTransposeTimesliced,
        );
    }
}

pub fn run(trap: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig3D};
    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the typed declarations and device kernels are compiled together.
    let module = unsafe { kernels::load_async(0)? };
    if let Some(trap) = trap {
        let plan = module.prepare_block_warp_timesliced::<4>(LaunchConfig3D::auto_shared(
            (1, 1, 1),
            (8, 4, 2),
        ))?;
        let source = DeviceBuffer::from_host(&stream, &vec![11_i32; 256])?;
        let target = DeviceBuffer::from_host(&stream, &vec![0_i32; 256])?;
        let (valid, mode) = match trap {
            "block-warp-count" => (257, 1),
            "block-warp-unwritten" => (256, 9),
            "block-warp-different-count" => (31, 5),
            _ => panic!("unknown block warp trap"),
        };
        let _ = module
            .block_warp_timesliced_async_owned::<4, _, _>(&plan, source, target, valid, mode)
            .sync()?;
        panic!("expected block warp trap returned successfully");
    }
    let mut cases = 0;
    macro_rules! run {
        ($prepare:ident,$launch:ident,$items:literal,$scratch:literal) => {{
            let plan =
                module.$prepare::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (8, 4, 2)))?;
            assert_eq!(plan.__raw_config().shared_mem_bytes, $scratch);
            assert!(
                module
                    .$prepare::<$items>(LaunchConfig3D::auto_shared((2, 1, 1), (64, 1, 1)))
                    .is_err()
            );
            for mode in [0, 1, 2, 3, 4, 6] {
                for valid in [0, 1, 32 * $items + 1, 64 * $items - 1, 64 * $items] {
                    if mode == 0 && valid != 64 * $items {
                        continue;
                    }
                    let input: Vec<i32> = (0..512).map(|i| i + 19).collect();
                    let source = DeviceBuffer::from_host(&stream, &input)?;
                    let target = DeviceBuffer::from_host(&stream, &vec![0_i32; 512])?;
                    let (_, target) = module
                        .$launch::<$items, _, _>(&plan, source, target, valid, mode)
                        .sync()?;
                    let mut result = vec![0_i32; 512];
                    target.copy_to_host(&stream, &mut result)?;
                    stream.synchronize()?;
                    for (index, &value) in result.iter().enumerate() {
                        let within = index % 256;
                        let written = if mode == 0 || mode == 2 {
                            64 * $items
                        } else if mode == 6 {
                            valid / 2
                        } else {
                            valid
                        };
                        let expected = if within < written {
                            if mode == 2 && within >= valid {
                                -17
                            } else {
                                input[index]
                            }
                        } else {
                            0
                        };
                        assert_eq!(
                            value,
                            expected,
                            "{} items={} mode={} valid={} index={}",
                            stringify!($launch),
                            $items,
                            mode,
                            valid,
                            index
                        );
                    }
                    cases += 1;
                }
            }
        }};
    }
    // CUB maximum plus the independent counted-Store bitmap plane, alignment16.
    run!(prepare_block_warp, block_warp_async_owned, 1, 272);
    run!(prepare_block_warp, block_warp_async_owned, 4, 1072);
    run!(
        prepare_block_warp_timesliced,
        block_warp_timesliced_async_owned,
        1,
        144
    );
    run!(
        prepare_block_warp_timesliced,
        block_warp_timesliced_async_owned,
        4,
        560
    );
    println!("PASS {cases} public block warp Load/Store cases");
    Ok(())
}
