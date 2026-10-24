// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Public storage policies and observable cross-lane reservation access.
use cuda_coop as coop;
use cuda_device::{CollectiveTiles, DisjointSlice, kernel, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn load_tile<const I: usize, const C: usize, const A: usize, const S: bool, const E: bool>(
        input: &[i32],
        values: &mut coop::ThreadData<i32, I>,
        offset: usize,
        valid: usize,
        scratch: &mut coop::TempStorage<C, A, S, E>,
    ) {
        coop::load(
            coop::this_block(),
            input,
            values,
            coop::LoadOptions::default()
                .offset(offset)
                .valid_items(valid)
                .oob_default(-7)
                .temp_storage(scratch)
                .algorithm(coop::Transpose),
        );
    }

    #[inline(never)]
    fn store_tile<const I: usize, const C: usize, const A: usize, const S: bool, const E: bool>(
        output: &mut DisjointSlice<i32, CollectiveTiles<4>>,
        values: &mut coop::ThreadData<i32, I>,
        offset: usize,
        valid: usize,
        scratch: &mut coop::TempStorage<C, A, S, E>,
    ) {
        coop::store(
            coop::this_block(),
            output,
            values,
            coop::StoreOptions::default()
                .offset(offset)
                .valid_items(valid)
                .algorithm(coop::Transpose)
                .temp_storage(scratch),
        );
    }

    #[inline(never)]
    fn descriptor() -> coop::TempStorage {
        coop::TempStorage::new()
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (33, 1, 1), dynamic_shared = inferred)]
    pub fn exercise<const ITEMS: usize, const MODE: usize>(
        input: &[i32],
        mut output: DisjointSlice<i32, CollectiveTiles<4>>,
        valid: usize,
    ) {
        let block = coop::this_block();
        let offset = thread::blockIdx_x() as usize * 33 * 4;
        let lane = thread::threadIdx_x() as usize;
        let mut values = coop::ThreadData::<_, ITEMS>::new();
        match MODE {
            0 => {
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default()
                        .offset(offset)
                        .valid_items(valid)
                        .oob_default(-7)
                        .algorithm(coop::Transpose),
                );
                coop::store(
                    block,
                    &mut output,
                    &mut values,
                    coop::StoreOptions::default()
                        .offset(offset)
                        .valid_items(valid)
                        .algorithm(coop::Transpose),
                );
            }
            1 => {
                let mut scratch = descriptor();
                load_tile(input, &mut values, offset, valid, &mut scratch);
                thread::sync_threads();
                store_tile(&mut output, &mut values, offset, valid, &mut scratch);
            }
            2 => {
                let mut scratch = coop::TempStorage::new().auto_sync();
                let mut iteration = 0;
                while iteration < 2 + (valid & 1) {
                    load_tile(input, &mut values, offset, valid, &mut scratch);
                    store_tile(&mut output, &mut values, offset, valid, &mut scratch);
                    iteration += 1;
                }
            }
            3 => {
                let mut scratch = coop::TempStorage::new().with_alignment::<64>().exclusive();
                // The same helper at two sites receives distinct exclusive slices.
                load_tile(input, &mut values, offset, valid, &mut scratch);
                load_tile(input, &mut values, offset, valid, &mut scratch);
                store_tile(&mut output, &mut values, offset, valid, &mut scratch);
            }
            4 => {
                let mut first = descriptor();
                let mut second = descriptor();
                load_tile(input, &mut values, offset, valid, &mut first);
                load_tile(input, &mut values, offset, valid, &mut second);
                thread::sync_threads();
                store_tile(&mut output, &mut values, offset, valid, &mut first);
            }
            5 => {
                let mut scratch = descriptor();
                let view = scratch.reserve::<i32, 33>();
                // SAFETY: each lane initializes its own in-bounds element, then
                // all writers complete before a different lane reads it.
                unsafe { core::ptr::write(view.as_mut_ptr().add(lane), lane as i32) };
                thread::sync_threads();
                let saved = unsafe { view.as_mut_ptr().add((lane + 1) % 33).read() };
                view.release();
                thread::sync_threads();
                load_tile(input, &mut values, offset, valid, &mut scratch);
                let mut item = 0;
                while item < ITEMS {
                    // Filled Load initialized every slot; the saved cross-lane
                    // value remains observable after the shared region is reused.
                    let value = values.get(item).unwrap_or(-100_000);
                    values.write(item, value + saved);
                    item += 1;
                }
                thread::sync_threads();
                store_tile(&mut output, &mut values, offset, valid, &mut scratch);
            }
            6 => {
                let mut scratch = coop::TempStorage::new().exclusive();
                let first = scratch.reserve::<i32, 33>();
                let second = scratch.reserve_aligned::<u64, 33, 32>();
                // SAFETY: distinct exclusive sites do not overlap. Each lane
                // writes its own slots before the full-block completion barrier.
                unsafe {
                    core::ptr::write(first.as_mut_ptr().add(lane), lane as i32);
                    core::ptr::write(second.as_mut_ptr().add(lane), (2 * lane + 1) as u64);
                }
                thread::sync_threads();
                let neighbor = (lane + 1) % 33;
                let saved = unsafe {
                    first.as_mut_ptr().add(neighbor).read()
                        + second.as_mut_ptr().add(neighbor).read() as i32
                };
                first.release();
                second.release();
                thread::sync_threads();
                load_tile(input, &mut values, offset, valid, &mut scratch);
                let mut item = 0;
                while item < ITEMS {
                    let value = values.get(item).unwrap_or(-100_000);
                    values.write(item, value + saved);
                    item += 1;
                }
                store_tile(&mut output, &mut values, offset, valid, &mut scratch);
            }
            7 => {
                let mut scratch = coop::TempStorage::new().auto_sync();
                coop::load(
                    block,
                    input,
                    &mut values,
                    coop::LoadOptions::default()
                        .offset(offset)
                        .valid_items(valid)
                        .oob_default(-7)
                        .temp_storage(&mut scratch),
                );
                coop::store(
                    block,
                    &mut output,
                    &values,
                    coop::StoreOptions::default()
                        .offset(offset)
                        .valid_items(valid)
                        .temp_storage(&mut scratch),
                );
            }
            _ => {
                let mut scratch = coop::TempStorage::new()
                    .with_capacity::<4096>()
                    .with_alignment::<64>()
                    .auto_sync();
                load_tile(input, &mut values, offset, valid, &mut scratch);
                store_tile(&mut output, &mut values, offset, valid, &mut scratch);
            }
        }
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::simt::device_context::{init_device_contexts, with_cuda_context};
    use cuda_async::simt::device_operation::DeviceOperation;
    use cuda_core::{DeviceBuffer, LaunchConfig1D};

    init_device_contexts(0, 1)?;
    let context = with_cuda_context(0, |context| context.clone())?;
    let stream = context.default_stream();
    // SAFETY: the embedded module and typed declarations are built together.
    let module = unsafe { kernels::load_async(0)? };
    let input: Vec<i32> = (0..3 * 33 * 4).map(|i| i * 7 + 3).collect();
    let mut cases = 0;
    macro_rules! run {
        ($items:literal, $mode:literal) => {{
            let plan =
                module.prepare_exercise::<$items, $mode>(LaunchConfig1D::auto_shared(3, 33))?;
            let scratch = plan.__raw_config().shared_mem_bytes;
            if $mode == 7 {
                assert_eq!(scratch, 0);
            } else if $mode == 8 {
                assert_eq!(scratch, 4096);
            } else {
                assert!(scratch > 0);
            }
            println!("storage mode={} items={} shared={scratch}", $mode, $items);
            for valid in [0, 1, 33 * $items - 1, 33 * $items] {
                let source = DeviceBuffer::from_host(&stream, &input)?;
                let target = DeviceBuffer::from_host(&stream, &vec![-19_i32; input.len()])?;
                let (_, target) = module
                    .exercise_async_owned::<$items, $mode, _, _>(&plan, source, target, valid)
                    .sync()?;
                let mut output = vec![0; input.len()];
                target.copy_to_host(&stream, &mut output)?;
                stream.synchronize()?;
                for (index, actual) in output.into_iter().enumerate() {
                    let local = index % (33 * 4);
                    let expected = if local < valid {
                        let neighbor = (local / $items + 1) % 33;
                        input[index]
                            + if $mode == 5 {
                                neighbor as i32
                            } else if $mode == 6 {
                                (3 * neighbor + 1) as i32
                            } else {
                                0
                            }
                    } else {
                        -19
                    };
                    assert_eq!(
                        actual, expected,
                        "mode={} items={} valid={valid} index={index}",
                        $mode, $items
                    );
                }
                cases += 1;
            }
        }};
    }
    macro_rules! modes {
        ($items:literal) => {
            run!($items, 0);
            run!($items, 1);
            run!($items, 2);
            run!($items, 3);
            run!($items, 4);
            run!($items, 5);
            run!($items, 6);
            run!($items, 7);
            run!($items, 8);
        };
    }
    modes!(1);
    modes!(4);
    println!("PASS: {cases} public storage cases");
    Ok(())
}
