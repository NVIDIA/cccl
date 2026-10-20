// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

pub use coop_marker_fixture_api::storage::{load, reload, store};
use cuda_device::{CollectiveTiles, DisjointSlice, library, thread};

/// # Safety
/// Called uniformly by a block. Each rooted constructor call is independent.
#[inline(never)]
pub unsafe fn descriptor() -> library::StorageDescriptor {
    unsafe { library::storage_new_v1::<0, 0, false, false>() }
}

/// # Safety
/// Exact 128-thread blocks own disjoint 512-element target tiles. Input covers
/// all source ranges; every control and descriptor agrees across block lanes.
#[inline(never)]
pub unsafe fn roundtrip<const ITEMS: usize, const TRANSPOSE: bool, const MODE: usize>(
    input: &[i32],
    output: &mut DisjointSlice<i32, CollectiveTiles<4>>,
    valid: usize,
) {
    let offset = thread::blockIdx_x() as usize * 512;
    let mut values = [0; ITEMS];
    unsafe {
        match MODE {
            0 => {
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, ());
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, ());
            }
            1 => {
                let mut scratch = descriptor();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                thread::sync_threads();
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
            2 => {
                let mut scratch = library::storage_new_v1::<0, 0, true, false>();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
            3 => {
                let mut scratch = library::storage_new_v1::<0, 64, false, true>();
                // Same generic helper, different static uses and assigned offsets.
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
            4 => {
                let mut first = descriptor();
                let mut second = descriptor();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut first);
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut second);
                thread::sync_threads();
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut first);
            }
            5 => {
                let mut scratch = descriptor();
                let view = library::storage_reserve_v1::<i32, _, 128, 32>(&mut scratch);
                let lane = thread::threadIdx_x() as usize;
                view.as_mut_ptr().add(lane).write(lane as i32);
                thread::sync_threads();
                let saved = view.as_mut_ptr().add((lane + 1) % 128).read();
                library::storage_release_v1(view);
                thread::sync_threads();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                // A cross-lane read remains observable after this region is reused.
                let mut item = 0;
                while item < ITEMS {
                    values[item] += saved;
                    item += 1;
                }
                thread::sync_threads();
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
            6 => {
                let mut scratch = library::storage_new_v1::<0, 32, false, true>();
                let first = library::storage_reserve_v1::<i32, _, 128, 0>(&mut scratch);
                let second = library::storage_reserve_v1::<u64, _, 128, 32>(&mut scratch);
                let lane = thread::threadIdx_x() as usize;
                first.as_mut_ptr().add(lane).write(lane as i32);
                second.as_mut_ptr().add(lane).write((lane * 2 + 1) as u64);
                thread::sync_threads();
                let neighbor = (lane + 1) % 128;
                let saved = first.as_mut_ptr().add(neighbor).read()
                    + second.as_mut_ptr().add(neighbor).read() as i32;
                library::storage_release_v1(first);
                library::storage_release_v1(second);
                thread::sync_threads();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                let mut item = 0;
                while item < ITEMS {
                    values[item] += saved;
                    item += 1;
                }
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
            7 => {
                let prefix =
                    cuda_device::DynamicSharedArray::<[u64; 8], 32>::get_fixed().cast::<u64>();
                let lane = thread::threadIdx_x() as usize;
                if lane < 8 {
                    prefix.add(lane).write(lane as u64 + 10);
                }
                thread::sync_threads();
                let mut first = descriptor();
                let mut second = library::storage_new_v1::<0, 64, false, false>();
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut first);
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut second);
                let saved = prefix.add(lane % 8).read() as i32;
                let mut item = 0;
                while item < ITEMS {
                    values[item] += saved;
                    item += 1;
                }
                thread::sync_threads();
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut first);
            }
            8 => {
                // Runtime trip count keeps reuse of these same static sites
                // observable, including the planned barrier on the backedge.
                let mut iteration = 0;
                while iteration < 2 + (valid & 3) {
                    load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, ());
                    let mut item = 0;
                    while item < ITEMS {
                        values[item] += iteration as i32;
                        item += 1;
                    }
                    store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, ());
                    iteration += 1;
                }
            }
            _ => {
                let original = descriptor();
                let mut scratch: library::StorageDescriptor<0, 64, true, false> =
                    library::storage_reconfigure_v1(original);
                load::<ITEMS, TRANSPOSE, _>(input, offset, valid, &mut values, &mut scratch);
                store::<ITEMS, TRANSPOSE, _>(output, offset, valid, &mut values, &mut scratch);
            }
        }
    }
}
