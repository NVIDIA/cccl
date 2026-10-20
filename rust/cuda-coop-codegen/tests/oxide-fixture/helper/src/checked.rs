// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Roundtrip through one shared generic helper reached from several kernels.
///
/// # Safety
/// Each block owns a disjoint tile. Launch exact 128-thread one-dimensional
/// blocks with sufficient scratch. Input/output cover every full tile; partial
/// counts are a global prefix within their combined extent. Every thread in
/// a block receives equal input/output origins, valid count and mode. The fixed
/// fill value is uniform, and each thread owns its separate initialized payload.
#[inline(never)]
pub unsafe fn roundtrip<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
    input: *const i32,
    output: *mut i32,
    post_store: *mut i32,
    valid: i32,
    mode: u32,
) {
    let lane = cuda_device::thread::threadIdx_x() as usize;
    let origin = cuda_device::thread::blockIdx_x() as usize * 128 * ITEMS_PER_THREAD;
    let valid = (valid - origin as i32)
        .max(0)
        .min((128 * ITEMS_PER_THREAD) as i32);
    let mut first = [0_i32; ITEMS_PER_THREAD];
    let mut second = [0_i32; ITEMS_PER_THREAD];
    // Lane-local choice reconverges before every collective.
    let payload = if lane & 1 == 0 {
        &mut first
    } else {
        &mut second
    };
    unsafe {
        // A uniform loop retains full participation. Scratch reuse still needs
        // the explicit caller-owned barrier; the checker inserts none.
        let mut iteration = 0;
        while iteration < 2 {
            coop_marker_fixture_api::checked_load::<ITEMS_PER_THREAD, TRANSPOSE>(
                input.add(origin),
                payload.as_mut_ptr(),
                valid,
                mode & 1 != 0,
            );
            cuda_device::thread::sync_threads();
            iteration += 1;
        }
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            payload[i] += 1;
            i += 1;
        }
        coop_marker_fixture_api::checked_store::<ITEMS_PER_THREAD, TRANSPOSE>(
            output.add(origin),
            payload.as_mut_ptr(),
            valid,
            mode & 2 != 0,
        );
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            post_store
                .add(origin + lane * ITEMS_PER_THREAD + i)
                .write(payload[i]);
            i += 1;
        }
        cuda_device::thread::sync_threads();
        if mode & 4 != 0 {
            coop_marker_fixture_api::checked_load::<ITEMS_PER_THREAD, TRANSPOSE>(
                output.add(origin),
                payload.as_mut_ptr(),
                valid,
                false,
            );
            let mut i = 0;
            while i < ITEMS_PER_THREAD {
                post_store
                    .add(origin + lane * ITEMS_PER_THREAD + i)
                    .write(payload[i]);
                i += 1;
            }
        }
    }
}
