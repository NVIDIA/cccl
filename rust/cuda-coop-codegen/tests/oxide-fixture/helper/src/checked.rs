// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Roundtrip through one shared generic helper reached from several kernels.
///
/// # Safety
/// Each block owns a disjoint tile. Launch exact root-declared one-dimensional
/// blocks with sufficient scratch. Input/output cover every full tile; partial
/// counts are a global prefix within their combined extent. Every thread in
/// a block receives equal input/output origins, valid count and mode. The fixed
/// fill value is uniform, and each thread owns its separate initialized payload.
#[cuda_device::device]
pub unsafe fn roundtrip<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
    input: *const i32,
    output: *mut i32,
    post_store: *mut i32,
    valid: i32,
    mode: u32,
) {
    unsafe {
        roundtrip_body::<ITEMS_PER_THREAD, TRANSPOSE>(input, output, post_store, valid, mode);
    }
}

// One original static identity must survive emission of this helper per root.
static mut CONTEXT_INCREMENT: i32 = 1;

#[inline(never)]
unsafe fn roundtrip_body<const ITEMS_PER_THREAD: usize, const TRANSPOSE: bool>(
    input: *const i32,
    output: *mut i32,
    post_store: *mut i32,
    valid: i32,
    mode: u32,
) {
    let block_threads = cuda_device::thread::blockDim_x() as usize;
    let lane = cuda_device::thread::threadIdx_x() as usize;
    let origin = cuda_device::thread::blockIdx_x() as usize * block_threads * ITEMS_PER_THREAD;
    let valid = (valid - origin as i32)
        .max(0)
        .min((block_threads * ITEMS_PER_THREAD) as i32);
    let mut first = [0_i32; ITEMS_PER_THREAD];
    let mut second = [0_i32; ITEMS_PER_THREAD];
    // Negative native modes: exactly lane zero fails a checked index, before
    // the collective sequence or between its first Load and reuse barrier.
    let trap_index = if lane == 0 { ITEMS_PER_THREAD } else { 0 };
    if mode & 8 != 0 {
        first[trap_index] = 0;
    }
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
            if mode & 16 != 0 {
                payload[trap_index] = 0;
            }
            cuda_device::thread::sync_threads();
            iteration += 1;
        }
        let mut i = 0;
        while i < ITEMS_PER_THREAD {
            payload[i] += CONTEXT_INCREMENT;
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
