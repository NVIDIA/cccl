// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#![no_std]

/// Preserve a nongeneric dependency boundary above the marked generic helper.
///
/// # Safety
/// The caller must meet `roundtrip::<4, true>`'s complete safety contract.
#[inline(never)]
pub unsafe fn shared_roundtrip(
    input: *const i32,
    output: *mut i32,
    post_store: *mut i32,
    valid: i32,
    mode: u32,
) {
    unsafe { roundtrip::<4, true>(input, output, post_store, valid, mode) }
}

/// Roundtrip through one shared generic helper reached from several kernels.
///
/// # Safety
/// Each block owns a disjoint tile. Launch exact 128-thread one-dimensional
/// blocks with sufficient scratch. Input/output cover every full tile; partial
/// counts are a global prefix within their combined extent. Every thread in
/// a block receives equal input/output origins, valid count and mode. The fixed
/// fill value is uniform, and each thread owns its separate initialized payload.
#[cfg(not(feature = "checked"))]
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
    let mut payload = [0_i32; ITEMS_PER_THREAD];
    unsafe {
        coop_marker_fixture_api::load::<ITEMS_PER_THREAD, TRANSPOSE>(
            input.add(origin),
            payload.as_mut_ptr(),
            valid,
            mode & 1 != 0,
        );
        cuda_device::thread::sync_threads();
        for item in &mut payload {
            *item += 1;
        }
        coop_marker_fixture_api::store::<ITEMS_PER_THREAD, TRANSPOSE>(
            output.add(origin),
            payload.as_mut_ptr(),
            valid,
            mode & 2 != 0,
        );
        for (i, item) in payload.iter().enumerate() {
            post_store
                .add(origin + lane * ITEMS_PER_THREAD + i)
                .write(*item);
        }
        cuda_device::thread::sync_threads();
        if mode & 4 != 0 {
            coop_marker_fixture_api::load::<ITEMS_PER_THREAD, TRANSPOSE>(
                output.add(origin),
                payload.as_mut_ptr(),
                valid,
                false,
            );
            for (i, item) in payload.iter().enumerate() {
                post_store
                    .add(origin + lane * ITEMS_PER_THREAD + i)
                    .write(*item);
            }
        }
    }
}

#[cfg(feature = "checked")]
mod checked;
#[cfg(feature = "checked")]
pub use checked::roundtrip;
