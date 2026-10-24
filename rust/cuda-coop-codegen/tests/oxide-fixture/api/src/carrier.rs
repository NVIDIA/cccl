// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Private, scratch-free full-block integration fixture. These wrappers are not
//! the public Load/Store API: the compiler checks participation and resource
//! epochs, while this trusted boundary checks the unchanged offset and extent.
use cuda_device::{CollectiveTiles, DisjointSlice, library, thread};

#[inline(always)]
pub(crate) fn extent<const ITEMS: usize>(len: usize, offset: usize, valid: usize) {
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    assert!(ITEMS > 0 && threads <= usize::MAX / ITEMS);
    assert!(valid <= threads * ITEMS && valid <= i32::MAX as usize);
    assert!(offset <= len && valid <= len - offset);
}

#[inline(always)]
pub(crate) fn containment<const CAPACITY: usize>(offset: usize, valid: usize) {
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    let groups =
        thread::gridDim_x() as usize * thread::gridDim_y() as usize * thread::gridDim_z() as usize;
    let rank = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    assert!(CAPACITY > 0 && threads <= usize::MAX / CAPACITY);
    let capacity = threads * CAPACITY;
    assert!(groups > 0 && groups <= usize::MAX / capacity);
    let period = groups * capacity;
    if valid != 0 {
        let within = offset % period;
        let begin = rank * capacity;
        assert!(within >= begin && within - begin <= capacity);
        assert!(valid <= capacity - (within - begin));
    }
}

#[inline(never)]
pub fn load<const ITEMS: usize>(
    source: &[i32],
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
) {
    extent::<ITEMS>(source.len(), offset, valid);
    // SAFETY: filled Load initializes every payload element. Bounds are checked
    // before forming the effective pointer. The witness and pointer describe
    // the same allocation; the compiler proves full-block control uniformity.
    unsafe {
        library::checked_block_resource_call_v1::<i32, _, _, _, _, ITEMS, false>(
            super::direct_load_filled,
            source,
            (source.as_ptr().add(offset), valid as i32, -7_i32),
            (values.as_mut_ptr(),),
        );
    }
}

#[inline(never)]
pub fn store<const ITEMS: usize, const CAPACITY: usize>(
    target: &mut DisjointSlice<i32, CollectiveTiles<CAPACITY>>,
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
) {
    extent::<ITEMS>(target.len(), offset, valid);
    containment::<CAPACITY>(offset, valid);
    let pointer = target.as_mut_ptr();
    // SAFETY: this exact carrier owns the checked range. No lane reference to
    // it can escape. The payload is initialized, and the compiler checks epochs.
    unsafe {
        library::checked_block_resource_call_v1::<i32, _, _, _, _, ITEMS, true>(
            super::direct_store_partial,
            target,
            (pointer.add(offset), valid as i32),
            (values.as_mut_ptr(),),
        );
    }
}

#[inline(never)]
pub fn reload<const ITEMS: usize, const CAPACITY: usize>(
    source: &mut DisjointSlice<i32, CollectiveTiles<CAPACITY>>,
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
) {
    extent::<ITEMS>(source.len(), offset, valid);
    containment::<CAPACITY>(offset, valid);
    let pointer = source.as_mut_ptr();
    // SAFETY: immutable witness and effective pointer have the same origin.
    // The compiler requires a full-block barrier after any prior write.
    unsafe {
        library::checked_block_resource_call_v1::<i32, _, _, _, _, ITEMS, false>(
            super::direct_load_filled,
            &*source,
            (pointer.add(offset).cast_const(), valid as i32, -7_i32),
            (values.as_mut_ptr(),),
        );
    }
}
