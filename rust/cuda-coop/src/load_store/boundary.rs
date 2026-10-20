// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{LoadOptions, StoreOptions};
use crate::{Alignment, Numeric, SupportedAlignment, ThreadData};
use cuda_device::{CollectiveTiles, DisjointSlice, library, thread};

macro_rules! registration {
    ($name:ident, $variable:literal) => {
        fn $name() -> &'static str {
            library::registration!($variable)
        }
    };
}
registration!(load_full, "COOP_VALIDITY_LOAD_FULL");
registration!(load_masked, "COOP_VALIDITY_LOAD_MASKED");
registration!(load_filled, "COOP_VALIDITY_LOAD_FILLED");
registration!(store_full, "COOP_VALIDITY_STORE_FULL");
registration!(store_partial, "COOP_VALIDITY_STORE_PARTIAL");

#[inline(always)]
pub(super) fn tile<const ITEMS: usize>() -> usize {
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    assert!(ITEMS > 0 && threads <= usize::MAX / ITEMS);
    threads * ITEMS
}

#[inline(always)]
fn extent<T, const ITEMS: usize>(len: usize, offset: usize, valid: usize) {
    assert!(valid <= tile::<ITEMS>() && valid <= i32::MAX as usize);
    assert!(offset <= len && valid <= len - offset);
    assert!(offset <= i64::MAX as usize);
    assert!(offset <= isize::MAX as usize / core::mem::size_of::<T>());
}

#[inline(always)]
pub(super) fn containment<const CAPACITY: usize>(offset: usize, valid: usize) {
    let capacity = tile::<CAPACITY>();
    let groups =
        thread::gridDim_x() as usize * thread::gridDim_y() as usize * thread::gridDim_z() as usize;
    let rank = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    assert!(groups > 0 && groups <= usize::MAX / capacity);
    let period = groups * capacity;
    if valid != 0 {
        let within = offset % period;
        let begin = rank * capacity;
        assert!(within >= begin && within - begin <= capacity);
        assert!(valid <= capacity - (within - begin));
    }
}

// The caller binds pointer/length to this immutable allocation witness and
// checks collective containment when the source is a writable carrier.
#[inline(always)]
pub(super) unsafe fn load<T: Numeric, Memory, const ITEMS: usize, const ALIGN: usize>(
    memory: Memory,
    pointer: *const T,
    len: usize,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = match options.valid_items {
        Some(count) => count,
        None => tile::<ITEMS>(),
    };
    extent::<T, ITEMS>(len, options.offset, valid);
    // SAFETY: the caller binds pointer/length to the source allocation; bounds
    // are checked before forming the pointer, including an empty prefix.
    let pointer = unsafe { pointer.add(options.offset) };
    let payload = output.block_io_parts();
    // SAFETY: the checked marker enforces participation, uniform controls and
    // resource epochs. The exact typed provider initializes values before flags
    // and never interprets a masked invalid slot as T. No initialized slice is
    // exposed. Each branch consumes the same source witness.
    unsafe {
        match (options.valid_items, options.oob_default) {
            (None, None) => library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                load_full,
                memory,
                (pointer,),
                payload,
            ),
            (Some(count), None) => {
                library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                    load_masked,
                    memory,
                    (pointer, count as i32),
                    payload,
                )
            }
            (Some(count), Some(default)) => {
                library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                    load_filled,
                    memory,
                    (pointer, count as i32, default),
                    payload,
                )
            }
            (None, Some(_)) => panic!(),
        }
    }
}

#[inline(always)]
pub(super) fn store<T: Numeric, const ITEMS: usize, const ALIGN: usize, const CAPACITY: usize>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = match options.valid_items {
        Some(count) => count,
        None => tile::<ITEMS>(),
    };
    extent::<T, ITEMS>(destination.len(), options.offset, valid);
    containment::<CAPACITY>(options.offset, valid);
    let pointer = destination.as_mut_ptr();
    let payload = value.block_io_read_parts();
    // SAFETY: this carrier owns the checked write range. The provider checks
    // the entire lane-local write domain before reading numeric bytes and takes
    // const payload/flag pointers. It preserves all slots, including invalid
    // ones. The compiler checks uniform controls, participation and epochs.
    unsafe {
        let pointer = pointer.add(options.offset);
        match options.valid_items {
            None => library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
                store_full,
                destination,
                (pointer,),
                payload,
            ),
            Some(count) => library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
                store_partial,
                destination,
                (pointer, count as i32),
                payload,
            ),
        }
    }
}
