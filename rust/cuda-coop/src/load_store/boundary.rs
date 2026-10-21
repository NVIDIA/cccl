// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{Algorithm, Direct, LoadOptions, StoreOptions, Transpose, sealed::Group};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
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

registration!(transpose_load_full, "COOP_TRANSPOSE_VALIDITY_LOAD_FULL");
registration!(transpose_load_masked, "COOP_TRANSPOSE_VALIDITY_LOAD_MASKED");
registration!(transpose_load_filled, "COOP_TRANSPOSE_VALIDITY_LOAD_FILLED");
registration!(transpose_store_full, "COOP_TRANSPOSE_VALIDITY_STORE_FULL");
registration!(
    transpose_store_partial,
    "COOP_TRANSPOSE_VALIDITY_STORE_PARTIAL"
);

#[inline(always)]
pub(super) fn tile<const ITEMS: usize>() -> usize {
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    assert!(ITEMS > 0 && threads <= usize::MAX / ITEMS);
    threads * ITEMS
}

#[inline(always)]
fn extent<T, const ITEMS: usize, G: Group>(len: usize, offset: usize, valid: usize) {
    assert!(valid <= group_tile::<ITEMS, G>() && valid <= i32::MAX as usize);
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
        let within = if offset < period {
            offset
        } else {
            offset % period
        };
        let begin = rank * capacity;
        assert!(within >= begin && within - begin <= capacity);
        assert!(valid <= capacity - (within - begin));
    }
}

#[inline(always)]
pub(super) fn group_tile<const ITEMS: usize, G: Group>() -> usize {
    if G::WARP {
        assert!(ITEMS > 0 && ITEMS <= usize::MAX / 32);
        32 * ITEMS
    } else {
        tile::<ITEMS>()
    }
}

#[inline(always)]
pub(super) fn group_offset<const ITEMS: usize, G: Group>(offset: usize) -> usize {
    if G::WARP {
        let group = library::warp_group_index_v1::<32>() as usize;
        let tile = group_tile::<ITEMS, G>();
        assert!(group <= usize::MAX / tile);
        let origin = group * tile;
        assert!(offset <= usize::MAX - origin);
        offset + origin
    } else {
        offset
    }
}

#[inline(always)]
pub(super) fn group_containment<const CAPACITY: usize, G: Group>(offset: usize, valid: usize) {
    if !G::WARP {
        containment::<CAPACITY>(offset, valid);
        return;
    }
    let block_capacity = tile::<CAPACITY>();
    let capacity = group_tile::<CAPACITY, G>();
    let groups =
        thread::gridDim_x() as usize * thread::gridDim_y() as usize * thread::gridDim_z() as usize;
    let rank = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    assert!(groups > 0 && groups <= usize::MAX / block_capacity);
    let period = groups * block_capacity;
    if valid != 0 {
        let within = if offset < period {
            offset
        } else {
            offset % period
        };
        let warp = library::warp_group_index_v1::<32>() as usize;
        let begin = rank * block_capacity + warp * capacity;
        assert!(within >= begin && within - begin <= capacity);
        assert!(valid <= capacity - (within - begin));
    }
}

// The caller binds pointer/length to this immutable allocation witness and
// checks collective containment when the source is a writable carrier.
#[inline(always)]
pub(super) unsafe fn load<
    T: Numeric,
    Memory,
    const ITEMS: usize,
    const ALIGN: usize,
    A: Algorithm,
    S: TempStorageLike,
    G: Group,
>(
    memory: Memory,
    pointer: *const T,
    len: usize,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T, A, S>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = match options.valid_items {
        Some(count) => count,
        None => group_tile::<ITEMS, G>(),
    };
    let offset = group_offset::<ITEMS, G>(options.offset);
    extent::<T, ITEMS, G>(len, offset, valid);
    // SAFETY: the caller binds pointer/length to the source allocation; bounds
    // are checked before forming the pointer, including an empty prefix.
    let pointer = unsafe { pointer.add(offset) };
    let payload = output.block_io_parts();
    let storage = options.storage.into_raw();
    // Each static algorithm keeps its registration function-item identity. The
    // storage marker plans the complete provider layout and either synchronizes
    // managed reuse or requires the caller's explicit completion.
    macro_rules! call {
        ($direct:ident, $transpose:ident, $controls:expr) => {
            if G::WARP {
                library::checked_warp_call_v1::<T, _, _, _, _, 32, ITEMS, false>(
                    $direct, memory, $controls, payload,
                )
            } else if A::TRANSPOSE {
                library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, false>(
                    $transpose, memory, storage, $controls, payload,
                )
            } else {
                library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, false>(
                    $direct, memory, storage, $controls, payload,
                )
            }
        };
    }
    // SAFETY: the checked marker enforces participation, uniform controls and
    // resource epochs. The exact typed provider initializes values before flags
    // and transports masked invalid cells without reading them as T. The
    // compiler allocates Transpose scratch and checks the required completion.
    unsafe {
        match (options.valid_items, options.oob_default) {
            (None, None) => call!(load_full, transpose_load_full, (pointer,)),
            (Some(count), None) => {
                call!(load_masked, transpose_load_masked, (pointer, count as i32))
            }
            (Some(count), Some(default)) => call!(
                load_filled,
                transpose_load_filled,
                (pointer, count as i32, default)
            ),
            (None, Some(_)) => panic!(),
        }
    }
}

#[inline(always)]
pub(super) fn store<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    const CAPACITY: usize,
    S: TempStorageLike,
    G: Group,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<Direct, S>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = match options.valid_items {
        Some(count) => count,
        None => group_tile::<ITEMS, G>(),
    };
    let offset = group_offset::<ITEMS, G>(options.offset);
    extent::<T, ITEMS, G>(destination.len(), offset, valid);
    group_containment::<CAPACITY, G>(offset, valid);
    let pointer = destination.as_mut_ptr();
    let payload = value.block_io_read_parts();
    let storage = options.storage.into_raw();
    // SAFETY: this carrier owns the checked write range. The provider checks
    // the entire lane-local write domain before reading numeric bytes and takes
    // const payload/flag pointers. It preserves all slots, including invalid
    // ones. The compiler checks uniform controls, participation and epochs.
    unsafe {
        let pointer = pointer.add(offset);
        if G::WARP {
            match options.valid_items {
                None => library::checked_warp_call_v1::<T, _, _, _, _, 32, ITEMS, true>(
                    store_full,
                    destination,
                    (pointer,),
                    payload,
                ),
                Some(count) => library::checked_warp_call_v1::<T, _, _, _, _, 32, ITEMS, true>(
                    store_partial,
                    destination,
                    (pointer, count as i32),
                    payload,
                ),
            }
        } else {
            match options.valid_items {
                None => library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, true>(
                    store_full,
                    destination,
                    storage,
                    (pointer,),
                    payload,
                ),
                Some(count) => {
                    library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, true>(
                        store_partial,
                        destination,
                        storage,
                        (pointer, count as i32),
                        payload,
                    )
                }
            }
        }
    }
}

#[inline(always)]
pub(super) fn store_transpose<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    const CAPACITY: usize,
    S: TempStorageLike,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &mut ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<Transpose, S>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = match options.valid_items {
        Some(count) => count,
        None => tile::<ITEMS>(),
    };
    extent::<T, ITEMS, crate::Block>(destination.len(), options.offset, valid);
    containment::<CAPACITY>(options.offset, valid);
    let pointer = destination.as_mut_ptr();
    let payload = value.block_io_parts();
    let storage = options.storage.into_raw();
    // SAFETY: this carrier owns the checked range. The provider checks the
    // source write domain before numeric reads, then permutes every opaque cell
    // together with its initialized bit. The exclusive payload borrow permits
    // that mutation. The compiler plans total scratch and completion before reuse.
    unsafe {
        let pointer = pointer.add(options.offset);
        match options.valid_items {
            None => library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, true>(
                transpose_store_full,
                destination,
                storage,
                (pointer,),
                payload,
            ),
            Some(count) => library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, true>(
                transpose_store_partial,
                destination,
                storage,
                (pointer, count as i32),
                payload,
            ),
        }
    }
}
