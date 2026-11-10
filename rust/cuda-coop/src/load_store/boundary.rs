// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{
    Algorithm, LoadOptions, StoreOptions,
    sealed::{
        DIRECT, Group, STRIPED, TRANSPOSE, VECTORIZE, WARP_TRANSPOSE, WARP_TRANSPOSE_TIMESLICED,
    },
};
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

registration!(striped_load_full, "COOP_STRIPED_VALIDITY_LOAD_FULL");
registration!(striped_load_masked, "COOP_STRIPED_VALIDITY_LOAD_MASKED");
registration!(striped_load_filled, "COOP_STRIPED_VALIDITY_LOAD_FILLED");
registration!(striped_store_full, "COOP_STRIPED_VALIDITY_STORE_FULL");
registration!(striped_store_partial, "COOP_STRIPED_VALIDITY_STORE_PARTIAL");

registration!(vectorize_load_full, "COOP_VECTORIZE_VALIDITY_LOAD_FULL");
registration!(vectorize_load_masked, "COOP_VECTORIZE_VALIDITY_LOAD_MASKED");
registration!(vectorize_load_filled, "COOP_VECTORIZE_VALIDITY_LOAD_FILLED");
registration!(vectorize_store_full, "COOP_VECTORIZE_VALIDITY_STORE_FULL");
registration!(
    vectorize_store_partial,
    "COOP_VECTORIZE_VALIDITY_STORE_PARTIAL"
);

registration!(transpose_load_full, "COOP_TRANSPOSE_VALIDITY_LOAD_FULL");
registration!(transpose_load_masked, "COOP_TRANSPOSE_VALIDITY_LOAD_MASKED");
registration!(transpose_load_filled, "COOP_TRANSPOSE_VALIDITY_LOAD_FILLED");
registration!(transpose_store_full, "COOP_TRANSPOSE_VALIDITY_STORE_FULL");
registration!(
    transpose_store_partial,
    "COOP_TRANSPOSE_VALIDITY_STORE_PARTIAL"
);

registration!(
    warp_transpose_load_full,
    "COOP_WARP_TRANSPOSE_VALIDITY_LOAD_FULL"
);
registration!(
    warp_transpose_load_masked,
    "COOP_WARP_TRANSPOSE_VALIDITY_LOAD_MASKED"
);
registration!(
    warp_transpose_load_filled,
    "COOP_WARP_TRANSPOSE_VALIDITY_LOAD_FILLED"
);
registration!(
    warp_transpose_store_full,
    "COOP_WARP_TRANSPOSE_VALIDITY_STORE_FULL"
);
registration!(
    warp_transpose_store_partial,
    "COOP_WARP_TRANSPOSE_VALIDITY_STORE_PARTIAL"
);
registration!(
    warp_transpose_timesliced_load_full,
    "COOP_WARP_TRANSPOSE_TIMESLICED_VALIDITY_LOAD_FULL"
);
registration!(
    warp_transpose_timesliced_load_masked,
    "COOP_WARP_TRANSPOSE_TIMESLICED_VALIDITY_LOAD_MASKED"
);
registration!(
    warp_transpose_timesliced_load_filled,
    "COOP_WARP_TRANSPOSE_TIMESLICED_VALIDITY_LOAD_FILLED"
);
registration!(
    warp_transpose_timesliced_store_full,
    "COOP_WARP_TRANSPOSE_TIMESLICED_VALIDITY_STORE_FULL"
);
registration!(
    warp_transpose_timesliced_store_partial,
    "COOP_WARP_TRANSPOSE_TIMESLICED_VALIDITY_STORE_PARTIAL"
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
        assert!(ITEMS > 0 && ITEMS <= usize::MAX / G::WIDTH);
        G::WIDTH * ITEMS
    } else {
        tile::<ITEMS>()
    }
}

#[inline(always)]
pub(super) fn group_offset<const ITEMS: usize, G: Group>(offset: usize) -> usize {
    if G::WARP {
        let group = G::group_index() as usize;
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
        let warp = G::group_index() as usize;
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
        ($direct:ident, $striped:ident, $vectorize:ident, $transpose:ident, $warp:ident, $timesliced:ident, $controls:expr) => {
            match A::KIND {
                DIRECT => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $direct, memory, storage, $controls, payload,
                ),
                STRIPED => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $striped, memory, storage, $controls, payload,
                ),
                VECTORIZE => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $vectorize, memory, storage, $controls, payload,
                ),
                TRANSPOSE => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $transpose, memory, storage, $controls, payload,
                ),
                WARP_TRANSPOSE => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $warp, memory, storage, $controls, payload,
                ),
                WARP_TRANSPOSE_TIMESLICED => G::call::<T, _, _, _, _, _, ITEMS, false>(
                    $timesliced,
                    memory,
                    storage,
                    $controls,
                    payload,
                ),
                _ => panic!(),
            }
        };
    }
    // SAFETY: the checked marker enforces participation, uniform controls and
    // resource epochs. The exact typed provider initializes values before flags
    // and transports masked invalid cells without reading them as T. The
    // compiler allocates exact scratch and checks the selected completion policy.
    unsafe {
        match (options.valid_items, options.oob_default) {
            (None, None) => call!(
                load_full,
                striped_load_full,
                vectorize_load_full,
                transpose_load_full,
                warp_transpose_load_full,
                warp_transpose_timesliced_load_full,
                (pointer,)
            ),
            (Some(count), None) => {
                call!(
                    load_masked,
                    striped_load_masked,
                    vectorize_load_masked,
                    transpose_load_masked,
                    warp_transpose_load_masked,
                    warp_transpose_timesliced_load_masked,
                    (pointer, count as i32)
                )
            }
            (Some(count), Some(default)) => call!(
                load_filled,
                striped_load_filled,
                vectorize_load_filled,
                transpose_load_filled,
                warp_transpose_load_filled,
                warp_transpose_timesliced_load_filled,
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
    A: Algorithm,
    S: TempStorageLike,
    G: Group,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<A, S>,
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
        macro_rules! call {
            ($direct:ident, $striped:ident, $vectorize:ident, $controls:expr) => {
                match A::KIND {
                    DIRECT => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $direct,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    STRIPED => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $striped,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    VECTORIZE => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $vectorize,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    // Only the shared-preserving Store implementations call this boundary.
                    _ => panic!(),
                }
            };
        }
        match options.valid_items {
            None => call!(
                store_full,
                striped_store_full,
                vectorize_store_full,
                (pointer,)
            ),
            Some(count) => call!(
                store_partial,
                striped_store_partial,
                vectorize_store_partial,
                (pointer, count as i32)
            ),
        }
    }
}

#[inline(always)]
pub(super) fn store_permuting<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    const CAPACITY: usize,
    A: Algorithm,
    S: TempStorageLike,
    G: Group,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &mut ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<A, S>,
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
    let payload = value.block_io_parts();
    let storage = options.storage.into_raw();
    // SAFETY: this carrier owns the checked range. The provider checks the
    // source write domain before numeric reads, then permutes every opaque cell
    // together with its initialized bit. The exclusive payload borrow permits
    // that mutation. The compiler plans total scratch and completion before reuse.
    unsafe {
        let pointer = pointer.add(offset);
        macro_rules! call {
            ($transpose:ident, $warp:ident, $timesliced:ident, $controls:expr) => {
                match A::KIND {
                    TRANSPOSE => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $transpose,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    WARP_TRANSPOSE => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $warp,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    WARP_TRANSPOSE_TIMESLICED => G::call::<T, _, _, _, _, _, ITEMS, true>(
                        $timesliced,
                        destination,
                        storage,
                        $controls,
                        payload,
                    ),
                    _ => panic!(),
                }
            };
        }
        match options.valid_items {
            None => call!(
                transpose_store_full,
                warp_transpose_store_full,
                warp_transpose_timesliced_store_full,
                (pointer,)
            ),
            Some(count) => call!(
                transpose_store_partial,
                warp_transpose_store_partial,
                warp_transpose_timesliced_store_partial,
                (pointer, count as i32)
            ),
        }
    }
}
