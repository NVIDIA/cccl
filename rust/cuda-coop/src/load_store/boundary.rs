// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{
    Algorithm, LoadOptions, StoreOptions, Transpose,
    sealed::{DIRECT, STRIPED, TRANSPOSE, VECTORIZE},
};
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

// The caller binds pointer/length to this immutable allocation witness and
// checks collective containment when the source is a writable carrier.
#[inline(always)]
pub(super) unsafe fn load<
    T: Numeric,
    Memory,
    const ITEMS: usize,
    const ALIGN: usize,
    A: Algorithm,
>(
    memory: Memory,
    pointer: *const T,
    len: usize,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T, A>,
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
    // Each static algorithm keeps its registration function-item identity. The
    // storage marker plans the complete provider layout and synchronizes reuse.
    macro_rules! call {
        ($direct:ident, $striped:ident, $vectorize:ident, $transpose:ident, $controls:expr) => {
            match A::KIND {
                DIRECT => library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                    $direct, memory, $controls, payload,
                ),
                STRIPED => library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                    $striped, memory, $controls, payload,
                ),
                VECTORIZE => {
                    library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
                        $vectorize, memory, $controls, payload,
                    )
                }
                TRANSPOSE => {
                    library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, false>(
                        $transpose,
                        memory,
                        (),
                        $controls,
                        payload,
                    )
                }
                _ => panic!(),
            }
        };
    }
    // SAFETY: the checked marker enforces participation, uniform controls and
    // resource epochs. The exact typed provider initializes values before flags
    // and transports masked invalid cells without reading them as T. The
    // compiler allocates Transpose scratch and completes its previous use.
    unsafe {
        match (options.valid_items, options.oob_default) {
            (None, None) => call!(
                load_full,
                striped_load_full,
                vectorize_load_full,
                transpose_load_full,
                (pointer,)
            ),
            (Some(count), None) => {
                call!(
                    load_masked,
                    striped_load_masked,
                    vectorize_load_masked,
                    transpose_load_masked,
                    (pointer, count as i32)
                )
            }
            (Some(count), Some(default)) => call!(
                load_filled,
                striped_load_filled,
                vectorize_load_filled,
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
    A: Algorithm,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<A>,
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
        macro_rules! call {
            ($direct:ident, $striped:ident, $vectorize:ident, $controls:expr) => {
                match A::KIND {
                    DIRECT => {
                        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
                            $direct,
                            destination,
                            $controls,
                            payload,
                        )
                    }
                    STRIPED => {
                        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
                            $striped,
                            destination,
                            $controls,
                            payload,
                        )
                    }
                    VECTORIZE => {
                        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
                            $vectorize,
                            destination,
                            $controls,
                            payload,
                        )
                    }
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
pub(super) fn store_transpose<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    const CAPACITY: usize,
>(
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: &mut ThreadData<T, ITEMS, ALIGN>,
    options: StoreOptions<Transpose>,
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
    let payload = value.block_io_parts();
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
                (),
                (pointer,),
                payload,
            ),
            Some(count) => library::checked_block_storage_call_v1::<T, _, _, _, _, _, ITEMS, true>(
                transpose_store_partial,
                destination,
                (),
                (pointer, count as i32),
                payload,
            ),
        }
    }
}
