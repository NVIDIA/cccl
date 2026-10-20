// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Private Direct integration boundary. This is not a public safe operation.
//! The checked marker proves participation, controls, allocation origin and
//! epochs. CUB uses opaque cells; the provider owns mask reset and preflight.
use crate::{Alignment, Block, Numeric, SupportedAlignment, ThreadData, ThreadGroup};
use cuda_device::{CollectiveTiles, DisjointSlice, library, thread};

#[inline(always)]
fn extent<T, const ITEMS: usize>(len: usize, offset: usize, valid: usize) {
    crate::carrier::extent::<ITEMS>(len, offset, valid);
    assert!(offset <= isize::MAX as usize / core::mem::size_of::<T>());
}

#[inline(always)]
fn tile<const ITEMS: usize>() -> usize {
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    assert!(ITEMS > 0 && threads <= usize::MAX / ITEMS);
    threads * ITEMS
}

#[inline(always)]
pub fn load_full<T: Numeric, const ITEMS: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    source: &[T],
    offset: usize,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    extent::<T, ITEMS>(source.len(), offset, tile::<ITEMS>());
    let payload = values.block_io_parts();
    // SAFETY: checked bounds precede pointer formation, source is the pointer's
    // allocation witness, and the provider initializes every cell and flag.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
            crate::validity_load_full,
            source,
            (source.as_ptr().add(offset),),
            payload,
        );
    }
}

#[inline(always)]
pub fn load_masked<T: Numeric, const ITEMS: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    source: &[T],
    offset: usize,
    valid: usize,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    extent::<T, ITEMS>(source.len(), offset, valid);
    let payload = values.block_io_parts();
    // SAFETY: only the checked prefix is read. The provider replaces all flags
    // and copies numeric bytes only for initialized cells.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
            crate::validity_load_masked,
            source,
            (source.as_ptr().add(offset), valid as i32),
            payload,
        );
    }
}

#[inline(always)]
pub fn load_filled<T: Numeric, const ITEMS: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    source: &[T],
    offset: usize,
    valid: usize,
    default: T,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    extent::<T, ITEMS>(source.len(), offset, valid);
    let payload = values.block_io_parts();
    // SAFETY: bounds and typed default match the exact native ABI. The provider
    // initializes every cell, including the out-of-prefix defaults.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
            crate::validity_load_filled,
            source,
            (source.as_ptr().add(offset), valid as i32, default),
            payload,
        );
    }
}

#[inline(always)]
pub fn store_full<T: Numeric, const ITEMS: usize, const ALIGN: usize, const CAPACITY: usize>(
    _group: ThreadGroup<Block>,
    target: &mut DisjointSlice<T, CollectiveTiles<CAPACITY>>,
    offset: usize,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    let valid = tile::<ITEMS>();
    extent::<T, ITEMS>(target.len(), offset, valid);
    crate::carrier::containment::<CAPACITY>(offset, valid);
    let pointer = target.as_mut_ptr();
    let payload = values.block_io_parts();
    // SAFETY: this carrier owns the checked range. The provider checks every
    // written slot's validity bit before interpreting its bytes as T.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
            crate::validity_store_full,
            target,
            (pointer.add(offset),),
            payload,
        );
    }
}

#[inline(always)]
pub fn store_partial<T: Numeric, const ITEMS: usize, const ALIGN: usize, const CAPACITY: usize>(
    _group: ThreadGroup<Block>,
    target: &mut DisjointSlice<T, CollectiveTiles<CAPACITY>>,
    offset: usize,
    valid: usize,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    extent::<T, ITEMS>(target.len(), offset, valid);
    crate::carrier::containment::<CAPACITY>(offset, valid);
    let pointer = target.as_mut_ptr();
    let payload = values.block_io_parts();
    // SAFETY: the guarded native iterator forms an effective pointer only for
    // a written slot. Preflight covers that same domain, with no tail reads.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, true>(
            crate::validity_store_partial,
            target,
            (pointer.add(offset), valid as i32),
            payload,
        );
    }
}

#[inline(always)]
pub fn reload_filled<T: Numeric, const ITEMS: usize, const ALIGN: usize, const CAPACITY: usize>(
    _group: ThreadGroup<Block>,
    source: &mut DisjointSlice<T, CollectiveTiles<CAPACITY>>,
    offset: usize,
    valid: usize,
    default: T,
    values: &mut ThreadData<T, ITEMS, ALIGN>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    extent::<T, ITEMS>(source.len(), offset, valid);
    crate::carrier::containment::<CAPACITY>(offset, valid);
    let pointer = source.as_mut_ptr();
    let payload = values.block_io_parts();
    // SAFETY: the immutable witness binds the same allocation. A prior Store
    // requires explicit block completion, enforced by the compiler.
    unsafe {
        library::checked_block_resource_call_v1::<T, _, _, _, _, ITEMS, false>(
            crate::validity_load_filled,
            &*source,
            (pointer.add(offset).cast_const(), valid as i32, default),
            payload,
        );
    }
}
