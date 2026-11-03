// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Private storage integration using the unchanged R01 numeric provider.
use cuda_device::{CollectiveTiles, DisjointSlice, library};

/// # Safety
/// All lanes agree on source, offset, valid count and storage origin. The source
/// contains the full accessed block range; the filled operation initializes all
/// lane payload slots. The root is exactly one 128-thread block per owned tile.
#[inline(never)]
pub unsafe fn load<const ITEMS: usize, const TRANSPOSE: bool, Storage>(
    source: &[i32],
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
    storage: Storage,
) {
    assert!(valid <= 128 * ITEMS && offset <= source.len() && valid <= source.len() - offset);
    unsafe {
        if TRANSPOSE {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, false>(
                super::transpose_load_filled,
                source,
                storage,
                (source.as_ptr().add(offset), valid as i32, -7_i32),
                (values.as_mut_ptr(),),
            );
        } else {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, false>(
                super::direct_load_filled,
                source,
                storage,
                (source.as_ptr().add(offset), valid as i32, -7_i32),
                (values.as_mut_ptr(),),
            );
        }
    }
}

/// # Safety
/// All lanes agree on target, offset, valid count and storage origin. Offset is
/// this block's canonical 512-element tile start; payload values are initialized.
#[inline(never)]
pub unsafe fn store<const ITEMS: usize, const TRANSPOSE: bool, Storage>(
    target: &mut DisjointSlice<i32, CollectiveTiles<4>>,
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
    storage: Storage,
) {
    assert!(valid <= 128 * ITEMS && offset <= target.len() && valid <= target.len() - offset);
    assert!(offset == cuda_device::thread::blockIdx_x() as usize * 512);
    let pointer = target.as_mut_ptr();
    unsafe {
        if TRANSPOSE {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, true>(
                super::transpose_store_partial,
                target,
                storage,
                (pointer.add(offset), valid as i32),
                (values.as_mut_ptr(),),
            );
        } else {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, true>(
                super::direct_store_partial,
                target,
                storage,
                (pointer.add(offset), valid as i32),
                (values.as_mut_ptr(),),
            );
        }
    }
}

/// # Safety
/// The carrier and range obey `store`'s ownership contract. A prior write must
/// be completed by an explicit or genuinely required planned block barrier.
#[inline(never)]
pub unsafe fn reload<const ITEMS: usize, const TRANSPOSE: bool, Storage>(
    source: &mut DisjointSlice<i32, CollectiveTiles<4>>,
    offset: usize,
    valid: usize,
    values: &mut [i32; ITEMS],
    storage: Storage,
) {
    assert!(valid <= 128 * ITEMS && offset <= source.len() && valid <= source.len() - offset);
    let pointer = source.as_mut_ptr();
    unsafe {
        if TRANSPOSE {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, false>(
                super::transpose_load_filled,
                &*source,
                storage,
                (pointer.add(offset).cast_const(), valid as i32, -7_i32),
                (values.as_mut_ptr(),),
            );
        } else {
            library::checked_block_storage_call_v1::<i32, _, _, _, _, _, ITEMS, false>(
                super::direct_load_filled,
                &*source,
                storage,
                (pointer.add(offset).cast_const(), valid as i32, -7_i32),
                (values.as_mut_ptr(),),
            );
        }
    }
}
