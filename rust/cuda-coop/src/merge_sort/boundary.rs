// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{MergeSortGroup, MergeSortOptions, MergeSortOrder, MergeSortPrefix, MergeSortValue};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

fn ascending_full() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_ASCENDING_FULL")
}
fn descending_full() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_DESCENDING_FULL")
}
fn ascending_partial() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_ASCENDING_PARTIAL")
}
fn descending_partial() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_DESCENDING_PARTIAL")
}

fn warp_ascending_full() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_WARP_ASCENDING_FULL")
}
fn warp_descending_full() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_WARP_DESCENDING_FULL")
}
fn warp_ascending_partial() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_WARP_ASCENDING_PARTIAL")
}
fn warp_descending_partial() -> &'static str {
    library::registration!("COOP_MERGE_SORT_KEYS_WARP_DESCENDING_PARTIAL")
}

#[inline(always)]
pub(super) fn merge_sort_keys<T, G, O, P, S, const N: usize, const ALIGN: usize>(
    input: &ThreadData<T, N, ALIGN>,
    options: MergeSortOptions<T, O, P, S>,
) -> ThreadData<T, N, ALIGN>
where
    T: Numeric,
    O: MergeSortOrder,
    P: MergeSortPrefix<T>,
    S: TempStorageLike,
    G: MergeSortGroup<S>,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let _order = options.order;
    let storage = options.storage.into_raw();
    macro_rules! call {
        ($registration:ident) => {
            G::call::<T, _, _, _, _, N>($registration, storage, options.prefix, payload)
        };
    }
    // SAFETY: borrowed input and fresh same-shape output are distinct aligned
    // lane-local buffers with independent N-bit maps. The closed recipe checks
    // wide count bounds and ALL input bits, even at count zero, before numeric
    // reads. It sorts a private copy, marks only the written output prefix and
    // clears all other output bits. No pointer escapes. Sealed options select
    // the exact full/partial scalar ABI. The genuine marker proves complete
    // group participation and uniform controls, and plans scratch/reuse; the
    // provider validates geometry, integer capacity and full padded layout.
    unsafe {
        match (G::WARP, O::DESCENDING, P::PARTIAL) {
            (false, false, false) => call!(ascending_full),
            (false, true, false) => call!(descending_full),
            (false, false, true) => call!(ascending_partial),
            (false, true, true) => call!(descending_partial),
            (true, false, false) => call!(warp_ascending_full),
            (true, true, false) => call!(warp_descending_full),
            (true, false, true) => call!(warp_ascending_partial),
            (true, true, true) => call!(warp_descending_partial),
        }
    }
    output
}

fn pairs_ascending_full<V: MergeSortValue>() -> &'static str {
    V::ASCENDING_FULL_REGISTRATION
}
fn pairs_descending_full<V: MergeSortValue>() -> &'static str {
    V::DESCENDING_FULL_REGISTRATION
}
fn pairs_ascending_partial<V: MergeSortValue>() -> &'static str {
    V::ASCENDING_PARTIAL_REGISTRATION
}
fn pairs_descending_partial<V: MergeSortValue>() -> &'static str {
    V::DESCENDING_PARTIAL_REGISTRATION
}
fn pairs_warp_ascending_full<V: MergeSortValue>() -> &'static str {
    V::WARP_ASCENDING_FULL_REGISTRATION
}
fn pairs_warp_descending_full<V: MergeSortValue>() -> &'static str {
    V::WARP_DESCENDING_FULL_REGISTRATION
}
fn pairs_warp_ascending_partial<V: MergeSortValue>() -> &'static str {
    V::WARP_ASCENDING_PARTIAL_REGISTRATION
}
fn pairs_warp_descending_partial<V: MergeSortValue>() -> &'static str {
    V::WARP_DESCENDING_PARTIAL_REGISTRATION
}

#[inline(always)]
pub(super) fn merge_sort_pairs<K, V, G, O, P, S, const N: usize, const KA: usize, const VA: usize>(
    keys: &ThreadData<K, N, KA>,
    values: &ThreadData<V, N, VA>,
    options: MergeSortOptions<K, O, P, S>,
) -> (ThreadData<K, N, KA>, ThreadData<V, N, VA>)
where
    K: Numeric,
    V: MergeSortValue,
    O: MergeSortOrder,
    P: MergeSortPrefix<K>,
    S: TempStorageLike,
    G: MergeSortGroup<S>,
    Alignment<KA>: SupportedAlignment,
    Alignment<VA>: SupportedAlignment,
{
    let mut output_keys = ThreadData::new();
    let mut output_values = ThreadData::new();
    let (keys_pointer, keys_bits) = keys.block_io_read_parts();
    let (values_pointer, values_bits) = values.block_io_read_parts();
    let (output_keys_pointer, output_keys_bits) = output_keys.block_io_parts();
    let (output_values_pointer, output_values_bits) = output_values.block_io_parts();
    let payload = (
        keys_pointer,
        output_keys_pointer,
        values_pointer,
        output_values_pointer,
        keys_bits,
        output_keys_bits,
        values_bits,
        output_values_bits,
    );
    let _order = options.order;
    let storage = options.storage.into_raw();
    macro_rules! call {
        ($registration:ident) => {
            G::call::<K, _, _, _, _, N>($registration::<V>, storage, options.prefix, payload)
        };
    }
    // SAFETY: shared inputs and two fresh same-shape outputs are live, aligned
    // lane-local buffers with independent N-bit maps. The closed recipe checks
    // wide count bounds and EVERY bit in both inputs before reading either
    // array, including at count zero. It sorts private copies, writes paired
    // valid-prefix outputs, and clears all other output bits. No pointer
    // escapes. Sealed value constants and options preserve exact K/V/count/
    // sentinel ABI. The genuine marker proves participation, uniform controls
    // and storage reuse; the provider queries complete pair scratch.
    unsafe {
        match (G::WARP, O::DESCENDING, P::PARTIAL) {
            (false, false, false) => call!(pairs_ascending_full),
            (false, true, false) => call!(pairs_descending_full),
            (false, false, true) => call!(pairs_ascending_partial),
            (false, true, true) => call!(pairs_descending_partial),
            (true, false, false) => call!(pairs_warp_ascending_full),
            (true, true, false) => call!(pairs_warp_descending_full),
            (true, false, true) => call!(pairs_warp_ascending_partial),
            (true, true, true) => call!(pairs_warp_descending_partial),
        }
    }
    (output_keys, output_values)
}
