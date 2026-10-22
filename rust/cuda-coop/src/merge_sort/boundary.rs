// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{MergeSortGroup, MergeSortOptions, MergeSortOrder, MergeSortPrefix};
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
