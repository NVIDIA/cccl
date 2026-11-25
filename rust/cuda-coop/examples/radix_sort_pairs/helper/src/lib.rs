// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer both payload types from Load, then stably sort their associated pairs.
#![no_std]
use cuda_coop as coop;

/// Ascending pairs, descending pairs, then the two preserved input payloads.
pub type SortedTiles<K, V, const N: usize, const KA: usize, const VA: usize> = (
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
);

/// Return ascending and descending selected-digit pairs and both intact inputs.
///
/// Begin-only ascending uses the key width as its end. Descending supplies
/// both runtime bounds. Keys and values have independent minimum alignments.
pub fn sort_tile<
    K: coop::RadixKey,
    V: coop::RadixValue,
    const ITEMS_PER_THREAD: usize,
    const KEY_ALIGN: usize,
    const VALUE_ALIGN: usize,
>(
    block: coop::ThreadGroup<coop::Block>,
    keys: &[K],
    values: &[V],
    offset: usize,
    begin: i64,
    end: i64,
) -> SortedTiles<K, V, ITEMS_PER_THREAD, KEY_ALIGN, VALUE_ALIGN>
where
    coop::Alignment<KEY_ALIGN>: coop::SupportedAlignment,
    coop::Alignment<VALUE_ALIGN>: coop::SupportedAlignment,
{
    let mut input_keys = coop::ThreadData::<_, ITEMS_PER_THREAD, KEY_ALIGN>::new();
    let mut input_values = coop::ThreadData::<_, ITEMS_PER_THREAD, VALUE_ALIGN>::new();
    coop::load(
        block,
        keys,
        &mut input_keys,
        coop::LoadOptions::default().offset(offset),
    );
    coop::load(
        block,
        values,
        &mut input_values,
        coop::LoadOptions::default().offset(offset),
    );
    let (ascending_keys, ascending_values) = coop::radix_sort_pairs(
        block,
        &input_keys,
        &input_values,
        coop::RadixSortOptions::default().begin_bit(begin),
    );
    let (descending_keys, descending_values) = coop::radix_sort_pairs(
        block,
        &input_keys,
        &input_values,
        coop::RadixSortOptions::default()
            .begin_bit(begin)
            .end_bit(end)
            .order(coop::Descending),
    );
    (
        ascending_keys,
        ascending_values,
        descending_keys,
        descending_values,
        input_keys,
        input_values,
    )
}
