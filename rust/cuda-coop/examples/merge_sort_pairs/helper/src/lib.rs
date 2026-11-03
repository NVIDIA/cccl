// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer both payload types from Load, then merge-sort their associated pairs.
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

/// Return full ascending pairs, partial descending pairs and both intact inputs.
///
/// Count and key-typed sentinel agree across the Block. Keys and values have
/// independent minimum alignments; every input slot is initialized.
pub fn sort_tile<
    K: coop::Numeric,
    V: coop::MergeSortValue,
    const ITEMS_PER_THREAD: usize,
    const KEY_ALIGN: usize,
    const VALUE_ALIGN: usize,
>(
    block: coop::ThreadGroup<coop::Block>,
    keys: &[K],
    values: &[V],
    offset: usize,
    valid_items: i64,
    sentinel: K,
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
    let (ascending_keys, ascending_values) =
        coop::merge_sort_pairs(block, &input_keys, &input_values, Default::default());
    let (descending_keys, descending_values) = coop::merge_sort_pairs(
        block,
        &input_keys,
        &input_values,
        coop::MergeSortOptions::default()
            .valid_items(valid_items, sentinel)
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
