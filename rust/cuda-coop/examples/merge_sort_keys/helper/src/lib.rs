// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Load inferred numeric keys and return fresh full and partial sort results.
#![no_std]
use cuda_coop as coop;

/// Full ascending, partial descending, then the preserved input payload.
pub type SortedTile<T, const N: usize, const ALIGN: usize> = (
    coop::ThreadData<T, N, ALIGN>,
    coop::ThreadData<T, N, ALIGN>,
    coop::ThreadData<T, N, ALIGN>,
);

/// Sort two ways while borrowing the same fully initialized key tile.
///
/// Count and sentinel are uniform; the sentinel follows valid descending keys.
pub fn sort_tile<T: coop::Numeric, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    block: coop::ThreadGroup<coop::Block>,
    keys: &[T],
    offset: usize,
    valid_items: i64,
    sentinel: T,
) -> SortedTile<T, ITEMS_PER_THREAD, ALIGN>
where
    coop::Alignment<ALIGN>: coop::SupportedAlignment,
{
    let mut input = coop::ThreadData::<_, ITEMS_PER_THREAD, ALIGN>::new();
    coop::load(
        block,
        keys,
        &mut input,
        coop::LoadOptions::default().offset(offset),
    );
    let ascending = coop::merge_sort_keys(block, &input, Default::default());
    let descending = coop::merge_sort_keys(
        block,
        &input,
        coop::MergeSortOptions::default()
            .valid_items(valid_items, sentinel)
            .order(coop::Descending),
    );
    (ascending, descending, input)
}

/// Sort full i32 Keys with either a built-in order or a stateless comparator.
///
/// The comparator defines a strict weak ordering. Its concrete type and the
/// selected storage are retained through this unchanged generic order bound.
pub fn sort_values<O: coop::MergeSortOrder, S: coop::TempStorageLike, const N: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &coop::ThreadData<i32, N>,
    options: coop::MergeSortOptions<i32, O, (), S>,
) -> coop::ThreadData<i32, N> {
    coop::merge_sort_keys(block, input, options)
}

/// Order even keys before odd keys, then order each parity numerically.
pub fn even_first(left: i32, right: i32) -> bool {
    (left & 1) < (right & 1) || ((left & 1) == (right & 1) && left < right)
}
