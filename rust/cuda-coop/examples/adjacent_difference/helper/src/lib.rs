// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Compute adjacent differences with inferred payload types and a static extent.
#![no_std]
use cuda_coop as coop;

/// Load a fully readable tile, subtract adjacent active items, and store the active prefix.
///
/// `zero` initializes out-of-bounds input slots before the operation. The count
/// limits subtraction, not input readability; the initialized suffix is copied.
/// Every thread participates, including threads beyond the final active item.
pub fn difference_tile<T, D, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    zero: T,
    direction: D,
) where
    T: coop::Numeric,
    D: coop::AdjacentDifferenceDirection<T, usize, (), ()>,
{
    let capacity = block.count() as usize * ITEMS_PER_THREAD;
    let remaining = if offset <= input.len() {
        input.len() - offset
    } else {
        0
    };
    let valid = remaining.min(capacity);
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default()
            .offset(offset)
            .valid_items(valid)
            .oob_default(zero),
    );
    let differences = coop::adjacent_difference(
        block,
        &values,
        coop::AdjacentDifferenceOptions::default()
            .direction(direction)
            .valid_items(valid),
    );
    coop::store(
        block,
        output,
        &differences,
        coop::StoreOptions::default()
            .offset(offset)
            .valid_items(valid),
    );
}
