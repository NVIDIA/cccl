// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Shift a full block tile while retaining its element type and static extent.
#![no_std]
use cuda_coop as coop;

/// Load a tile, shift it, initialize the exposed edge, and store the result.
///
/// Every thread participates. `edge` supplies the one missing boundary value;
/// inspecting validity and writing that slot use the ordinary safe payload API.
pub fn shuffle_tile<T: coop::Numeric, M: coop::ShuffleMode, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    edge: T,
    options: coop::ShuffleOptions<M>,
) {
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default().offset(offset),
    );
    let mut shifted = coop::shuffle(block, &values, options);
    let mut item = 0;
    while item < ITEMS_PER_THREAD {
        if shifted.get(item).is_none() {
            shifted.write(item, edge);
        }
        item += 1;
    }
    coop::store(
        block,
        output,
        &shifted,
        coop::StoreOptions::default().offset(offset),
    );
}
