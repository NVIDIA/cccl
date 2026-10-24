// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer input types while returning distinct head and tail flags for each tile.
#![no_std]
use cuda_coop as coop;

/// Load a complete tile and return head flags, tail flags and preserved input.
///
/// Every input slot participates. Outside neighbors have the exact input type
/// and must be uniform within the Block. The two fresh flag payloads contain i32.
pub fn flag_tile<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    offset: usize,
    predecessor: T,
    successor: T,
) -> (
    coop::ThreadData<i32, ITEMS_PER_THREAD>,
    coop::ThreadData<i32, ITEMS_PER_THREAD>,
    coop::ThreadData<T, ITEMS_PER_THREAD>,
) {
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default().offset(offset),
    );
    let (heads, tails) = coop::discontinuity(
        block,
        &values,
        coop::DiscontinuityOptions::default()
            .mode(coop::HeadsAndTails)
            .tile_predecessor_item(predecessor)
            .tile_successor_item(successor),
    );
    (heads, tails, values)
}
