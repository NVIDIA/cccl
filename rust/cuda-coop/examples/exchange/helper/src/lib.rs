// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A reusable, element-type-preserving block layout conversion helper.
#![no_std]
use cuda_coop as coop;

/// Copy a tile through striped input and blocked output layouts.
///
/// `fill` initializes the missing tail before Exchange. Only the valid prefix
/// is stored, so any value of `T` may fill slots outside that prefix.
pub fn exchange_tile<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    valid: usize,
    fill: T,
) {
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default()
            .algorithm(coop::Striped)
            .offset(offset)
            .valid_items(valid)
            .oob_default(fill),
    );
    let blocked = coop::exchange(block, &values, Default::default());
    coop::store(
        block,
        output,
        &blocked,
        coop::StoreOptions::default()
            .offset(offset)
            .valid_items(valid),
    );
}
