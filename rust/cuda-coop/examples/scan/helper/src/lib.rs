// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A reusable, element-type-preserving block prefix helper.
#![no_std]
use cuda_coop as coop;

/// Write exclusive sums for the valid prefix of one block tile.
///
/// `zero` fills the missing tail before Scan, which requires every input slot
/// initialized. The caller supplies the additive identity for `T`.
pub fn prefix<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    valid: usize,
    zero: T,
) {
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
    let prefixes = coop::scan(block, &values, Default::default());
    coop::store(
        block,
        output,
        &prefixes,
        coop::StoreOptions::default()
            .offset(offset)
            .valid_items(valid),
    );
}
