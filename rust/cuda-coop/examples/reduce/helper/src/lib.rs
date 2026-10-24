// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A reusable block sum that preserves the input element type.
#![no_std]
use cuda_coop as coop;

/// Sum one block tile, returning its total only at block rank zero.
///
/// Every block member calls this helper. `zero` fills the missing input tail
/// before reduction, which requires every payload slot initialized. The caller
/// supplies the additive identity for `T`.
pub fn sum_tile<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    offset: usize,
    valid: usize,
    zero: T,
) -> Option<T> {
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
    coop::sum(block, &values, Default::default())
}
