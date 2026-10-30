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

/// Reduce a fully initialized i32 tile with a built-in or callable operator.
///
/// Every block member calls this helper. The associative operator is retained
/// as its concrete type. Auto-sync scratch completes each reduction before
/// the caller reuses it; the result is valid only at block rank zero.
/// A captured i32 must have the same proved value throughout the block.
pub fn reduce_values<Op: coop::ReduceOperator<i32>, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    values: &coop::ThreadData<i32, ITEMS_PER_THREAD>,
    op: Op,
    scratch: &mut coop::TempStorage<0, 0, true, false>,
) -> Option<i32> {
    coop::reduce(
        block,
        values,
        coop::ReduceOptions::default()
            .binary_op(op)
            .algorithm(coop::Raking)
            .temp_storage(scratch),
    )
}
