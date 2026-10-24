// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A generic eight-lane batched sum with a calculated output payload.
#![no_std]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use cuda_coop as coop;

/// Reduce each input slot across eight lanes and store its one striped result.
///
/// `input_offset` and `output_offset` locate this block's input and padded output
/// regions. Warp Load/Store add each subgroup's origin themselves. `valid` is
/// the initialized source prefix for this subgroup; `zero` fills its short tail
/// before reduction. It must be the additive identity for `T`.
pub fn batches<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    group: coop::ThreadGroup<coop::ThreadsWithinWarp<8>>,
    input: &[T],
    output: &mut coop::DisjointSlice<
        '_,
        T,
        coop::CollectiveTiles<{ coop::batched_extent(ITEMS_PER_THREAD, 8) }>,
    >,
    input_offset: usize,
    output_offset: usize,
    valid: usize,
    zero: T,
) where
    [(); coop::batched_extent(ITEMS_PER_THREAD, 8)]:,
{
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        group,
        input,
        &mut values,
        coop::LoadOptions::default()
            .offset(input_offset)
            .valid_items(valid)
            .oob_default(zero),
    );
    let totals = coop::reduce_batched(group, &values, Default::default());
    // Every member participates, including lanes with no initialized result.
    // Striped Store's count selects exactly the B owned batch results.
    coop::store(
        group,
        output,
        &totals,
        coop::StoreOptions::default()
            .algorithm(coop::Striped)
            .offset(output_offset)
            .valid_items(ITEMS_PER_THREAD),
    );
}
