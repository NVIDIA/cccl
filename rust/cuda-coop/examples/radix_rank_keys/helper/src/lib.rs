// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer the key type from Load and return independent i32 rank payloads.
#![no_std]
use cuda_coop as coop;

/// Ascending ranks, descending ranks, then the preserved key payload.
pub type RankedTile<T, const N: usize, const ALIGN: usize> = (
    coop::ThreadData<i32, N>,
    coop::ThreadData<i32, N>,
    coop::ThreadData<T, N, ALIGN>,
);

/// Rank a sign-touching three-bit digit with static and runtime begin options.
///
/// The static interval derives its width. The runtime interval supplies width
/// explicitly. Both results have natural i32 alignment, independent of input.
pub fn rank_tile<T: coop::RadixKey, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    block: coop::ThreadGroup<coop::Block>,
    keys: &[T],
    offset: usize,
    begin: i64,
) -> RankedTile<T, ITEMS_PER_THREAD, ALIGN>
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
    let ascending = coop::radix_rank_keys(
        block,
        &input,
        coop::RadixRankOptions::default()
            .begin_bit_const::<61>()
            .end_bit_const::<64>(),
    );
    let descending = coop::radix_rank_keys(
        block,
        &input,
        coop::RadixRankOptions::default()
            .begin_bit(begin)
            .radix_bits::<3>()
            .end_bit_const::<64>()
            .order(coop::Descending),
    );
    (ascending, descending, input)
}
