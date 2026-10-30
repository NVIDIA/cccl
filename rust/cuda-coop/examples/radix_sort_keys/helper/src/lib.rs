// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer generic keys from Load, then sort full and selected bit intervals.
#![no_std]
use cuda_coop as coop;

/// Return three independent sorts and the preserved input payload.
///
/// The begin-only ascending sort uses the full key width as its end. The
/// descending sort uses both runtime endpoints. Every input slot is loaded.
pub fn sort_tile<T: coop::RadixKey, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    offset: usize,
    begin: i64,
    end: i64,
) -> (
    coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
)
where
    coop::Alignment<ALIGN>: coop::SupportedAlignment,
{
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD, ALIGN>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default().offset(offset),
    );
    let full = coop::radix_sort_keys(block, &values, Default::default());
    let selected = coop::radix_sort_keys(
        block,
        &values,
        coop::RadixSortOptions::default().begin_bit(begin),
    );
    let descending = coop::radix_sort_keys(
        block,
        &values,
        coop::RadixSortOptions::default()
            .begin_bit(begin)
            .end_bit(end)
            .order(coop::Descending),
    );
    (full, selected, descending, values)
}
