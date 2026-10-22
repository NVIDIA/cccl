// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Infer sample types while returning independently shaped Histogram counters.
#![no_std]
use cuda_coop as coop;

/// Load a full tile and count 65 bins into two striped u64 counters per thread.
///
/// Every sample must be in 0..65. Sort works on a private native copy, so the
/// returned input payload remains readable and unchanged.
pub fn count_tile<T: coop::HistogramSample, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    offset: usize,
) -> (
    coop::ThreadData<u64, 2>,
    coop::ThreadData<T, ITEMS_PER_THREAD>,
) {
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default().offset(offset),
    );
    let counts = coop::histogram(
        block,
        &values,
        coop::HistogramOptions::<65>::default()
            .counter_type::<u64>()
            .bins_per_thread::<2>()
            .algorithm(coop::Sort),
    );
    (counts, values)
}
