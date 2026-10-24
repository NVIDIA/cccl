// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! A reusable generic helper shared by downstream CUDA kernels.
#![no_std]
use cuda_coop as coop;

/// Copy a partial block tile, preserving the source's element width.
pub fn copy<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    valid: usize,
    default: T,
) {
    copy_with_algorithm::<T, ITEMS_PER_THREAD, _>(
        block,
        input,
        output,
        offset,
        valid,
        default,
        coop::Direct,
    );
}

/// Copy a partial tile using a compile-time algorithm chosen by the caller.
pub fn copy_with_algorithm<T: coop::Numeric, const ITEMS_PER_THREAD: usize, A: coop::Algorithm>(
    block: coop::ThreadGroup<coop::Block>,
    input: &[T],
    output: &mut coop::DisjointSlice<'_, T, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
    offset: usize,
    valid: usize,
    default: T,
    algorithm: A,
) {
    let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
    coop::load(
        block,
        input,
        &mut values,
        coop::LoadOptions::default()
            .valid_items(valid)
            .oob_default(default)
            .offset(offset)
            .algorithm(algorithm),
    );
    coop::store(
        block,
        output,
        &mut values,
        coop::StoreOptions::default()
            .valid_items(valid)
            .offset(offset)
            .algorithm(algorithm),
    );
}
