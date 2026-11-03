// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Generic four-way unsorted TopK with independently aligned key/value inputs.
#![no_std]
use cuda_coop as coop;

/// Min/max keys, min/max pairs, and the two unchanged input payloads.
pub type SelectedTiles<K, V, const N: usize, const KA: usize, const VA: usize> = (
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
    coop::ThreadData<K, N, KA>,
    coop::ThreadData<V, N, VA>,
);

/// Exercise static k/full, runtime k/valid and static k/valid registrations.
pub fn select_tile<
    K: coop::Numeric,
    V: coop::TopKValue,
    const N: usize,
    const KA: usize,
    const VA: usize,
>(
    block: coop::ThreadGroup<coop::Block>,
    keys: &[K],
    values: &[V],
    offset: usize,
    k: i64,
    valid: i64,
) -> SelectedTiles<K, V, N, KA, VA>
where
    coop::Alignment<KA>: coop::SupportedAlignment,
    coop::Alignment<VA>: coop::SupportedAlignment,
{
    let mut input_k = coop::ThreadData::<_, N, KA>::new();
    let mut input_v = coop::ThreadData::<_, N, VA>::new();
    coop::load(
        block,
        keys,
        &mut input_k,
        coop::LoadOptions::default().offset(offset),
    );
    coop::load(
        block,
        values,
        &mut input_v,
        coop::LoadOptions::default().offset(offset),
    );
    let min_k = coop::topk_min_keys(
        block,
        &input_k,
        coop::TopKOptions::default().k_const::<17>(),
    );
    let max_k = coop::topk_max_keys(
        block,
        &input_k,
        coop::TopKOptions::default().k(k).valid_items(valid),
    );
    let (min_pk, min_pv) = coop::topk_min_pairs(
        block,
        &input_k,
        &input_v,
        coop::TopKOptions::default()
            .valid_items_const::<31>()
            .k_const::<17>(),
    );
    let (max_pk, max_pv) = coop::topk_max_pairs(
        block,
        &input_k,
        &input_v,
        coop::TopKOptions::default().valid_items(valid).k(k),
    );
    (
        min_k, max_k, min_pk, min_pv, max_pk, max_pv, input_k, input_v,
    )
}
