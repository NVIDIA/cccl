// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Decode a fresh window and the full stream with independent run/window extents.
#![no_std]
use cuda_coop as coop;

/// Fresh D3 window, all-lane numeric total and preserved N-item input planes.
pub type DecodedTile<const N: usize> = (
    coop::ThreadData<i64, 3, 32>,
    u32,
    coop::ThreadData<i64, N, 32>,
    coop::ThreadData<u64, N, 16>,
);

/// Load every input slot, then keep the window alive across the complete decode.
///
/// The returned total is reported as data; it is not used as a later collective
/// control. Destination ownership allows seven decoded elements per member.
pub fn decode_tile<const ITEMS_PER_THREAD: usize>(
    block: coop::ThreadGroup<coop::Block>,
    values: &[i64],
    lengths: &[u64],
    input_offset: usize,
    destination: &mut coop::DisjointSlice<'_, i64, coop::CollectiveTiles<7>>,
    destination_offset: u64,
) -> DecodedTile<ITEMS_PER_THREAD> {
    let mut input_values = coop::ThreadData::<_, ITEMS_PER_THREAD, 32>::new();
    let mut input_lengths = coop::ThreadData::<_, ITEMS_PER_THREAD, 16>::new();
    coop::load(
        block,
        values,
        &mut input_values,
        coop::LoadOptions::default().offset(input_offset),
    );
    coop::load(
        block,
        lengths,
        &mut input_lengths,
        coop::LoadOptions::default().offset(input_offset),
    );
    let window = coop::run_length_decode(
        block,
        &input_values,
        &input_lengths,
        coop::RunLengthDecodeOptions::<3>::default().decoded_window_offset_const::<196>(),
    );
    let total = coop::run_length_decode_into(
        block,
        &input_values,
        &input_lengths,
        destination,
        coop::RunLengthDecodeIntoOptions::<3>::default().destination_offset(destination_offset),
    );
    (window, total, input_values, input_lengths)
}
