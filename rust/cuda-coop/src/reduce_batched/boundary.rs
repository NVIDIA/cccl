// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{BatchedOutputLayout, ReduceBatchedGroup, ReduceBatchedOptions, batched_extent};
use crate::{Alignment, Numeric, ReduceOperator, SupportedAlignment, ThreadData};
use cuda_device::library;

macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (sum_striped, "COOP_REDUCE_BATCHED_SUM_STRIPED"),
    (multiplies_striped, "COOP_REDUCE_BATCHED_MULTIPLIES_STRIPED"),
    (min_striped, "COOP_REDUCE_BATCHED_MIN_STRIPED"),
    (max_striped, "COOP_REDUCE_BATCHED_MAX_STRIPED"),
    (bit_and_striped, "COOP_REDUCE_BATCHED_BIT_AND_STRIPED"),
    (bit_or_striped, "COOP_REDUCE_BATCHED_BIT_OR_STRIPED"),
    (bit_xor_striped, "COOP_REDUCE_BATCHED_BIT_XOR_STRIPED"),
    (sum_blocked, "COOP_REDUCE_BATCHED_SUM_BLOCKED"),
    (multiplies_blocked, "COOP_REDUCE_BATCHED_MULTIPLIES_BLOCKED"),
    (min_blocked, "COOP_REDUCE_BATCHED_MIN_BLOCKED"),
    (max_blocked, "COOP_REDUCE_BATCHED_MAX_BLOCKED"),
    (bit_and_blocked, "COOP_REDUCE_BATCHED_BIT_AND_BLOCKED"),
    (bit_or_blocked, "COOP_REDUCE_BATCHED_BIT_OR_BLOCKED"),
    (bit_xor_blocked, "COOP_REDUCE_BATCHED_BIT_XOR_BLOCKED"),
);

struct Selection<T, Op, L>(core::marker::PhantomData<(T, Op, L)>);
impl<T: Numeric, Op: ReduceOperator<T>, L: BatchedOutputLayout> Selection<T, Op, L> {
    const KIND: u8 = 7 * L::KIND + Op::KIND;
}

#[inline(always)]
pub(super) fn reduce<T, G, Op, L, const ITEMS: usize, const ALIGN: usize>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: ReduceBatchedOptions<T, Op, L>,
) -> ThreadData<T, { batched_extent(ITEMS, G::WIDTH) }>
where
    T: Numeric,
    G: ReduceBatchedGroup,
    Op: ReduceOperator<T>,
    L: BatchedOutputLayout,
    Alignment<ALIGN>: SupportedAlignment,
    [(); batched_extent(ITEMS, G::WIDTH)]:,
{
    let mut output = ThreadData::<T, { batched_extent(ITEMS, G::WIDTH) }>::new();
    let (input, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input, output_pointer, input_bits, output_bits);
    let _operator = options.binary_op;
    // SAFETY: input and fresh output are distinct live lane-local buffers with
    // their actual extents and alignments. The typed recipe derives the same
    // output extent from input extent/width, preflights every input bit before
    // numeric reads, and preserves input. It clears output bits and copies only
    // owned initialized results, then marks their bits. No invalid CUB output
    // is read. Pointers cannot escape the synchronous call. The genuine marker
    // checks participation and compiler-planned scratch allocation/reuse.
    unsafe {
        match Selection::<T, Op, L>::KIND {
            0 => G::call::<T, _, _, ITEMS>(sum_striped, payload),
            1 => G::call::<T, _, _, ITEMS>(multiplies_striped, payload),
            2 => G::call::<T, _, _, ITEMS>(min_striped, payload),
            3 => G::call::<T, _, _, ITEMS>(max_striped, payload),
            4 => G::call::<T, _, _, ITEMS>(bit_and_striped, payload),
            5 => G::call::<T, _, _, ITEMS>(bit_or_striped, payload),
            6 => G::call::<T, _, _, ITEMS>(bit_xor_striped, payload),
            7 => G::call::<T, _, _, ITEMS>(sum_blocked, payload),
            8 => G::call::<T, _, _, ITEMS>(multiplies_blocked, payload),
            9 => G::call::<T, _, _, ITEMS>(min_blocked, payload),
            10 => G::call::<T, _, _, ITEMS>(max_blocked, payload),
            11 => G::call::<T, _, _, ITEMS>(bit_and_blocked, payload),
            12 => G::call::<T, _, _, ITEMS>(bit_or_blocked, payload),
            13 => G::call::<T, _, _, ITEMS>(bit_xor_blocked, payload),
            _ => panic!(),
        }
    }
    output
}
