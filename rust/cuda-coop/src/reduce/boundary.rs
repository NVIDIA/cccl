// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{ReduceAlgorithm, ReduceGroup, ReduceOptions, sealed::Count};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

// Each static recipe keeps a concrete function-item registration identity.
macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (
        raking_commutative_only_sum_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_SUM_SCALAR"
    ),
    (raking_sum_scalar, "COOP_REDUCE_RAKING_SUM_SCALAR"),
    (
        warp_reductions_sum_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_SUM_SCALAR"
    ),
    (warp_sum_scalar, "COOP_REDUCE_WARP_SUM_SCALAR"),
    (
        raking_commutative_only_multiplies_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MULTIPLIES_SCALAR"
    ),
    (
        raking_multiplies_scalar,
        "COOP_REDUCE_RAKING_MULTIPLIES_SCALAR"
    ),
    (
        warp_reductions_multiplies_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_MULTIPLIES_SCALAR"
    ),
    (warp_multiplies_scalar, "COOP_REDUCE_WARP_MULTIPLIES_SCALAR"),
    (
        raking_commutative_only_min_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MIN_SCALAR"
    ),
    (raking_min_scalar, "COOP_REDUCE_RAKING_MIN_SCALAR"),
    (
        warp_reductions_min_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_MIN_SCALAR"
    ),
    (warp_min_scalar, "COOP_REDUCE_WARP_MIN_SCALAR"),
    (
        raking_commutative_only_max_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MAX_SCALAR"
    ),
    (raking_max_scalar, "COOP_REDUCE_RAKING_MAX_SCALAR"),
    (
        warp_reductions_max_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_MAX_SCALAR"
    ),
    (warp_max_scalar, "COOP_REDUCE_WARP_MAX_SCALAR"),
    (
        raking_commutative_only_bit_and_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_AND_SCALAR"
    ),
    (raking_bit_and_scalar, "COOP_REDUCE_RAKING_BIT_AND_SCALAR"),
    (
        warp_reductions_bit_and_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_AND_SCALAR"
    ),
    (warp_bit_and_scalar, "COOP_REDUCE_WARP_BIT_AND_SCALAR"),
    (
        raking_commutative_only_bit_or_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_OR_SCALAR"
    ),
    (raking_bit_or_scalar, "COOP_REDUCE_RAKING_BIT_OR_SCALAR"),
    (
        warp_reductions_bit_or_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_OR_SCALAR"
    ),
    (warp_bit_or_scalar, "COOP_REDUCE_WARP_BIT_OR_SCALAR"),
    (
        raking_commutative_only_bit_xor_scalar,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_XOR_SCALAR"
    ),
    (raking_bit_xor_scalar, "COOP_REDUCE_RAKING_BIT_XOR_SCALAR"),
    (
        warp_reductions_bit_xor_scalar,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_XOR_SCALAR"
    ),
    (warp_bit_xor_scalar, "COOP_REDUCE_WARP_BIT_XOR_SCALAR"),
    (
        raking_commutative_only_sum_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_SUM_PREFIX"
    ),
    (raking_sum_prefix, "COOP_REDUCE_RAKING_SUM_PREFIX"),
    (
        warp_reductions_sum_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_SUM_PREFIX"
    ),
    (warp_sum_prefix, "COOP_REDUCE_WARP_SUM_PREFIX"),
    (
        raking_commutative_only_multiplies_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MULTIPLIES_PREFIX"
    ),
    (
        raking_multiplies_prefix,
        "COOP_REDUCE_RAKING_MULTIPLIES_PREFIX"
    ),
    (
        warp_reductions_multiplies_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_MULTIPLIES_PREFIX"
    ),
    (warp_multiplies_prefix, "COOP_REDUCE_WARP_MULTIPLIES_PREFIX"),
    (
        raking_commutative_only_min_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MIN_PREFIX"
    ),
    (raking_min_prefix, "COOP_REDUCE_RAKING_MIN_PREFIX"),
    (
        warp_reductions_min_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_MIN_PREFIX"
    ),
    (warp_min_prefix, "COOP_REDUCE_WARP_MIN_PREFIX"),
    (
        raking_commutative_only_max_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MAX_PREFIX"
    ),
    (raking_max_prefix, "COOP_REDUCE_RAKING_MAX_PREFIX"),
    (
        warp_reductions_max_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_MAX_PREFIX"
    ),
    (warp_max_prefix, "COOP_REDUCE_WARP_MAX_PREFIX"),
    (
        raking_commutative_only_bit_and_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_AND_PREFIX"
    ),
    (raking_bit_and_prefix, "COOP_REDUCE_RAKING_BIT_AND_PREFIX"),
    (
        warp_reductions_bit_and_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_AND_PREFIX"
    ),
    (warp_bit_and_prefix, "COOP_REDUCE_WARP_BIT_AND_PREFIX"),
    (
        raking_commutative_only_bit_or_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_OR_PREFIX"
    ),
    (raking_bit_or_prefix, "COOP_REDUCE_RAKING_BIT_OR_PREFIX"),
    (
        warp_reductions_bit_or_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_OR_PREFIX"
    ),
    (warp_bit_or_prefix, "COOP_REDUCE_WARP_BIT_OR_PREFIX"),
    (
        raking_commutative_only_bit_xor_prefix,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_XOR_PREFIX"
    ),
    (raking_bit_xor_prefix, "COOP_REDUCE_RAKING_BIT_XOR_PREFIX"),
    (
        warp_reductions_bit_xor_prefix,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_XOR_PREFIX"
    ),
    (warp_bit_xor_prefix, "COOP_REDUCE_WARP_BIT_XOR_PREFIX"),
    (
        raking_commutative_only_sum_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_SUM_PAYLOAD"
    ),
    (raking_sum_payload, "COOP_REDUCE_RAKING_SUM_PAYLOAD"),
    (
        warp_reductions_sum_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_SUM_PAYLOAD"
    ),
    (warp_sum_payload, "COOP_REDUCE_WARP_SUM_PAYLOAD"),
    (
        raking_commutative_only_multiplies_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MULTIPLIES_PAYLOAD"
    ),
    (
        raking_multiplies_payload,
        "COOP_REDUCE_RAKING_MULTIPLIES_PAYLOAD"
    ),
    (
        warp_reductions_multiplies_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_MULTIPLIES_PAYLOAD"
    ),
    (
        warp_multiplies_payload,
        "COOP_REDUCE_WARP_MULTIPLIES_PAYLOAD"
    ),
    (
        raking_commutative_only_min_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MIN_PAYLOAD"
    ),
    (raking_min_payload, "COOP_REDUCE_RAKING_MIN_PAYLOAD"),
    (
        warp_reductions_min_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_MIN_PAYLOAD"
    ),
    (warp_min_payload, "COOP_REDUCE_WARP_MIN_PAYLOAD"),
    (
        raking_commutative_only_max_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_MAX_PAYLOAD"
    ),
    (raking_max_payload, "COOP_REDUCE_RAKING_MAX_PAYLOAD"),
    (
        warp_reductions_max_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_MAX_PAYLOAD"
    ),
    (warp_max_payload, "COOP_REDUCE_WARP_MAX_PAYLOAD"),
    (
        raking_commutative_only_bit_and_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_AND_PAYLOAD"
    ),
    (raking_bit_and_payload, "COOP_REDUCE_RAKING_BIT_AND_PAYLOAD"),
    (
        warp_reductions_bit_and_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_AND_PAYLOAD"
    ),
    (warp_bit_and_payload, "COOP_REDUCE_WARP_BIT_AND_PAYLOAD"),
    (
        raking_commutative_only_bit_or_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_OR_PAYLOAD"
    ),
    (raking_bit_or_payload, "COOP_REDUCE_RAKING_BIT_OR_PAYLOAD"),
    (
        warp_reductions_bit_or_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_OR_PAYLOAD"
    ),
    (warp_bit_or_payload, "COOP_REDUCE_WARP_BIT_OR_PAYLOAD"),
    (
        raking_commutative_only_bit_xor_payload,
        "COOP_REDUCE_RAKING_COMMUTATIVE_ONLY_BIT_XOR_PAYLOAD"
    ),
    (raking_bit_xor_payload, "COOP_REDUCE_RAKING_BIT_XOR_PAYLOAD"),
    (
        warp_reductions_bit_xor_payload,
        "COOP_REDUCE_WARP_REDUCTIONS_BIT_XOR_PAYLOAD"
    ),
    (warp_bit_xor_payload, "COOP_REDUCE_WARP_BIT_XOR_PAYLOAD"),
);

#[inline(always)]
pub(super) fn reduce<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    G: ReduceGroup<A, S>,
    Op: super::sealed::Operator,
    A: ReduceAlgorithm,
    C: Count,
    S: TempStorageLike,
    const PAYLOAD: bool,
>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: ReduceOptions<T, Op, A, C, S>,
) -> Option<T>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::<T, 1>::new();
    let (input, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input, output_pointer, input_bits, output_bits);
    let uniform = C::controls::<G>(options.valid_items);
    let storage = options.storage.into_raw();
    let _operator = options.binary_op;
    macro_rules! call {
        ($commutative:ident, $raking:ident, $warps:ident, $warp:ident) => {
            if G::WARP {
                G::call::<T, _, _, _, _, ITEMS>($warp, storage, uniform, payload);
            } else {
                match A::KIND {
                    0 | 3 => G::call::<T, _, _, _, _, ITEMS>($warps, storage, uniform, payload),
                    1 => G::call::<T, _, _, _, _, ITEMS>($commutative, storage, uniform, payload),
                    2 => G::call::<T, _, _, _, _, ITEMS>($raking, storage, uniform, payload),
                    _ => panic!(),
                }
            }
        };
    }
    // SAFETY: the const input and fresh output are distinct, aligned, live
    // lane-local payloads. Sealed selections establish the exact native ABI.
    // The recipe checks required input bits before numeric reads, preserves
    // input, clears every output bit, and writes a value then its bit only at
    // group rank zero. The pointers cannot escape the synchronous call.
    // The checked marker enforces participation, scalar-control uniformity,
    // layout and scratch reuse; no global-memory completion is fabricated.
    unsafe {
        match const {
            if PAYLOAD {
                14 + Op::KIND
            } else {
                7 * C::KIND + Op::KIND
            }
        } {
            0 => call!(
                raking_commutative_only_sum_scalar,
                raking_sum_scalar,
                warp_reductions_sum_scalar,
                warp_sum_scalar
            ),
            1 => call!(
                raking_commutative_only_multiplies_scalar,
                raking_multiplies_scalar,
                warp_reductions_multiplies_scalar,
                warp_multiplies_scalar
            ),
            2 => call!(
                raking_commutative_only_min_scalar,
                raking_min_scalar,
                warp_reductions_min_scalar,
                warp_min_scalar
            ),
            3 => call!(
                raking_commutative_only_max_scalar,
                raking_max_scalar,
                warp_reductions_max_scalar,
                warp_max_scalar
            ),
            4 => call!(
                raking_commutative_only_bit_and_scalar,
                raking_bit_and_scalar,
                warp_reductions_bit_and_scalar,
                warp_bit_and_scalar
            ),
            5 => call!(
                raking_commutative_only_bit_or_scalar,
                raking_bit_or_scalar,
                warp_reductions_bit_or_scalar,
                warp_bit_or_scalar
            ),
            6 => call!(
                raking_commutative_only_bit_xor_scalar,
                raking_bit_xor_scalar,
                warp_reductions_bit_xor_scalar,
                warp_bit_xor_scalar
            ),
            7 => call!(
                raking_commutative_only_sum_prefix,
                raking_sum_prefix,
                warp_reductions_sum_prefix,
                warp_sum_prefix
            ),
            8 => call!(
                raking_commutative_only_multiplies_prefix,
                raking_multiplies_prefix,
                warp_reductions_multiplies_prefix,
                warp_multiplies_prefix
            ),
            9 => call!(
                raking_commutative_only_min_prefix,
                raking_min_prefix,
                warp_reductions_min_prefix,
                warp_min_prefix
            ),
            10 => call!(
                raking_commutative_only_max_prefix,
                raking_max_prefix,
                warp_reductions_max_prefix,
                warp_max_prefix
            ),
            11 => call!(
                raking_commutative_only_bit_and_prefix,
                raking_bit_and_prefix,
                warp_reductions_bit_and_prefix,
                warp_bit_and_prefix
            ),
            12 => call!(
                raking_commutative_only_bit_or_prefix,
                raking_bit_or_prefix,
                warp_reductions_bit_or_prefix,
                warp_bit_or_prefix
            ),
            13 => call!(
                raking_commutative_only_bit_xor_prefix,
                raking_bit_xor_prefix,
                warp_reductions_bit_xor_prefix,
                warp_bit_xor_prefix
            ),
            14 => call!(
                raking_commutative_only_sum_payload,
                raking_sum_payload,
                warp_reductions_sum_payload,
                warp_sum_payload
            ),
            15 => call!(
                raking_commutative_only_multiplies_payload,
                raking_multiplies_payload,
                warp_reductions_multiplies_payload,
                warp_multiplies_payload
            ),
            16 => call!(
                raking_commutative_only_min_payload,
                raking_min_payload,
                warp_reductions_min_payload,
                warp_min_payload
            ),
            17 => call!(
                raking_commutative_only_max_payload,
                raking_max_payload,
                warp_reductions_max_payload,
                warp_max_payload
            ),
            18 => call!(
                raking_commutative_only_bit_and_payload,
                raking_bit_and_payload,
                warp_reductions_bit_and_payload,
                warp_bit_and_payload
            ),
            19 => call!(
                raking_commutative_only_bit_or_payload,
                raking_bit_or_payload,
                warp_reductions_bit_or_payload,
                warp_bit_or_payload
            ),
            20 => call!(
                raking_commutative_only_bit_xor_payload,
                raking_bit_xor_payload,
                warp_reductions_bit_xor_payload,
                warp_bit_xor_payload
            ),
            _ => panic!(),
        }
    }
    // get checks root-only validity before reading the numerical output.
    output.get(0)
}

fn raking_callable_payload() -> &'static str {
    library::registration!("COOP_REDUCE_RAKING_CALLABLE_PAYLOAD")
}

unsafe extern "C" fn binary_thunk<F: Fn(i32, i32) -> i32>(
    environment: *const F,
    left: i32,
    right: i32,
) -> i32 {
    // SAFETY: the synchronous checked call keeps its shared environment alive.
    unsafe { (&*environment)(left, right) }
}

#[inline(always)]
pub(super) fn callable<
    const ITEMS: usize,
    const ALIGN: usize,
    F: Fn(i32, i32) -> i32 + Copy,
    S: TempStorageLike,
>(
    input: &ThreadData<i32, ITEMS, ALIGN>,
    operator: F,
    storage: S,
) -> Option<i32>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::<i32, 1>::new();
    let (input, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input, output_pointer, input_bits, output_bits);
    let storage = storage.into_raw();
    // SAFETY: input and fresh output are distinct live payloads. The recipe
    // checks every input bit, preserves input and writes only the root output.
    // The compiler checks Block participation, scratch reuse and the genuine
    // callable and any captured value. The environment borrow lives through this call.
    unsafe {
        library::checked_block_payload_callback_call_v1::<i32, _, _, _, _, _, _, ITEMS>(
            raking_callable_payload,
            storage,
            (),
            payload,
            binary_thunk::<F>,
            &operator,
        )
    }
    output.get(0)
}
