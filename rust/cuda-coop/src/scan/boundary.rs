// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{ScanAlgorithm, ScanGroup, ScanMode, ScanOperator, ScanOptions};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

// Each static recipe keeps a concrete function-item registration identity.
macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (raking_exclusive_zero, "COOP_SCAN_RAKING_EXCLUSIVE_ZERO"),
    (
        raking_memoize_exclusive_zero,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_ZERO"
    ),
    (
        warp_scans_exclusive_zero,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_ZERO"
    ),
    (warp_exclusive_zero, "COOP_SCAN_WARP_EXCLUSIVE_ZERO"),
    (raking_exclusive_sum, "COOP_SCAN_RAKING_EXCLUSIVE_SUM"),
    (
        raking_memoize_exclusive_sum,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_SUM"
    ),
    (
        warp_scans_exclusive_sum,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_SUM"
    ),
    (warp_exclusive_sum, "COOP_SCAN_WARP_EXCLUSIVE_SUM"),
    (
        raking_exclusive_multiplies,
        "COOP_SCAN_RAKING_EXCLUSIVE_MULTIPLIES"
    ),
    (
        raking_memoize_exclusive_multiplies,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_MULTIPLIES"
    ),
    (
        warp_scans_exclusive_multiplies,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_MULTIPLIES"
    ),
    (
        warp_exclusive_multiplies,
        "COOP_SCAN_WARP_EXCLUSIVE_MULTIPLIES"
    ),
    (raking_exclusive_min, "COOP_SCAN_RAKING_EXCLUSIVE_MIN"),
    (
        raking_memoize_exclusive_min,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_MIN"
    ),
    (
        warp_scans_exclusive_min,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_MIN"
    ),
    (warp_exclusive_min, "COOP_SCAN_WARP_EXCLUSIVE_MIN"),
    (raking_exclusive_max, "COOP_SCAN_RAKING_EXCLUSIVE_MAX"),
    (
        raking_memoize_exclusive_max,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_MAX"
    ),
    (
        warp_scans_exclusive_max,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_MAX"
    ),
    (warp_exclusive_max, "COOP_SCAN_WARP_EXCLUSIVE_MAX"),
    (
        raking_exclusive_bit_and,
        "COOP_SCAN_RAKING_EXCLUSIVE_BIT_AND"
    ),
    (
        raking_memoize_exclusive_bit_and,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_BIT_AND"
    ),
    (
        warp_scans_exclusive_bit_and,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_BIT_AND"
    ),
    (warp_exclusive_bit_and, "COOP_SCAN_WARP_EXCLUSIVE_BIT_AND"),
    (raking_exclusive_bit_or, "COOP_SCAN_RAKING_EXCLUSIVE_BIT_OR"),
    (
        raking_memoize_exclusive_bit_or,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_BIT_OR"
    ),
    (
        warp_scans_exclusive_bit_or,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_BIT_OR"
    ),
    (warp_exclusive_bit_or, "COOP_SCAN_WARP_EXCLUSIVE_BIT_OR"),
    (
        raking_exclusive_bit_xor,
        "COOP_SCAN_RAKING_EXCLUSIVE_BIT_XOR"
    ),
    (
        raking_memoize_exclusive_bit_xor,
        "COOP_SCAN_RAKING_MEMOIZE_EXCLUSIVE_BIT_XOR"
    ),
    (
        warp_scans_exclusive_bit_xor,
        "COOP_SCAN_WARP_SCANS_EXCLUSIVE_BIT_XOR"
    ),
    (warp_exclusive_bit_xor, "COOP_SCAN_WARP_EXCLUSIVE_BIT_XOR"),
    (raking_inclusive_sum, "COOP_SCAN_RAKING_INCLUSIVE_SUM"),
    (
        raking_memoize_inclusive_sum,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_SUM"
    ),
    (
        warp_scans_inclusive_sum,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_SUM"
    ),
    (warp_inclusive_sum, "COOP_SCAN_WARP_INCLUSIVE_SUM"),
    (
        raking_inclusive_multiplies,
        "COOP_SCAN_RAKING_INCLUSIVE_MULTIPLIES"
    ),
    (
        raking_memoize_inclusive_multiplies,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_MULTIPLIES"
    ),
    (
        warp_scans_inclusive_multiplies,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_MULTIPLIES"
    ),
    (
        warp_inclusive_multiplies,
        "COOP_SCAN_WARP_INCLUSIVE_MULTIPLIES"
    ),
    (raking_inclusive_min, "COOP_SCAN_RAKING_INCLUSIVE_MIN"),
    (
        raking_memoize_inclusive_min,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_MIN"
    ),
    (
        warp_scans_inclusive_min,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_MIN"
    ),
    (warp_inclusive_min, "COOP_SCAN_WARP_INCLUSIVE_MIN"),
    (raking_inclusive_max, "COOP_SCAN_RAKING_INCLUSIVE_MAX"),
    (
        raking_memoize_inclusive_max,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_MAX"
    ),
    (
        warp_scans_inclusive_max,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_MAX"
    ),
    (warp_inclusive_max, "COOP_SCAN_WARP_INCLUSIVE_MAX"),
    (
        raking_inclusive_bit_and,
        "COOP_SCAN_RAKING_INCLUSIVE_BIT_AND"
    ),
    (
        raking_memoize_inclusive_bit_and,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_BIT_AND"
    ),
    (
        warp_scans_inclusive_bit_and,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_BIT_AND"
    ),
    (warp_inclusive_bit_and, "COOP_SCAN_WARP_INCLUSIVE_BIT_AND"),
    (raking_inclusive_bit_or, "COOP_SCAN_RAKING_INCLUSIVE_BIT_OR"),
    (
        raking_memoize_inclusive_bit_or,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_BIT_OR"
    ),
    (
        warp_scans_inclusive_bit_or,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_BIT_OR"
    ),
    (warp_inclusive_bit_or, "COOP_SCAN_WARP_INCLUSIVE_BIT_OR"),
    (
        raking_inclusive_bit_xor,
        "COOP_SCAN_RAKING_INCLUSIVE_BIT_XOR"
    ),
    (
        raking_memoize_inclusive_bit_xor,
        "COOP_SCAN_RAKING_MEMOIZE_INCLUSIVE_BIT_XOR"
    ),
    (
        warp_scans_inclusive_bit_xor,
        "COOP_SCAN_WARP_SCANS_INCLUSIVE_BIT_XOR"
    ),
    (warp_inclusive_bit_xor, "COOP_SCAN_WARP_INCLUSIVE_BIT_XOR"),
);

#[inline(always)]
pub(super) fn scan<
    T: Numeric,
    const ITEMS: usize,
    const ALIGN: usize,
    G: ScanGroup<A, S>,
    M: ScanMode<T, Op, Seed>,
    Op: ScanOperator<T>,
    A: ScanAlgorithm,
    Seed,
    S: TempStorageLike,
>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: ScanOptions<T, M, Op, A, Seed, S>,
) -> ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (input, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input, output_pointer, input_bits, output_bits);
    let uniform = M::controls(options.initial_value);
    let storage = options.storage.into_raw();
    let _operator = options.scan_op;
    macro_rules! call {
        ($raking:ident, $memoize:ident, $warps:ident, $warp:ident) => {
            if G::WARP {
                G::call::<T, _, _, _, _, ITEMS>($warp, storage, uniform, payload);
            } else {
                match A::KIND {
                    0 | 1 => G::call::<T, _, _, _, _, ITEMS>($raking, storage, uniform, payload),
                    2 => G::call::<T, _, _, _, _, ITEMS>($memoize, storage, uniform, payload),
                    3 => G::call::<T, _, _, _, _, ITEMS>($warps, storage, uniform, payload),
                    _ => panic!(),
                }
            }
        };
    }
    // SAFETY: input and fresh output are distinct live aligned lane-local
    // payloads. The exact native recipe validates every input bit before
    // numeric reads, preserves input, and initializes all outputs before
    // setting their bits. The pointers cannot escape the synchronous call.
    // Sealed static selections establish ABI/mode/operator compatibility;
    // the compiler enforces participation, seed uniformity, exact layout and
    // scratch policy through the payload-only checked marker. No fabricated
    // global allocation witness or memory completion is introduced.
    unsafe {
        match M::KIND {
            0 => call!(
                raking_exclusive_zero,
                raking_memoize_exclusive_zero,
                warp_scans_exclusive_zero,
                warp_exclusive_zero
            ),
            1 => call!(
                raking_exclusive_sum,
                raking_memoize_exclusive_sum,
                warp_scans_exclusive_sum,
                warp_exclusive_sum
            ),
            2 => call!(
                raking_exclusive_multiplies,
                raking_memoize_exclusive_multiplies,
                warp_scans_exclusive_multiplies,
                warp_exclusive_multiplies
            ),
            3 => call!(
                raking_exclusive_min,
                raking_memoize_exclusive_min,
                warp_scans_exclusive_min,
                warp_exclusive_min
            ),
            4 => call!(
                raking_exclusive_max,
                raking_memoize_exclusive_max,
                warp_scans_exclusive_max,
                warp_exclusive_max
            ),
            5 => call!(
                raking_exclusive_bit_and,
                raking_memoize_exclusive_bit_and,
                warp_scans_exclusive_bit_and,
                warp_exclusive_bit_and
            ),
            6 => call!(
                raking_exclusive_bit_or,
                raking_memoize_exclusive_bit_or,
                warp_scans_exclusive_bit_or,
                warp_exclusive_bit_or
            ),
            7 => call!(
                raking_exclusive_bit_xor,
                raking_memoize_exclusive_bit_xor,
                warp_scans_exclusive_bit_xor,
                warp_exclusive_bit_xor
            ),
            8 => call!(
                raking_inclusive_sum,
                raking_memoize_inclusive_sum,
                warp_scans_inclusive_sum,
                warp_inclusive_sum
            ),
            9 => call!(
                raking_inclusive_multiplies,
                raking_memoize_inclusive_multiplies,
                warp_scans_inclusive_multiplies,
                warp_inclusive_multiplies
            ),
            10 => call!(
                raking_inclusive_min,
                raking_memoize_inclusive_min,
                warp_scans_inclusive_min,
                warp_inclusive_min
            ),
            11 => call!(
                raking_inclusive_max,
                raking_memoize_inclusive_max,
                warp_scans_inclusive_max,
                warp_inclusive_max
            ),
            12 => call!(
                raking_inclusive_bit_and,
                raking_memoize_inclusive_bit_and,
                warp_scans_inclusive_bit_and,
                warp_inclusive_bit_and
            ),
            13 => call!(
                raking_inclusive_bit_or,
                raking_memoize_inclusive_bit_or,
                warp_scans_inclusive_bit_or,
                warp_inclusive_bit_or
            ),
            14 => call!(
                raking_inclusive_bit_xor,
                raking_memoize_inclusive_bit_xor,
                warp_scans_inclusive_bit_xor,
                warp_inclusive_bit_xor
            ),
            _ => panic!(),
        }
    }
    output
}
