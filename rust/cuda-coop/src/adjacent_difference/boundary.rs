// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{AdjacentDifferenceDirection, AdjacentDifferenceOptions};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (left_full, "COOP_ADJACENT_DIFFERENCE_LEFT_FULL"),
    (
        left_full_predecessor,
        "COOP_ADJACENT_DIFFERENCE_LEFT_FULL_PREDECESSOR"
    ),
    (left_partial, "COOP_ADJACENT_DIFFERENCE_LEFT_PARTIAL"),
    (
        left_partial_predecessor,
        "COOP_ADJACENT_DIFFERENCE_LEFT_PARTIAL_PREDECESSOR"
    ),
    (right_full, "COOP_ADJACENT_DIFFERENCE_RIGHT_FULL"),
    (
        right_full_successor,
        "COOP_ADJACENT_DIFFERENCE_RIGHT_FULL_SUCCESSOR"
    ),
    (right_partial, "COOP_ADJACENT_DIFFERENCE_RIGHT_PARTIAL"),
);

#[inline(always)]
pub(super) fn count<const ITEMS: usize>(count: usize) -> i32 {
    let threads = crate::this_block().count() as usize;
    // Validate capacity without multiplication overflow, then narrow the count.
    assert!(ITEMS <= i32::MAX as usize / threads);
    assert!(count <= threads * ITEMS);
    count as i32
}

#[inline(always)]
pub(super) fn adjacent_difference<
    T,
    D,
    Count,
    Pred,
    Succ,
    S,
    const ITEMS: usize,
    const ALIGN: usize,
>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: AdjacentDifferenceOptions<T, D, Count, Pred, Succ, S>,
) -> ThreadData<T, ITEMS, ALIGN>
where
    T: Numeric,
    D: AdjacentDifferenceDirection<T, Count, Pred, Succ>,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let uniform = D::controls::<ITEMS>(options.count, options.predecessor, options.successor);
    let storage = options.storage.into_raw();
    macro_rules! call {
        ($registration:ident) => {
            library::checked_block_payload_call_v1::<T, _, _, _, _, ITEMS>(
                $registration,
                storage,
                uniform,
                payload,
            )
        };
    }
    // SAFETY: input and fresh output are distinct live aligned lane-local
    // numeric buffers. The exact recipe validates every input bit, including
    // copied suffix items at count zero, before numeric reads. It preserves
    // input, initializes all output slots and then marks their bits. Pointers
    // cannot escape the synchronous call. Sealed selections give the exact
    // scalar ABI; the marker checks uniform controls, full participation and
    // genuine storage origin/layout/reuse under the existing policy.
    unsafe {
        match D::KIND {
            0 => call!(left_full),
            1 => call!(left_full_predecessor),
            2 => call!(left_partial),
            3 => call!(left_partial_predecessor),
            4 => call!(right_full),
            5 => call!(right_full_successor),
            6 => call!(right_partial),
            _ => panic!(),
        }
    }
    output
}
