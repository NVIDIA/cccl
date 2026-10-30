// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{DiscontinuityOptions, sealed};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (heads, "COOP_DISCONTINUITY_HEADS"),
    (heads_predecessor, "COOP_DISCONTINUITY_HEADS_PREDECESSOR"),
    (tails, "COOP_DISCONTINUITY_TAILS"),
    (tails_successor, "COOP_DISCONTINUITY_TAILS_SUCCESSOR"),
    (heads_and_tails, "COOP_DISCONTINUITY_HEADS_AND_TAILS"),
    (
        heads_and_tails_predecessor,
        "COOP_DISCONTINUITY_HEADS_AND_TAILS_PREDECESSOR"
    ),
    (
        heads_and_tails_successor,
        "COOP_DISCONTINUITY_HEADS_AND_TAILS_SUCCESSOR"
    ),
    (
        heads_and_tails_both,
        "COOP_DISCONTINUITY_HEADS_AND_TAILS_BOTH"
    ),
);

#[inline(always)]
pub(super) fn single<T, M, Pred, Succ, S, const N: usize, const ALIGN: usize>(
    input: &ThreadData<T, N, ALIGN>,
    options: DiscontinuityOptions<T, M, Pred, Succ, S>,
) -> ThreadData<i32, N, ALIGN>
where
    T: Numeric,
    M: sealed::Mode<T, Pred, Succ>,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let uniform = M::controls(options.predecessor, options.successor);
    let storage = options.storage.into_raw();
    macro_rules! call {
        ($registration:ident) => {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                $registration,
                storage,
                uniform,
                payload,
            )
        };
    }
    // SAFETY: input and fresh outputs are distinct live aligned lane-local
    // buffers. The exact recipe checks every input bit before numeric reads,
    // preserves input and initializes all i32 flags before marking their bits.
    // Each output has its own validity buffer. Pointers cannot escape the
    // synchronous call. Sealed selections give the exact typed control ABI;
    // the marker checks uniformity, complete Block participation and genuine
    // storage origin, layout and reuse under the existing policy.
    unsafe {
        match M::KIND {
            0 => call!(heads),
            1 => call!(heads_predecessor),
            2 => call!(tails),
            3 => call!(tails_successor),
            _ => panic!(),
        }
    }
    output
}

#[inline(always)]
pub(super) fn pair<T, M, Pred, Succ, S, const N: usize, const ALIGN: usize>(
    input: &ThreadData<T, N, ALIGN>,
    options: DiscontinuityOptions<T, M, Pred, Succ, S>,
) -> (ThreadData<i32, N, ALIGN>, ThreadData<i32, N, ALIGN>)
where
    T: Numeric,
    M: sealed::Mode<T, Pred, Succ>,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let mut tails = ThreadData::new();
    let (tail_pointer, tail_bits) = tails.block_io_parts();
    let payload = (
        input_pointer,
        output_pointer,
        tail_pointer,
        input_bits,
        output_bits,
        tail_bits,
    );
    let uniform = M::controls(options.predecessor, options.successor);
    let storage = options.storage.into_raw();
    macro_rules! call {
        ($registration:ident) => {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                $registration,
                storage,
                uniform,
                payload,
            )
        };
    }
    // SAFETY: input and fresh outputs are distinct live aligned lane-local
    // buffers. The exact recipe checks every input bit before numeric reads,
    // preserves input and initializes all i32 flags before marking their bits.
    // Each output has its own validity buffer. Pointers cannot escape the
    // synchronous call. Sealed selections give the exact typed control ABI;
    // the marker checks uniformity, complete Block participation and genuine
    // storage origin, layout and reuse under the existing policy.
    unsafe {
        match M::KIND {
            4 => call!(heads_and_tails),
            5 => call!(heads_and_tails_predecessor),
            6 => call!(heads_and_tails_successor),
            7 => call!(heads_and_tails_both),
            _ => panic!(),
        }
    }
    (output, tails)
}
