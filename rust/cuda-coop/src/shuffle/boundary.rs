// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{ShuffleMode, ShuffleOptions};
use crate::{Alignment, Numeric, SupportedAlignment, ThreadData};
use cuda_device::library;

fn down() -> &'static str {
    library::registration!("COOP_SHUFFLE_DOWN")
}
fn up() -> &'static str {
    library::registration!("COOP_SHUFFLE_UP")
}

#[inline(always)]
pub(super) fn shuffle<T, M, const ITEMS: usize, const ALIGN: usize>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: ShuffleOptions<M>,
) -> ThreadData<T, ITEMS, ALIGN>
where
    T: Numeric,
    M: ShuffleMode,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::<T, ITEMS, ALIGN>::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let _mode = options.mode;
    // SAFETY: the input and fresh output are distinct live per-thread buffers
    // of the same numeric type, extent and alignment. The typed provider checks
    // all input bits before reading numbers, preserves input, and only copies
    // initialized CUB outputs before marking their bits. The exposed edge and
    // padding bits remain invalid. Pointers do not escape the synchronous call.
    // The genuine marker checks full-block participation and planned scratch.
    unsafe {
        match M::KIND {
            0 => library::checked_block_payload_call_v1::<T, _, _, _, _, ITEMS>(
                down,
                (),
                (),
                payload,
            ),
            1 => {
                library::checked_block_payload_call_v1::<T, _, _, _, _, ITEMS>(up, (), (), payload)
            }
            _ => panic!(),
        }
    }
    output
}
