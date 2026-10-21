// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{ExchangeGroup, ExchangeMode, ExchangeOptions};
use crate::{Alignment, Numeric, SupportedAlignment, ThreadData};
use cuda_device::library;

macro_rules! registrations {
    ($(($name:ident, $variable:literal)),+ $(,)?) => {$ (
        fn $name() -> &'static str { library::registration!($variable) }
    )+};
}
registrations!(
    (
        block_striped_to_blocked,
        "COOP_EXCHANGE_BLOCK_STRIPED_TO_BLOCKED"
    ),
    (
        block_blocked_to_striped,
        "COOP_EXCHANGE_BLOCK_BLOCKED_TO_STRIPED"
    ),
    (
        warp_striped_to_blocked,
        "COOP_EXCHANGE_WARP_STRIPED_TO_BLOCKED"
    ),
    (
        warp_blocked_to_striped,
        "COOP_EXCHANGE_WARP_BLOCKED_TO_STRIPED"
    ),
);

struct Selection<G, M>(core::marker::PhantomData<(G, M)>);
impl<G: ExchangeGroup, M: ExchangeMode> Selection<G, M> {
    const KIND: u8 = 2 * G::WARP as u8 + M::KIND;
}

#[inline(always)]
pub(super) fn exchange<T, G, M, const ITEMS: usize, const ALIGN: usize>(
    input: &ThreadData<T, ITEMS, ALIGN>,
    options: ExchangeOptions<M>,
) -> ThreadData<T, ITEMS, ALIGN>
where
    T: Numeric,
    G: ExchangeGroup,
    M: ExchangeMode,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::<T, ITEMS, ALIGN>::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let _mode = options.mode;
    // SAFETY: input and fresh output are distinct live per-thread buffers with
    // the same numeric type and extent and their actual alignment. The typed
    // provider preflights every input bit before numeric reads, preserves input,
    // and initializes each output value before marking its validity bit. No
    // pointer escapes the synchronous call. Genuine markers check participation
    // and compiler-planned implicit scratch allocation and reuse.
    unsafe {
        match Selection::<G, M>::KIND {
            0 => G::call::<T, _, _, ITEMS>(block_striped_to_blocked, payload),
            1 => G::call::<T, _, _, ITEMS>(block_blocked_to_striped, payload),
            2 => G::call::<T, _, _, ITEMS>(warp_striped_to_blocked, payload),
            3 => G::call::<T, _, _, ITEMS>(warp_blocked_to_striped, payload),
            _ => panic!(),
        }
    }
    output
}
