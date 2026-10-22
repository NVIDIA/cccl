// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{RadixKey, RadixSortBit, RadixSortOptions, RadixSortOrder};
use crate::{Alignment, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

fn ascending() -> &'static str {
    library::registration!("COOP_RADIX_SORT_KEYS_ASCENDING")
}
fn descending() -> &'static str {
    library::registration!("COOP_RADIX_SORT_KEYS_DESCENDING")
}

struct Bounds<T, Begin, End>(core::marker::PhantomData<(T, Begin, End)>);
impl<T: RadixKey, Begin: RadixSortBit, End: RadixSortBit> Bounds<T, Begin, End> {
    const BEGIN: Option<i64> = if Begin::DEFAULT {
        Some(0)
    } else {
        Begin::STATIC
    };
    const END: Option<i64> = if End::DEFAULT {
        Some(T::WIDTH)
    } else {
        End::STATIC
    };
    const VALID: () = {
        if let Some(begin) = Self::BEGIN {
            assert!(
                begin >= 0 && begin < T::WIDTH,
                "Radix Sort begin must be in 0..key_width"
            );
        }
        if let Some(end) = Self::END {
            assert!(
                end > 0 && end <= T::WIDTH,
                "Radix Sort end must be in 1..=key_width"
            );
        }
        if let (Some(begin), Some(end)) = (Self::BEGIN, Self::END) {
            assert!(begin < end, "Radix Sort requires begin < end");
        }
    };
}

#[inline(always)]
pub(super) fn radix_sort_keys<T, O, Begin, End, S, const N: usize, const ALIGN: usize>(
    input: &ThreadData<T, N, ALIGN>,
    options: RadixSortOptions<T, O, Begin, End, S>,
) -> ThreadData<T, N, ALIGN>
where
    T: RadixKey,
    O: RadixSortOrder,
    Begin: RadixSortBit,
    End: RadixSortBit,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    let () = Bounds::<T, Begin, End>::VALID;
    let mut output = ThreadData::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let uniform = (options.begin.value(0), options.end.value(T::WIDTH));
    let storage = options.storage.into_raw();
    // SAFETY: input and fresh output are distinct aligned live local buffers
    // with independent N-bit validity maps. The exact recipe checks the wide
    // bounds and every input bit before numeric reads, sorts a private copy,
    // then writes every result and output bit. No pointer escapes. The genuine
    // marker checks complete Block participation, uniform controls and storage
    // origin/layout/reuse; the provider checks the 65,535-item tile capacity.
    unsafe {
        if O::DESCENDING {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                descending, storage, uniform, payload,
            );
        } else {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                ascending, storage, uniform, payload,
            );
        }
    }
    output
}
