// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{RadixKey, RadixSortBit, RadixSortOptions, RadixSortOrder, RadixValue};
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

fn pairs_ascending<V: RadixValue>() -> &'static str {
    V::ASCENDING_REGISTRATION
}
fn pairs_descending<V: RadixValue>() -> &'static str {
    V::DESCENDING_REGISTRATION
}

#[inline(always)]
pub(super) fn radix_sort_pairs<
    K,
    V,
    O,
    Begin,
    End,
    S,
    const N: usize,
    const KA: usize,
    const VA: usize,
>(
    keys: &ThreadData<K, N, KA>,
    values: &ThreadData<V, N, VA>,
    options: RadixSortOptions<K, O, Begin, End, S>,
) -> (ThreadData<K, N, KA>, ThreadData<V, N, VA>)
where
    K: RadixKey,
    V: RadixValue,
    O: RadixSortOrder,
    Begin: RadixSortBit,
    End: RadixSortBit,
    S: TempStorageLike,
    Alignment<KA>: SupportedAlignment,
    Alignment<VA>: SupportedAlignment,
{
    let () = Bounds::<K, Begin, End>::VALID;
    let mut output_keys = ThreadData::new();
    let mut output_values = ThreadData::new();
    let (keys_pointer, keys_bits) = keys.block_io_read_parts();
    let (output_keys_pointer, output_keys_bits) = output_keys.block_io_parts();
    let (values_pointer, values_bits) = values.block_io_read_parts();
    let (output_values_pointer, output_values_bits) = output_values.block_io_parts();
    let payload = (
        keys_pointer,
        output_keys_pointer,
        values_pointer,
        output_values_pointer,
        keys_bits,
        output_keys_bits,
        values_bits,
        output_values_bits,
    );
    let uniform = (options.begin.value(0), options.end.value(K::WIDTH));
    let storage = options.storage.into_raw();
    // SAFETY: both borrowed inputs and both fresh outputs are aligned live
    // local buffers, each with its own N-bit validity map. The closed recipe
    // identifies K through the marker and V through its genuine registration.
    // It validates wide bounds and BOTH input maps before numeric reads, sorts
    // private copies and initializes every output slot/map. No pointer escapes.
    // The marker proves complete Block participation, control uniformity and
    // storage policy; the provider checks the 65,535-item capacity and mixed ABI.
    unsafe {
        if O::DESCENDING {
            library::checked_block_payload_call_v1::<K, _, _, _, _, N>(
                pairs_descending::<V>,
                storage,
                uniform,
                payload,
            );
        } else {
            library::checked_block_payload_call_v1::<K, _, _, _, _, N>(
                pairs_ascending::<V>,
                storage,
                uniform,
                payload,
            );
        }
    }
    (output_keys, output_values)
}
