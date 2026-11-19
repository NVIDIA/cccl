// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{HistogramAlgorithm, HistogramCounter, HistogramOptions, HistogramSample};
use crate::{Alignment, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

const TOKEN: &[u8] = b"\"histogram_xxx_bins0000000000_k0000000000_validity_bits_v1\"";

const fn decimal(bytes: &mut [u8], offset: usize, mut value: usize) {
    let mut remaining = 10;
    while remaining > 0 {
        remaining -= 1;
        bytes[offset + remaining] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    assert!(value == 0, "Histogram decimal field overflow");
}

// Preserve registration!'s tracked JSON, changing only the closed operation
// token. Each algorithm's concrete template length is independent of BINS/K.
const fn encode<const LEN: usize>(base: &str, tag: [u8; 3], bins: usize, k: usize) -> [u8; LEN] {
    assert!(
        bins > 0 && bins <= i32::MAX as usize,
        "Histogram requires positive bins <= i32::MAX"
    );
    assert!(
        k > 0 && k <= i32::MAX as usize,
        "Histogram requires positive output extent <= i32::MAX"
    );
    let source = base.as_bytes();
    assert!(source.len() == LEN, "Histogram template length changed");
    let mut bytes = [0_u8; LEN];
    let mut i = 0;
    while i < LEN {
        bytes[i] = source[i];
        i += 1;
    }
    let mut found = 0;
    let mut position = 0;
    let mut start = 0;
    while start + TOKEN.len() <= LEN {
        let mut j = 0;
        while j < TOKEN.len() && source[start + j] == TOKEN[j] {
            j += 1;
        }
        if j == TOKEN.len() {
            found += 1;
            position = start;
        }
        start += 1;
    }
    assert!(
        found == 1,
        "expected exactly one Histogram operation template"
    );
    let counter = position + b"\"histogram_".len();
    let bins_offset = counter + 3 + b"_bins".len();
    let k_offset = bins_offset + 10 + b"_k".len();
    i = 0;
    while i < 3 {
        bytes[counter + i] = tag[i];
        i += 1;
    }
    decimal(&mut bytes, bins_offset, bins);
    decimal(&mut bytes, k_offset, k);
    bytes
}

macro_rules! registration {
    ($module:ident, $env:literal) => {
        mod $module {
            use super::*;
            const BASE: &str = library::registration!($env);
            const LEN: usize = BASE.len();
            struct Text<C, const BINS: usize, const K: usize>(core::marker::PhantomData<C>);
            impl<C: HistogramCounter, const BINS: usize, const K: usize> Text<C, BINS, K> {
                const BYTES: [u8; LEN] = encode(BASE, C::TAG, BINS, K);
                const REGISTRATION: &'static str = match core::str::from_utf8(&Self::BYTES) {
                    Ok(text) => text,
                    Err(_) => panic!("invalid Histogram registration UTF-8"),
                };
            }
            pub(super) fn registration<C: HistogramCounter, const BINS: usize, const K: usize>()
            -> &'static str {
                Text::<C, BINS, K>::REGISTRATION
            }
        }
    };
}
registration!(atomic, "COOP_HISTOGRAM_ATOMIC");
registration!(sort, "COOP_HISTOGRAM_SORT");

#[inline(always)]
pub(super) fn histogram<
    T,
    C,
    A,
    S,
    const BINS: usize,
    const N: usize,
    const K: usize,
    const ALIGN: usize,
>(
    input: &ThreadData<T, N, ALIGN>,
    options: HistogramOptions<BINS, C, K, A, S>,
) -> ThreadData<C, K>
where
    T: HistogramSample,
    C: HistogramCounter,
    A: HistogramAlgorithm,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    let mut output = ThreadData::<C, K>::new();
    let (input_pointer, input_bits) = input.block_io_read_parts();
    let (output_pointer, output_bits) = output.block_io_parts();
    let payload = (input_pointer, output_pointer, input_bits, output_bits);
    let storage = options.storage.into_raw();
    // SAFETY: input and fresh output are distinct live naturally aligned local
    // buffers with separate N-bit and K-bit validity maps. The exact recipe
    // validates all input bits and sample bounds before CUB, copies input,
    // initializes all C/K counters (including padding), then marks output bits.
    // No pointer escapes the synchronous call. Sealed static selections encode
    // the exact output type/extent. The genuine marker checks full Block
    // participation and scratch origin, queried layout and reuse policy.
    unsafe {
        if A::SORT {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                sort::registration::<C, BINS, K>,
                storage,
                (),
                payload,
            );
        } else {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                atomic::registration::<C, BINS, K>,
                storage,
                (),
                payload,
            );
        }
    }
    output
}
