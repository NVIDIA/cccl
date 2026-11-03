// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{RunLength, RunLengthDecodeIntoOptions, RunLengthDecodeOffset, RunLengthDecodeOptions};
use crate::{
    Alignment, CollectiveTiles, DisjointSlice, Numeric, SupportedAlignment, TempStorageLike,
    ThreadData,
};
use cuda_device::{library, thread};

struct Extent<const N: usize>;
impl<const N: usize> Extent<N> {
    const VALID: () = assert!(
        N > 0 && N <= i32::MAX as usize,
        "Run Length Decode requires positive run extent <= i32::MAX"
    );
}
struct DecodedExtent<const D: usize>;
impl<const D: usize> DecodedExtent<D> {
    const VALID: () = assert!(
        D > 0 && D <= i32::MAX as usize,
        "Run Length Decode requires positive decoded extent <= i32::MAX"
    );
}
struct Capacity<const C: usize>;
impl<const C: usize> Capacity<C> {
    const VALID: () = assert!(
        C > 0,
        "Run Length Decode requires positive destination ownership capacity"
    );
}

const fn encode<const LEN: usize>(
    base: &str,
    token: &[u8],
    length: [u8; 3],
    decoded: usize,
    control: [u8; 3],
) -> [u8; LEN] {
    assert!(
        decoded > 0 && decoded <= i32::MAX as usize,
        "Run Length Decode requires positive decoded extent <= i32::MAX"
    );
    assert!(base.len() == LEN);
    let source = base.as_bytes();
    let mut bytes = [0; LEN];
    let mut i = 0;
    while i < LEN {
        bytes[i] = source[i];
        i += 1;
    }
    let mut found = 0;
    let mut position = 0;
    let mut start = 0;
    while start + token.len() <= LEN {
        let mut n = 0;
        while n < token.len() && source[start + n] == token[n] {
            n += 1;
        }
        if n == token.len() {
            found += 1;
            position = start;
        }
        start += 1;
    }
    assert!(
        found == 1,
        "expected exactly one Run Length Decode operation template"
    );
    // Fixed suffix: lTTT_dDDDDDDDDDD_oTTT_validity_bits_v1 followed by quote.
    let suffix = position + token.len() - b"lTTT_dDDDDDDDDDD_oTTT_validity_bits_v1\"".len();
    i = 0;
    while i < 3 {
        bytes[suffix + 1 + i] = length[i];
        bytes[suffix + 18 + i] = control[i];
        i += 1;
    }
    let mut value = decoded;
    i = 10;
    while i > 0 {
        i -= 1;
        bytes[suffix + 6 + i] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    assert!(value == 0);
    bytes
}

macro_rules! registration {
    ($module:ident, $env:literal, $token:literal) => {
        mod $module {
            use super::*;
            const BASE: &str = library::registration!($env);
            const LEN: usize = BASE.len();
            struct Text<L, O, const D: usize>(core::marker::PhantomData<(L, O)>);
            impl<L: RunLength, O: RunLengthDecodeOffset, const D: usize> Text<L, O, D> {
                const BYTES: [u8; LEN] = encode(BASE, $token, L::TAG, D, O::TAG);
                const REGISTRATION: &'static str = match core::str::from_utf8(&Self::BYTES) {
                    Ok(s) => s,
                    Err(_) => panic!("invalid Run Length Decode registration UTF-8"),
                };
            }
            pub(super) fn registration<L: RunLength, O: RunLengthDecodeOffset, const D: usize>()
            -> &'static str {
                Text::<L, O, D>::REGISTRATION
            }
        }
    };
}
registration!(
    window,
    "COOP_RUN_LENGTH_DECODE_WINDOW",
    b"\"run_length_decode_window_lu32_d0000000001_ou64_validity_bits_v1\""
);
registration!(
    into,
    "COOP_RUN_LENGTH_DECODE_INTO",
    b"\"run_length_decode_into_lu32_d0000000001_ou64_validity_bits_v1\""
);

#[inline(always)]
pub(super) fn window<
    T: Numeric,
    L: RunLength,
    O: RunLengthDecodeOffset,
    S: TempStorageLike,
    const N: usize,
    const D: usize,
    const VA: usize,
    const LA: usize,
>(
    values: &ThreadData<T, N, VA>,
    lengths: &ThreadData<L, N, LA>,
    options: RunLengthDecodeOptions<D, O, S>,
) -> ThreadData<T, D, VA>
where
    Alignment<VA>: SupportedAlignment,
    Alignment<LA>: SupportedAlignment,
{
    let () = Extent::<N>::VALID;
    let () = DecodedExtent::<D>::VALID;
    let mut output = ThreadData::<T, D, VA>::new();
    let (values, value_bits) = values.block_io_read_parts();
    let (lengths, length_bits) = lengths.block_io_read_parts();
    let (result, result_bits) = output.block_io_parts();
    let storage = options.storage.into_raw();
    // SAFETY: both fully checked input planes are preserved and disjoint from
    // the fresh D/T/VA output and its map. The native driver validates all N
    // input bits, wide controls and run totals before CUB, zeros every tail and
    // marks all D results. Its queried scratch includes the validation scan;
    // the genuine marker checks full Block participation and scratch reuse.
    unsafe {
        library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
            window::registration::<L, O, D>,
            storage,
            (options.offset.value(),),
            (
                values,
                result,
                lengths,
                value_bits,
                length_bits,
                result_bits,
            ),
        );
    }
    output
}

#[inline(always)]
pub(super) fn into<
    T: Numeric,
    L: RunLength,
    O: RunLengthDecodeOffset,
    S: TempStorageLike,
    const N: usize,
    const D: usize,
    const VA: usize,
    const LA: usize,
    const CAPACITY: usize,
>(
    values: &ThreadData<T, N, VA>,
    lengths: &ThreadData<L, N, LA>,
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    options: RunLengthDecodeIntoOptions<D, O, S>,
) -> u32
where
    Alignment<VA>: SupportedAlignment,
    Alignment<LA>: SupportedAlignment,
{
    let () = Extent::<N>::VALID;
    let () = DecodedExtent::<D>::VALID;
    let () = Capacity::<CAPACITY>::VALID;
    let threads = thread::blockDim_x() as usize
        * thread::blockDim_y() as usize
        * thread::blockDim_z() as usize;
    assert!(threads <= usize::MAX / CAPACITY);
    let region_capacity = threads * CAPACITY;
    let blocks =
        thread::gridDim_x() as usize * thread::gridDim_y() as usize * thread::gridDim_z() as usize;
    assert!(blocks > 0 && blocks <= usize::MAX / region_capacity);
    let period = blocks * region_capacity;
    let rank = thread::blockIdx_x() as usize
        + thread::gridDim_x() as usize
            * (thread::blockIdx_y() as usize
                + thread::gridDim_y() as usize * thread::blockIdx_z() as usize);
    let region_begin = rank * region_capacity;
    let capacity = destination.len();
    assert!(capacity <= isize::MAX as usize / core::mem::size_of::<T>());
    let pointer = destination.as_mut_ptr();
    let (values, value_bits) = values.block_io_read_parts();
    let (lengths, length_bits) = lengths.block_io_read_parts();
    let mut total = 0_u32;
    let storage = options.storage.into_raw();
    // SAFETY: the genuine exclusive carrier owns the destination. No offset
    // pointer is formed here. The native driver validates the complete decoded
    // interval against allocation and cyclic Block ownership before any write,
    // and preserves both fully initialized input planes. The local total is
    // initialized before return; numerical equality supplies no compiler proof
    // of uniformity. The WRITE marker retains destination epochs and scratch
    // completion through the entire synchronous decode loop.
    unsafe {
        library::checked_block_storage_call_v1::<T, _, _, _, _, _, N, true>(
            into::registration::<L, O, D>,
            destination,
            storage,
            (
                pointer,
                options.offset.value(),
                capacity as u64,
                period as u64,
                region_begin as u64,
                region_capacity as u64,
            ),
            (
                values,
                lengths,
                value_bits,
                length_bits,
                &mut total as *mut u32,
            ),
        );
    }
    total
}
