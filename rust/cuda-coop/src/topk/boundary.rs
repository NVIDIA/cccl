// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::{TopKOptions, TopKSelectionCount, TopKValidItems, TopKValue};
use crate::{Alignment, Numeric, SupportedAlignment, TempStorageLike, ThreadData};
use cuda_device::library;

const fn decimal(bytes: &mut [u8], offset: usize, mut value: i64) {
    assert!(
        value >= 0 && value <= i32::MAX as i64,
        "TopK static count exceeds int capacity"
    );
    let mut n = 10;
    while n > 0 {
        n -= 1;
        bytes[offset + n] = b'0' + (value % 10) as u8;
        value /= 10;
    }
}
// As in Histogram, preserve registration!'s tracked JSON and replace only its
// closed operation token. Runtime fields remain canonical zero decimal text.
const fn encode<const LEN: usize>(
    base: &str,
    token: &str,
    tag: [u8; 3],
    km: u8,
    k: i64,
    vm: u8,
    v: i64,
) -> [u8; LEN] {
    let source = base.as_bytes();
    let needle = token.as_bytes();
    assert!(source.len() == LEN);
    let mut bytes = [0; LEN];
    let mut i = 0;
    while i < LEN {
        bytes[i] = source[i];
        i += 1;
    }
    let mut found = 0;
    let mut pos = 0;
    i = 0;
    while i + needle.len() <= LEN {
        let mut j = 0;
        while j < needle.len() && source[i + j] == needle[j] {
            j += 1;
        }
        if j == needle.len() {
            found += 1;
            pos = i;
        }
        i += 1;
    }
    assert!(found == 1, "expected exactly one TopK operation template");
    // The suffix has fixed widths: _k + mode/10 digits + _v + mode/10 digits.
    let counts = pos + needle.len() - b"_kr0000000000_vr0000000000_validity_bits_v1\"".len();
    bytes[counts + 2] = km;
    decimal(&mut bytes, counts + 3, k);
    // Keep the selected template's control ABI: full retains f/zero,
    // while partial records the runtime or static validity selection.
    if source[counts + 15] == b'r' {
        bytes[counts + 15] = if vm == b'f' { b'r' } else { vm };
        decimal(&mut bytes, counts + 16, v);
    }
    // Keys have no value tag. Pairs carry exactly the three bytes before _k.
    if needle[10] == b'p' {
        i = 0;
        while i < 3 {
            bytes[counts - 3 + i] = tag[i];
            i += 1;
        }
    }
    bytes
}

macro_rules! registration {
    ($module:ident,$env:literal,$token:literal) => {
        mod $module {
            use super::*;
            const BASE: &str = library::registration!($env);
            const LEN: usize = BASE.len();
            pub(super) struct Text<T, C, V>(core::marker::PhantomData<(T, C, V)>);
            impl<T: TopKValue, C: TopKSelectionCount, V: TopKValidItems> Text<T, C, V> {
                const BYTES: [u8; LEN] =
                    encode(BASE, $token, T::TAG, C::MODE, C::STATIC, V::MODE, V::STATIC);
                pub(super) const REGISTRATION: &'static str =
                    match core::str::from_utf8(&Self::BYTES) {
                        Ok(s) => s,
                        Err(_) => panic!("invalid TopK registration UTF-8"),
                    };
            }
        }
    };
}
registration!(
    min_keys_full,
    "COOP_TOPK_MIN_KEYS_FULL",
    "\"topk_min_keys_kr0000000000_vf0000000000_validity_bits_v1\""
);
registration!(
    min_keys_partial,
    "COOP_TOPK_MIN_KEYS_PARTIAL",
    "\"topk_min_keys_kr0000000000_vr0000000000_validity_bits_v1\""
);
registration!(
    min_pairs_full,
    "COOP_TOPK_MIN_PAIRS_FULL",
    "\"topk_min_pairs_i32_kr0000000000_vf0000000000_validity_bits_v1\""
);
registration!(
    min_pairs_partial,
    "COOP_TOPK_MIN_PAIRS_PARTIAL",
    "\"topk_min_pairs_i32_kr0000000000_vr0000000000_validity_bits_v1\""
);
registration!(
    max_keys_full,
    "COOP_TOPK_MAX_KEYS_FULL",
    "\"topk_max_keys_kr0000000000_vf0000000000_validity_bits_v1\""
);
registration!(
    max_keys_partial,
    "COOP_TOPK_MAX_KEYS_PARTIAL",
    "\"topk_max_keys_kr0000000000_vr0000000000_validity_bits_v1\""
);
registration!(
    max_pairs_full,
    "COOP_TOPK_MAX_PAIRS_FULL",
    "\"topk_max_pairs_i32_kr0000000000_vf0000000000_validity_bits_v1\""
);
registration!(
    max_pairs_partial,
    "COOP_TOPK_MAX_PAIRS_PARTIAL",
    "\"topk_max_pairs_i32_kr0000000000_vr0000000000_validity_bits_v1\""
);

// Select both the registration and controls before collecting native calls.
// Each public operation therefore contributes exactly one scratch use.
struct Registration<T, C, V, const MAX: bool>(core::marker::PhantomData<(T, C, V)>);
impl<T: TopKValue, C: TopKSelectionCount, V: TopKValidItems, const MAX: bool>
    Registration<T, C, V, MAX>
{
    const KEYS: &'static str = match (MAX, V::MODE == b'f') {
        (false, true) => min_keys_full::Text::<T, C, V>::REGISTRATION,
        (false, false) => min_keys_partial::Text::<T, C, V>::REGISTRATION,
        (true, true) => max_keys_full::Text::<T, C, V>::REGISTRATION,
        (true, false) => max_keys_partial::Text::<T, C, V>::REGISTRATION,
    };
    const PAIRS: &'static str = match (MAX, V::MODE == b'f') {
        (false, true) => min_pairs_full::Text::<T, C, V>::REGISTRATION,
        (false, false) => min_pairs_partial::Text::<T, C, V>::REGISTRATION,
        (true, true) => max_pairs_full::Text::<T, C, V>::REGISTRATION,
        (true, false) => max_pairs_partial::Text::<T, C, V>::REGISTRATION,
    };
}
fn keys_registration<C: TopKSelectionCount, V: TopKValidItems, const MAX: bool>() -> &'static str {
    Registration::<i32, C, V, MAX>::KEYS
}
fn pairs_registration<T: TopKValue, C: TopKSelectionCount, V: TopKValidItems, const MAX: bool>()
-> &'static str {
    Registration::<T, C, V, MAX>::PAIRS
}

#[inline(always)]
pub(super) fn keys<K, C, V, S, const N: usize, const A: usize, const MAX: bool>(
    input: &ThreadData<K, N, A>,
    options: TopKOptions<C, V, S>,
) -> ThreadData<K, N, A>
where
    K: Numeric,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<A>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let (ip, ib) = input.block_io_read_parts();
    let (op, ob) = output.block_io_parts();
    let payload = (ip, op, ib, ob);
    let storage = options.storage.into_raw();
    let controls = options.valid.controls(options.count.value());
    // SAFETY: preserved shared inputs and fresh aligned outputs have live,
    // disjoint arrays/maps of exactly N items. The closed provider checks all
    // input maps and wide controls before reads, including empty selections;
    // only the selected output prefix is initialized. No pointer escapes.
    // Genuine Block authentication proves participation/control uniformity and
    // retains existing scratch/manual/auto policies. Static token fields are
    // independently checked against the concrete tile by the provider.
    unsafe {
        library::checked_block_payload_call_v1::<K, _, _, _, _, N>(
            keys_registration::<C, V, MAX>,
            storage,
            controls,
            payload,
        );
    }
    output
}

#[inline(always)]
pub(super) fn pairs<
    K,
    T,
    C,
    V,
    S,
    const N: usize,
    const KA: usize,
    const VA: usize,
    const MAX: bool,
>(
    input: &ThreadData<K, N, KA>,
    values: &ThreadData<T, N, VA>,
    options: TopKOptions<C, V, S>,
) -> (ThreadData<K, N, KA>, ThreadData<T, N, VA>)
where
    K: Numeric,
    T: TopKValue,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<KA>: SupportedAlignment,
    Alignment<VA>: SupportedAlignment,
{
    let mut output = ThreadData::new();
    let mut output_values = ThreadData::new();
    let (ip, ib) = input.block_io_read_parts();
    let (op, ob) = output.block_io_parts();
    let (vp, vb) = values.block_io_read_parts();
    let (ovp, ovb) = output_values.block_io_parts();
    let payload = (ip, op, vp, ovp, ib, ob, vb, ovb);
    let storage = options.storage.into_raw();
    let controls = options.valid.controls(options.count.value());
    // SAFETY: preserved shared inputs and fresh aligned outputs have live,
    // disjoint arrays/maps of exactly N items. The closed provider checks all
    // input maps and wide controls before reads, including empty selections;
    // only the selected output prefix is initialized. No pointer escapes.
    // Genuine Block authentication proves participation/control uniformity and
    // retains existing scratch/manual/auto policies. Static token fields are
    // independently checked against the concrete tile by the provider.
    unsafe {
        library::checked_block_payload_call_v1::<K, _, _, _, _, N>(
            pairs_registration::<T, C, V, MAX>,
            storage,
            controls,
            payload,
        );
    }
    (output, output_values)
}
