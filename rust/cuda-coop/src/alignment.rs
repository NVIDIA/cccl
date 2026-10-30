// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Type-level byte alignment for [`crate::ThreadData`].
///
/// `BYTES` must be a power of two from 1 through 536,870,912 (2²⁹), inclusive.
/// These are Rust's supported explicit alignments. The payload's actual
/// alignment is at least this value and the element type's natural alignment.
/// Larger alignments can increase the payload's size through padding.
pub struct Alignment<const BYTES: usize>;

/// A supported [`Alignment`] selector.
///
/// This trait is sealed: implementations are supplied for all powers of two
/// from 1 through 2²⁹. Use this public bound when the alignment is generic:
///
/// ```
/// use cuda_coop as coop;
///
/// fn payload<const ITEMS_PER_THREAD: usize, const ALIGN: usize>(value: u32)
///     -> coop::ThreadData<u32, ITEMS_PER_THREAD, ALIGN>
/// where
///     coop::Alignment<ALIGN>: coop::SupportedAlignment,
/// {
///     let mut items = coop::ThreadData::<_, ITEMS_PER_THREAD, ALIGN>::new();
///     items.write(0, value);
///     items
/// }
///
/// assert_eq!(payload::<4, 32>(11).get(0), Some(11));
/// ```
pub trait SupportedAlignment: sealed::Alignment {}

pub(crate) mod sealed {
    pub trait Alignment {
        type Marker: Copy;
    }

    macro_rules! alignments {
        ($($marker:ident: $bytes:literal),* $(,)?) => {
            $(
                #[repr(align($bytes))]
                #[derive(Clone, Copy)]
                pub struct $marker;

                impl Alignment for super::Alignment<$bytes> {
                    type Marker = $marker;
                }

                impl super::SupportedAlignment for super::Alignment<$bytes> {}
            )*
        };
    }

    alignments! {
        A1: 1, A2: 2, A4: 4, A8: 8, A16: 16, A32: 32,
        A64: 64, A128: 128, A256: 256, A512: 512, A1024: 1024,
        A2048: 2048, A4096: 4096, A8192: 8192, A16384: 16384,
        A32768: 32768, A65536: 65536, A131072: 131072, A262144: 262144,
        A524288: 524288, A1048576: 1048576, A2097152: 2097152,
        A4194304: 4194304, A8388608: 8388608, A16777216: 16777216,
        A33554432: 33554432, A67108864: 67108864, A134217728: 134217728,
        A268435456: 268435456, A536870912: 536870912,
    }
}
