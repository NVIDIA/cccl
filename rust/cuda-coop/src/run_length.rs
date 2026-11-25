// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{
    Alignment, Block, CollectiveTiles, DisjointSlice, Numeric, SupportedAlignment, TempStorageLike,
    ThreadData, ThreadGroup,
};

mod boundary;

/// Integer run lengths supported by Run Length Decode.
///
/// Sealed to i8/u8, i16/u16, i32/u32 and i64/u64. Positive lengths must precede
/// all zero padding across the Block; an all-zero tile is an empty stream.
/// Negative lengths and totals exceeding u32::MAX trap before decoding.
pub trait RunLength: Numeric + sealed::Length {}

/// A Block-uniform integer window or destination offset.
///
/// The eight fixed-width integer types widen without losing signedness; u64
/// never passes through i64. Negative runtime offsets trap. The default `()`
/// denotes zero; [`RunLengthOffset`] carries a static unsigned offset.
pub trait RunLengthDecodeOffset: Copy + sealed::Offset {}

/// Static nonnegative offset, in decoded elements, up to u64::MAX.
///
/// Normally selected with [`RunLengthDecodeOptions::decoded_window_offset_const`]
/// or [`RunLengthDecodeIntoOptions::destination_offset_const`]. Its const-u64
/// parameter rejects negative or out-of-range literals during compilation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunLengthOffset<const OFFSET: u64>;

mod sealed {
    pub trait Length {
        const TAG: [u8; 3];
    }
    pub trait Offset {
        type Wide: Copy;
        const TAG: [u8; 3];
        fn value(self) -> Self::Wide;
    }
}
macro_rules! integers {
    ($($ty:ty => $tag:literal, $wide:ty, $control:literal;)*) => {$(
        impl sealed::Length for $ty { const TAG: [u8; 3] = *$tag; }
        impl RunLength for $ty {}
        impl sealed::Offset for $ty {
            type Wide = $wide;
            const TAG: [u8; 3] = *$control;
            #[inline(always)] fn value(self) -> Self::Wide { self as $wide }
        }
        impl RunLengthDecodeOffset for $ty {}
    )*};
}
integers! {
    i8 => b"i8_", i64, b"i64"; u8 => b"u8_", u64, b"u64";
    i16 => b"i16", i64, b"i64"; u16 => b"u16", u64, b"u64";
    i32 => b"i32", i64, b"i64"; u32 => b"u32", u64, b"u64";
    i64 => b"i64", i64, b"i64"; u64 => b"u64", u64, b"u64";
}
impl sealed::Offset for () {
    type Wide = u64;
    const TAG: [u8; 3] = *b"u64";
    #[inline(always)]
    fn value(self) -> u64 {
        0
    }
}
impl RunLengthDecodeOffset for () {}
impl<const O: u64> sealed::Offset for RunLengthOffset<O> {
    type Wide = u64;
    const TAG: [u8; 3] = *b"u64";
    #[inline(always)]
    fn value(self) -> u64 {
        O
    }
}
impl<const O: u64> RunLengthDecodeOffset for RunLengthOffset<O> {}

/// Decode a fresh window of `D` values per member, starting at offset zero.
///
/// `D` is required, positive and independent of the input run extent. `O`
/// selects the inferred offset type; `S=()` requests implicit shared scratch.
/// Builders retain the other choices and may be called in either order.
/// Use [`Self::temp_storage`] for an existing explicit descriptor.
///
/// ```
/// use cuda_coop as coop;
/// let _ = coop::RunLengthDecodeOptions::<3>::default()
///     .decoded_window_offset_const::<196>();
/// let _ = coop::RunLengthDecodeOptions::<3>::default()
///     .decoded_window_offset(u64::MAX);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RunLengthDecodeOptions<const D: usize, O = (), S = ()> {
    offset: O,
    storage: S,
}

/// Decode the entire stream through internal windows of `D` items per member.
///
/// `D` is required and independent of both the input run extent and destination
/// ownership capacity. The destination offset defaults to zero, not an input
/// stream offset. `O` retains the offset type and `S=()` requests implicit scratch.
/// Builders preserve each other and may be called in either order.
///
/// ```
/// use cuda_coop as coop;
/// let _ = coop::RunLengthDecodeIntoOptions::<3>::default()
///     .destination_offset(3_u64);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RunLengthDecodeIntoOptions<const D: usize, O = (), S = ()> {
    offset: O,
    storage: S,
}

macro_rules! options {
    ($options:ident, $offset:ident, $constant:ident) => {
        impl<const D: usize> Default for $options<D> {
            fn default() -> Self {
                Self {
                    offset: (),
                    storage: (),
                }
            }
        }
        impl<const D: usize, O, S> $options<D, O, S> {
            /// Select a Block-uniform integer offset without narrowing it.
            #[inline(always)]
            pub fn $offset<P: RunLengthDecodeOffset>(self, offset: P) -> $options<D, P, S> {
                $options {
                    offset,
                    storage: self.storage,
                }
            }
            /// Select a static nonnegative element offset up to u64::MAX.
            #[inline(always)]
            pub fn $constant<const OFFSET: u64>(self) -> $options<D, RunLengthOffset<OFFSET>, S> {
                $options {
                    offset: RunLengthOffset,
                    storage: self.storage,
                }
            }
            /// Borrow existing scratch for this complete operation.
            ///
            /// Manual scratch requires explicit Block completion before reuse;
            /// auto-sync scratch inserts a trailing barrier. Into retains its
            /// prepared table throughout every internal decode window.
            #[inline(always)]
            pub fn temp_storage<P: TempStorageLike>(self, storage: P) -> $options<D, O, P> {
                $options {
                    offset: self.offset,
                    storage,
                }
            }
        }
    };
}
options!(
    RunLengthDecodeOptions,
    decoded_window_offset,
    decoded_window_offset_const
);
options!(
    RunLengthDecodeIntoOptions,
    destination_offset,
    destination_offset_const
);

/// Return a fresh, fully initialized blocked window of the decoded run stream.
///
/// Every member of a complete one-dimensional Block must participate with the
/// same offset and scratch descriptor. Values support all ten [`Numeric`] types;
/// lengths use [`RunLength`]. Both input payloads have the same positive `N`,
/// with independent `VA`/`LA` alignments, and every slot must be initialized even
/// for zero-length padding. Neither input changes. Positive lengths precede
/// all zero padding; the total must fit u32. An all-zero tile is valid.
///
/// `D` comes from the options. The fresh `ThreadData<T,D,VA>` preserves value
/// alignment. Lane `t` receives stream items starting at `offset+t*D`; tails
/// contain explicit zeros and all D slots are valid. An offset at or beyond the
/// total returns only zeros. Both `Block*N` and `Block*D` must fit i32.
///
/// Each call prepares its own run table in queried shared scratch. Omitted
/// scratch uses compiler-managed completion before reuse. Explicit manual
/// scratch requires the caller's Block barrier before conflicting reuse.
///
/// # Errors
/// Unsupported geometry or static extents fail compilation. Missing input,
/// negative lengths/offsets, interior zero padding, or u32 total overflow trap
/// before decoding. Runtime validity does not make an uninitialized Rust slot
/// readable. Requires the supported Oxide compiler with generic native registration.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn window(g: coop::ThreadGroup<coop::Block>, v: &coop::ThreadData<i64,4,32>,
///           n: &coop::ThreadData<u64,4,16>) -> coop::ThreadData<i64,3,32> {
///     coop::run_length_decode(g, v, n, coop::RunLengthDecodeOptions::<3>::default())
/// }
/// ```
#[inline(always)]
pub fn run_length_decode<
    T: Numeric,
    L: RunLength,
    O: RunLengthDecodeOffset,
    S: TempStorageLike,
    const N: usize,
    const D: usize,
    const VA: usize,
    const LA: usize,
>(
    _group: ThreadGroup<Block>,
    values: &ThreadData<T, N, VA>,
    lengths: &ThreadData<L, N, LA>,
    options: RunLengthDecodeOptions<D, O, S>,
) -> ThreadData<T, D, VA>
where
    Alignment<VA>: SupportedAlignment,
    Alignment<LA>: SupportedAlignment,
{
    boundary::window(values, lengths, options)
}

/// Decode the complete stream into a checked global destination interval.
///
/// Input, participation and length requirements match [`run_length_decode`].
/// All Block members supply the same destination and offset. `CAPACITY` fixes
/// the destination's canonical per-Block ownership region independently of `N`
/// and internal window extent `D`. The complete interval is checked against
/// both that region and actual allocation bounds before any destination write.
/// Only `offset..offset+total` changes; empty input writes nothing. An offset
/// outside allocation capacity traps even for an empty stream. The destination
/// must not overlap either input; the genuine exclusive carrier enforces its
/// Rust ownership. Completion is required before conflicting global reuse.
///
/// A u32 total is returned numerically equal to every member. The existing
/// compiler's participation and uniform-control requirements still apply to
/// any subsequent collective using this value. No result-uniformity annotation
/// or unchecked assumption is introduced by this operation.
///
/// The native driver prepares one run table, retains it across internal windows
/// and masks the final store. The same scratch rules as [`run_length_decode`]
/// apply. Both inputs and all destination elements outside the interval remain
/// unchanged. Requires the supported Oxide compiler with generic registration.
///
/// # Errors
/// Has the geometry, input and length failures of [`run_length_decode`]. Negative
/// offsets, insufficient allocation capacity or a nonempty interval outside this
/// Block's canonical ownership region trap before any destination write.
#[inline(always)]
pub fn run_length_decode_into<
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
    _group: ThreadGroup<Block>,
    values: &ThreadData<T, N, VA>,
    lengths: &ThreadData<L, N, LA>,
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    options: RunLengthDecodeIntoOptions<D, O, S>,
) -> u32
where
    Alignment<VA>: SupportedAlignment,
    Alignment<LA>: SupportedAlignment,
{
    boundary::into(values, lengths, destination, options)
}
