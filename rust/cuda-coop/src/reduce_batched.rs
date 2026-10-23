// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, GroupScope, Numeric, ReduceOperator, Striped, Sum, SupportedAlignment, ThreadData,
    ThreadGroup, ThreadsWithinWarp, Warp,
};
use cuda_device::library;

mod boundary;

/// Capacity of each lane's batched-reduction result: ceiling division of
/// `batches` by `width`, without overflowing the addition in `(batches + width - 1)`.
///
/// Both arguments must be positive. This is capacity, not the number of valid
/// slots in every lane. [`reduce_batched`] determines validity from output layout
/// and lane rank. Generic calculated signatures repeat this expression in a
/// `[(); batched_extent(B, W)]:` well-formedness bound.
pub const fn batched_extent(batches: usize, width: usize) -> usize {
    assert!(batches > 0);
    assert!(width > 0);
    batches.div_ceil(width)
}

/// Consecutive batched results per lane: lane `r`, slot `i` owns batch
/// `r * batched_extent(B, W) + i`. Slots with a batch index at least `B` are invalid.
#[derive(Clone, Copy, Debug, Default)]
pub struct Blocked;

/// A sealed batched-result layout: [`Striped`] (default) or [`Blocked`].
///
/// Striped lane `r`, slot `i` owns batch `r + i * W`; Blocked owns
/// `r * batched_extent(B, W) + i`. This selection describes result ownership,
/// independently of a Load/Store algorithm.
pub trait BatchedOutputLayout: sealed::Layout {}
impl sealed::Layout for Striped {
    const KIND: u8 = 0;
}
impl sealed::Layout for Blocked {
    const KIND: u8 = 1;
}
impl BatchedOutputLayout for Striped {}
impl BatchedOutputLayout for Blocked {}

/// A sealed group supporting batched reduction: physical [`Warp`] or logical
/// widths 1, 2, 4, 8, 16 and 32, with either exhaustive selection.
///
/// The enclosing Block must contain complete physical warps. Block groups and
/// non-power-of-two prefixes are unsupported. All members of each selected
/// group must participate; sibling logical groups may choose different paths.
pub trait ReduceBatchedGroup: GroupScope + sealed::Group {
    /// Number of lanes contributing to each batch.
    ///
    /// In helpers also bounded by [`crate::LoadStoreGroup`], spell this as
    /// `<G as ReduceBatchedGroup>::WIDTH` to distinguish the group capabilities.
    const WIDTH: usize;
}
macro_rules! warp_group {
    ($scope:ty, $width:literal $(, $e:ident)?) => {
        impl<$(const $e: bool)?> ReduceBatchedGroup for $scope { const WIDTH: usize = $width; }
        impl<$(const $e: bool)?> sealed::Group for $scope {
            #[inline(always)]
            unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, P: Copy, const I: usize>(
                registration: R, payload: P,
            ) {
                // SAFETY: the private boundary provides live payloads and the
                // matching recipe; the checked marker proves participation.
                unsafe { library::checked_warp_payload_call_v1::<T, _, _, _, $width, I>(registration, (), payload) }
            }
        }
    };
}
warp_group!(Warp, 32);
warp_group!(ThreadsWithinWarp<1, E>, 1, E);
warp_group!(ThreadsWithinWarp<2, E>, 2, E);
warp_group!(ThreadsWithinWarp<4, E>, 4, E);
warp_group!(ThreadsWithinWarp<8, E>, 8, E);
warp_group!(ThreadsWithinWarp<16, E>, 16, E);
warp_group!(ThreadsWithinWarp<32, E>, 32, E);

/// Static operator and output layout for [`reduce_batched`].
///
/// `T` is inferred from the borrowed input. Canonical [`Default`] selects
/// [`Sum`] and [`Striped`]; builders consume options and preserve the other
/// selection. Built-ins use [`ReduceOperator`]'s numeric and integer bounds.
/// There is no count, seed, algorithm or explicit-storage option.
pub struct ReduceBatchedOptions<T, Op = Sum, L = Striped> {
    binary_op: Op,
    selection: PhantomData<(T, L)>,
}
impl<T: Numeric> Default for ReduceBatchedOptions<T> {
    fn default() -> Self {
        Self {
            binary_op: Sum,
            selection: PhantomData,
        }
    }
}
impl<T: Numeric, Op, L> ReduceBatchedOptions<T, Op, L> {
    /// Select a built-in operator, retaining its actual generic value.
    pub fn binary_op<Op2: ReduceOperator<T>>(self, op: Op2) -> ReduceBatchedOptions<T, Op2, L> {
        ReduceBatchedOptions {
            binary_op: op,
            selection: PhantomData,
        }
    }
    /// Select [`Striped`] or [`Blocked`] result ownership.
    pub fn output_layout<L2: BatchedOutputLayout>(
        self,
        _layout: L2,
    ) -> ReduceBatchedOptions<T, Op, L2> {
        ReduceBatchedOptions {
            binary_op: self.binary_op,
            selection: PhantomData,
        }
    }
}

/// Reduce each input slot independently across a Warp group into fresh partial
/// [`ThreadData`], preserving all input values and validity bits.
///
/// Each lane supplies `ITEMS_PER_THREAD` initialized numeric slots (batches).
/// The result capacity is `ceil(ITEMS_PER_THREAD / G::WIDTH)`, with default
/// alignment. Each batch has one owning result slot according to the selected
/// [`BatchedOutputLayout`]; absent batches remain invalid and `get` returns None.
/// Input may be shared or mutably reborrowed; scalar input is unsupported.
///
/// All selected members participate. Uninitialized input traps before native
/// numeric reads. Compiler-planned implicit Warp scratch is synchronized before
/// reuse; the operation does not imply global-memory completion. Use the matching
/// group's `sync()` when a later global access depends on preceding accesses.
///
/// This crate and each crate instantiating the calculated result signature use
/// unstable `generic_const_exprs`. Ordinary payload/Load/Scan/Reduce consumers do
/// not need that feature. Generic helpers carry the explicit well-formedness
/// bound, as below; no caller-chosen output extent is needed.
///
/// ```no_run
/// #![feature(generic_const_exprs)]
/// #![allow(incomplete_features)]
/// use cuda_coop as coop;
/// fn batches<T: coop::Numeric, G: coop::ReduceBatchedGroup, const ITEMS_PER_THREAD: usize>(
///     group: coop::ThreadGroup<G>, values: &coop::ThreadData<T, ITEMS_PER_THREAD>,
/// ) -> coop::ThreadData<T, { coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH) }>
/// where [(); coop::batched_extent(ITEMS_PER_THREAD, G::WIDTH)]:,
/// {
///     coop::reduce_batched(group, values, Default::default())
/// }
/// ```
#[inline(always)]
pub fn reduce_batched<T, G, Op, L, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<G>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: ReduceBatchedOptions<T, Op, L>,
) -> ThreadData<T, { batched_extent(ITEMS_PER_THREAD, <G as ReduceBatchedGroup>::WIDTH) }>
where
    T: Numeric,
    G: ReduceBatchedGroup,
    Op: ReduceOperator<T>,
    L: BatchedOutputLayout,
    Alignment<ALIGN>: SupportedAlignment,
    [(); batched_extent(ITEMS_PER_THREAD, <G as ReduceBatchedGroup>::WIDTH)]:,
{
    boundary::reduce::<T, G, Op, L, ITEMS_PER_THREAD, ALIGN>(input, options)
}

mod sealed {
    pub trait Layout {
        const KIND: u8;
    }
    pub trait Group {
        unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, P: Copy, const I: usize>(
            registration: R,
            payload: P,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_handles_partial_and_maximum_batches() {
        assert_eq!(batched_extent(1, 32), 1);
        assert_eq!(batched_extent(10, 8), 2);
        assert_eq!(batched_extent(usize::MAX, 1), usize::MAX);
        assert_eq!(batched_extent(usize::MAX, 32), usize::MAX / 32 + 1);
    }
}
