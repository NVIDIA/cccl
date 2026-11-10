// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{Alignment, Block, Numeric, SupportedAlignment, ThreadData, ThreadGroup};

mod boundary;

/// Read the next item in blocked order (the default Shuffle direction).
///
/// The last item of the last thread has no successor and remains invalid.
#[derive(Clone, Copy, Debug, Default)]
pub struct Down;

/// Read the previous item in blocked order.
///
/// The first item of the first thread has no predecessor and remains invalid.
#[derive(Clone, Copy, Debug, Default)]
pub struct Up;

/// A sealed compile-time Shuffle direction: [`Down`] or [`Up`].
/// Neither direction changes the input or wraps around the block boundary.
pub trait ShuffleMode: sealed::Mode {}
impl sealed::Mode for Down {
    const KIND: u8 = 0;
}
impl sealed::Mode for Up {
    const KIND: u8 = 1;
}
impl ShuffleMode for Down {}
impl ShuffleMode for Up {}

/// Static options for [`shuffle`]. [`Default`] selects [`Down`] at distance one.
///
/// `M` is a [`ShuffleMode`], selected with [`mode`](Self::mode). The only
/// supported distance is one, optionally written as `.distance::<1>()`.
/// There are no count, fill, algorithm or explicit-storage options.
#[derive(Clone, Copy, Debug)]
pub struct ShuffleOptions<M = Down> {
    mode: M,
}
impl Default for ShuffleOptions {
    fn default() -> Self {
        Self { mode: Down }
    }
}
impl<M: ShuffleMode> ShuffleOptions<M> {
    /// Select the compile-time direction, preserving the distance of one.
    ///
    /// ```
    /// use cuda_coop as coop;
    /// let _: coop::ShuffleOptions<coop::Up> =
    ///     coop::ShuffleOptions::default().mode(coop::Up).distance::<1>();
    /// ```
    pub fn mode<M2: ShuffleMode>(self, mode: M2) -> ShuffleOptions<M2> {
        ShuffleOptions { mode }
    }

    /// Specify the compile-time distance, which must be exactly one.
    ///
    /// `DISTANCE` is a const generic, not a runtime value. Instantiating this
    /// builder with any other distance fails compilation. It may appear before
    /// or after [`mode`](Self::mode).
    ///
    /// ```compile_fail,E0080
    /// use cuda_coop as coop;
    /// let _ = coop::ShuffleOptions::default().distance::<2>();
    /// ```
    #[inline(always)]
    pub fn distance<const DISTANCE: usize>(self) -> Self {
        let () = Distance::<DISTANCE>::VALID;
        self
    }
}

struct Distance<const DISTANCE: usize>;
impl<const DISTANCE: usize> Distance<DISTANCE> {
    const VALID: () = assert!(DISTANCE == 1, "Shuffle supports only distance one");
}

/// Shift a complete block's values by one item into a fresh partial payload,
/// preserving every input value and validity bit.
///
/// `group` must be the complete [`Block`]. `input` contains
/// `ITEMS_PER_THREAD` initialized items per thread in blocked order: thread
/// rank `r`, item `i` has index `r * ITEMS_PER_THREAD + i`. Rank is x-fastest
/// across the block's three launch dimensions. All threads must participate
/// with the same type, extent and direction. `options` defaults to [`Down`].
///
/// Down reads the next item and leaves the final block item invalid. [`Up`]
/// reads the previous item and leaves the first block item invalid. Every
/// other result slot is initialized. The boundary does not wrap or fill;
/// [`ThreadData::get`] returns `None` for that slot. A one-thread, one-item
/// block returns an entirely invalid payload. Use [`ThreadData::write`] to
/// initialize an exposed edge before a full [`crate::store`].
///
/// `T` is inferred from input and supports all ten [`Numeric`] types.
/// `ITEMS_PER_THREAD` must be positive; `ALIGN` preserves the input's supported
/// minimum alignment. Shared input and ordinary mutable reborrows are accepted.
/// The fresh result has the same `T`, extent and alignment, and changing it
/// cannot change the input. This signature needs no `generic_const_exprs`
/// feature in consumers. Scalars, Warp groups and partial Blocks are unsupported.
///
/// Every input slot must be initialized, including the one that does not
/// contribute to a valid output; otherwise execution traps before native
/// numeric reads. Oxide plans implicit shared scratch from the actual CUB
/// layout and synchronizes before reuse. Shuffle does not promise global-memory
/// completion: use `cuda_device::thread::sync_threads()` for dependent global accesses.
/// Unsupported groups, modes or distances fail compilation. Execution requires
/// an Oxide-compiled CUDA kernel; host code can check the signatures.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn next<T: coop::Numeric, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     input: &coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
/// ) -> coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::shuffle(block, input, Default::default())
/// }
/// ```
#[inline(always)]
pub fn shuffle<T, M, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: ShuffleOptions<M>,
) -> ThreadData<T, ITEMS_PER_THREAD, ALIGN>
where
    T: Numeric,
    M: ShuffleMode,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::shuffle(input, options)
}

mod sealed {
    pub trait Mode {
        const KIND: u8;
    }
}
