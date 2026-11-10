// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Block, Numeric, SupportedAlignment, TempStorage, TempStorageLike, ThreadData,
    ThreadGroup,
};

mod boundary;

/// Subtract the previous item in blocked order (the default direction).
///
/// Without an outside predecessor, copy the first active item unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct Left;

/// Subtract the next item in blocked order.
///
/// Without an outside successor, copy the last active item unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct Right;

/// A sealed legal combination of direction, count and outside neighbors.
///
/// [`Left`] permits an omitted count or `usize` and an omitted predecessor or
/// exactly `T`; its successor must be omitted. [`Right`] permits an omitted
/// count with an omitted successor or exactly `T`, or a `usize` count with no
/// successor. Its predecessor must be omitted. Omitted fields have type `()`.
/// A supplied count remains the partial form even when it equals tile capacity.
/// The device compiler checks control uniformity separately from these types.
pub trait AdjacentDifferenceDirection<T: Numeric, Count, Pred, Succ>:
    sealed::Direction<T, Count, Pred, Succ>
{
}

/// Static direction and typed controls for [`adjacent_difference`].
///
/// [`Default`] selects [`Left`], the full tile, no outside neighbor and implicit
/// scratch. `T` is inferred from the input. `D`, `Count`, `Pred`, `Succ` and `S`
/// record consuming builder selections. Builders preserve every other field;
/// only the seven combinations admitted by [`AdjacentDifferenceDirection`]
/// can be passed to the operation. There is no algorithm or operator selector.
///
/// A right successor cannot be combined with any supplied count:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(input: &coop::ThreadData<i32, 4>) {
///     coop::adjacent_difference(coop::this_block(), input,
///         coop::AdjacentDifferenceOptions::default().direction(coop::Right)
///             .valid_items(128).tile_successor_item(0));
/// }
/// ```
/// Neighbors must have the input's exact type:
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(input: &coop::ThreadData<i16, 4>) {
///     coop::adjacent_difference(coop::this_block(), input,
///         coop::AdjacentDifferenceOptions::default().tile_predecessor_item(0_i32));
/// }
/// ```
pub struct AdjacentDifferenceOptions<T, D = Left, Count = (), Pred = (), Succ = (), S = ()> {
    count: Count,
    predecessor: Pred,
    successor: Succ,
    storage: S,
    selection: PhantomData<(T, D)>,
}

impl<T: Numeric> Default for AdjacentDifferenceOptions<T> {
    fn default() -> Self {
        Self {
            count: (),
            predecessor: (),
            successor: (),
            storage: (),
            selection: PhantomData,
        }
    }
}

impl<T: Numeric, D, Count, Pred, Succ, S> AdjacentDifferenceOptions<T, D, Count, Pred, Succ, S> {
    /// Select [`Left`] or [`Right`], retaining the count and both neighbor fields.
    ///
    /// Changing direction does not reinterpret a predecessor as a successor.
    pub fn direction<D2>(
        self,
        _direction: D2,
    ) -> AdjacentDifferenceOptions<T, D2, Count, Pred, Succ, S> {
        AdjacentDifferenceOptions {
            count: self.count,
            predecessor: self.predecessor,
            successor: self.successor,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Limit subtraction to a Block-uniform prefix of `count` items.
    ///
    /// The count must be in `0..=block_threads * ITEMS_PER_THREAD`. The initialized
    /// suffix is copied unchanged; zero copies the entire input. All input slots
    /// must still be initialized. Any supplied count excludes a right successor.
    pub fn valid_items(
        self,
        count: usize,
    ) -> AdjacentDifferenceOptions<T, D, usize, Pred, Succ, S> {
        AdjacentDifferenceOptions {
            count,
            predecessor: self.predecessor,
            successor: self.successor,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Supply the exact-T, Block-uniform predecessor for [`Left`].
    ///
    /// The first active result is input minus this value; at count zero no item
    /// is active. This field is unsupported for [`Right`].
    pub fn tile_predecessor_item(
        self,
        value: T,
    ) -> AdjacentDifferenceOptions<T, D, Count, T, Succ, S> {
        AdjacentDifferenceOptions {
            count: self.count,
            predecessor: value,
            successor: self.successor,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Supply the exact-T, Block-uniform successor for a full [`Right`] tile.
    ///
    /// The last result is input minus this value. A predecessor, any supplied
    /// count or the [`Left`] direction makes this combination unsupported.
    pub fn tile_successor_item(
        self,
        value: T,
    ) -> AdjacentDifferenceOptions<T, D, Count, Pred, T, S> {
        AdjacentDifferenceOptions {
            count: self.count,
            predecessor: self.predecessor,
            successor: value,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Borrow explicit scratch with its existing sharing/completion policy.
    ///
    /// Manual reuse requires a full Block barrier. An auto-sync descriptor
    /// inserts its existing trailing barrier. Options are consumed without a
    /// Copy requirement, so a later call can use an ordinary mutable reborrow.
    pub fn temp_storage<const C: usize, const A: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, A, AUTO, EX>,
    ) -> AdjacentDifferenceOptions<T, D, Count, Pred, Succ, &mut TempStorage<C, A, AUTO, EX>> {
        AdjacentDifferenceOptions {
            count: self.count,
            predecessor: self.predecessor,
            successor: self.successor,
            storage,
            selection: PhantomData,
        }
    }
}

/// Subtract adjacent blocked items, preserving input and returning a fresh payload.
///
/// `group` must be the complete [`Block`], including every lane of an XYZ block
/// in x-fastest rank order. All participants use the same static direction and
/// Block-uniform count/neighbors. `input` contains positive `ITEMS_PER_THREAD`
/// initialized items per thread. `T` supports all ten [`Numeric`] types and is
/// inferred from input; the result preserves its extent and minimum `ALIGN`.
/// Every output slot is initialized. Shared input and mutable reborrows work;
/// this signature requires no consumer `generic_const_exprs` feature.
///
/// Default [`Left`] computes current minus previous, copying the first item
/// unless a predecessor is supplied. [`Right`] computes current minus next,
/// copying the last active item unless a full-tile successor is supplied.
/// With `valid_items`, items outside the active prefix are copied unchanged.
/// Count zero copies all input even with a predecessor. **Every input slot**,
/// including that copied suffix, must be initialized before any native read.
/// Subtraction follows CUB's typed minus; no saturation is added.
///
/// Omitting storage selects compiler-planned scratch and synchronization before
/// dirty reuse. Explicit [`TempStorage`] retains its manual/auto-sync policy.
/// This operation adds no promise to complete unrelated global accesses; use
/// `cuda_device::thread::sync_threads()` when those accesses require completion.
///
/// Unsupported groups, scalar inputs, directions or neighbor combinations fail
/// compilation. Invalid counts and unreadable input trap during device execution.
/// Count is validated before narrowing to CUB's signed index type; static tile
/// capacity must fit `i32`. Execution requires an Oxide-compiled CUDA kernel.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn differences<T: coop::Numeric, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     input: &coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
/// ) -> coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::adjacent_difference(block, input, Default::default())
/// }
/// ```
#[inline(always)]
pub fn adjacent_difference<
    T,
    D,
    Count,
    Pred,
    Succ,
    S,
    const ITEMS_PER_THREAD: usize,
    const ALIGN: usize,
>(
    _group: ThreadGroup<Block>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: AdjacentDifferenceOptions<T, D, Count, Pred, Succ, S>,
) -> ThreadData<T, ITEMS_PER_THREAD, ALIGN>
where
    T: Numeric,
    D: AdjacentDifferenceDirection<T, Count, Pred, Succ>,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::adjacent_difference(input, options)
}

mod sealed {
    pub trait Direction<T, Count, Pred, Succ> {
        type Uniform: Copy;
        const KIND: u8;
        fn controls<const ITEMS: usize>(
            count: Count,
            predecessor: Pred,
            successor: Succ,
        ) -> Self::Uniform;
    }
}

// Only these seven forms can reach a registration with the matching scalar ABI.
macro_rules! direction {
    ($direction:ty, $count:ty, $pred:ty, $succ:ty, $kind:literal,
     |$items:ident; $c:ident, $p:ident, $s:ident| -> $uniform:ty $body:block) => {
        impl<T: Numeric> sealed::Direction<T, $count, $pred, $succ> for $direction {
            type Uniform = $uniform;
            const KIND: u8 = $kind;
            #[inline(always)]
            fn controls<const $items: usize>($c: $count, $p: $pred, $s: $succ) -> Self::Uniform $body
        }
        impl<T: Numeric> AdjacentDifferenceDirection<T, $count, $pred, $succ> for $direction {}
    };
}
direction!(Left, (), (), (), 0, |ITEMS; _c, _p, _s| -> () {});
direction!(Left, (), T, (), 1, |ITEMS; _c, p, _s| -> (T,) { (p,) });
direction!(Left, usize, (), (), 2, |ITEMS; c, _p, _s| -> (i32,) { (boundary::count::<ITEMS>(c),) });
direction!(Left, usize, T, (), 3, |ITEMS; c, p, _s| -> (i32, T) { (boundary::count::<ITEMS>(c), p) });
direction!(Right, (), (), (), 4, |ITEMS; _c, _p, _s| -> () {});
direction!(Right, (), (), T, 5, |ITEMS; _c, _p, s| -> (T,) { (s,) });
direction!(Right, usize, (), (), 6, |ITEMS; c, _p, _s| -> (i32,) { (boundary::count::<ITEMS>(c),) });
