// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Block, Numeric, SupportedAlignment, TempStorage, TempStorageLike, ThreadData,
    ThreadGroup,
};

mod boundary;

/// Flag items unequal to their predecessor (the default mode).
///
/// Without an outside predecessor, the first tile flag is one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Heads;

/// Flag items unequal to their successor.
///
/// Without an outside successor, the last tile flag is one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tails;

/// Return separate head and tail flags, in that order.
///
/// Each omitted outside neighbor makes its corresponding boundary flag one.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadsAndTails;

/// A sealed Discontinuity mode and legal combination of typed outside neighbors.
///
/// `T` is the input type; `Pred` and `Succ` are `()` when omitted or exactly `T`.
/// [`Heads`] allows only a predecessor, [`Tails`] only a successor, and
/// [`HeadsAndTails`] allows either, both or neither. Every output flag is `i32`;
/// the static item count and minimum alignment are inherited from the input.
/// Generic mode-forwarding helpers can name this trait's [`Output`](Self::Output).
pub trait DiscontinuityMode<
    T: Numeric,
    Pred,
    Succ,
    const ITEMS_PER_THREAD: usize,
    const ALIGN: usize,
>: sealed::Mode<T, Pred, Succ> where
    Alignment<ALIGN>: SupportedAlignment,
{
    /// One flag payload for Heads/Tails, or `(heads, tails)` for HeadsAndTails.
    type Output;

    #[doc(hidden)]
    fn apply<S: TempStorageLike>(
        input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
        options: DiscontinuityOptions<T, Self, Pred, Succ, S>,
    ) -> Self::Output
    where
        Self: Sized;
}

/// Static mode, outside neighbors and scratch policy for [`discontinuity`].
///
/// [`Default`] selects [`Heads`], no outside neighbor and implicit scratch.
/// `T` is inferred from the input. `M`, `Pred`, `Succ` and `S` record consuming
/// builder selections. Builders retain all other fields; the operation accepts
/// only the eight combinations admitted by [`DiscontinuityMode`]. There is no
/// count, layout, algorithm or operator selector.
///
/// A successor cannot be used with Heads, even when it was set before the mode:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(input: &coop::ThreadData<i32, 4>) {
///     coop::discontinuity(coop::this_block(), input,
///         coop::DiscontinuityOptions::default().tile_successor_item(7).mode(coop::Heads));
/// }
/// ```
/// Outside neighbors retain the input's exact type:
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(input: &coop::ThreadData<f64, 4>) {
///     coop::discontinuity(coop::this_block(), input,
///         coop::DiscontinuityOptions::default().tile_predecessor_item(1_f32));
/// }
/// ```
pub struct DiscontinuityOptions<T, M = Heads, Pred = (), Succ = (), S = ()> {
    predecessor: Pred,
    successor: Succ,
    storage: S,
    selection: PhantomData<(T, M)>,
}

impl<T: Numeric> Default for DiscontinuityOptions<T> {
    fn default() -> Self {
        Self {
            predecessor: (),
            successor: (),
            storage: (),
            selection: PhantomData,
        }
    }
}

impl<T: Numeric, M, Pred, Succ, S> DiscontinuityOptions<T, M, Pred, Succ, S> {
    /// Select Heads, Tails or HeadsAndTails, retaining both neighbor fields.
    ///
    /// Changing mode does not reinterpret a predecessor as a successor.
    pub fn mode<M2>(self, _mode: M2) -> DiscontinuityOptions<T, M2, Pred, Succ, S> {
        DiscontinuityOptions {
            predecessor: self.predecessor,
            successor: self.successor,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Supply the exact-T, Block-uniform predecessor for Heads or HeadsAndTails.
    ///
    /// The first head flag then compares the predecessor and first input item.
    pub fn tile_predecessor_item(self, value: T) -> DiscontinuityOptions<T, M, T, Succ, S> {
        DiscontinuityOptions {
            predecessor: value,
            successor: self.successor,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Supply the exact-T, Block-uniform successor for Tails or HeadsAndTails.
    ///
    /// The last tail flag then compares the last input item and successor.
    pub fn tile_successor_item(self, value: T) -> DiscontinuityOptions<T, M, Pred, T, S> {
        DiscontinuityOptions {
            predecessor: self.predecessor,
            successor: value,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Borrow explicit scratch with its existing sharing and completion policy.
    ///
    /// Manual reuse requires a full Block barrier. An auto-sync descriptor
    /// supplies its existing trailing barrier. Options are consumed, so later
    /// calls can use ordinary mutable reborrows without requiring Copy storage.
    pub fn temp_storage<const C: usize, const A: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, A, AUTO, EX>,
    ) -> DiscontinuityOptions<T, M, Pred, Succ, &mut TempStorage<C, A, AUTO, EX>> {
        DiscontinuityOptions {
            predecessor: self.predecessor,
            successor: self.successor,
            storage,
            selection: PhantomData,
        }
    }
}

/// Flag unequal adjacent blocked items, preserving input and returning fresh flags.
///
/// `group` must be the complete [`Block`], including all lanes of an XYZ block
/// in x-fastest rank order. Every participant uses the same static mode and
/// Block-uniform outside neighbors. `input` contains positive `ITEMS_PER_THREAD`
/// initialized items per thread. All ten [`Numeric`] types are supported and T
/// is inferred from input. Shared input and mutable reborrows are accepted.
///
/// Heads compares each item with its predecessor; Tails compares it with its
/// successor. Without an outside neighbor, the first head or last tail is one.
/// Otherwise the flag is one for inequality and zero for equality. Floating
/// comparisons use native inequality, including NaNs and signed zero.
///
/// Heads and Tails return `ThreadData<i32, ITEMS_PER_THREAD, ALIGN>`.
/// HeadsAndTails returns `(heads, tails)` with two independent payloads of that
/// type. Every result slot is initialized. N and minimum ALIGN are preserved;
/// output storage also satisfies i32's natural alignment. Input values and
/// validity are unchanged. Same-extent helpers require no consumer GCE feature.
/// There is no partial operation: every initialized Load fill participates in
/// comparisons, and every input slot must be readable before this call.
///
/// Implicit scratch uses managed synchronization before dirty reuse. Explicit
/// [`TempStorage`] retains its manual/auto-sync policy. This operation adds no
/// promise to complete unrelated global accesses; use
/// `cuda_device::thread::sync_threads()` when they require Block completion.
///
/// Unsupported groups, scalar inputs, modes or neighbor combinations fail
/// compilation. Unreadable input traps during device execution. The positive
/// tile capacity must fit CUB's signed 32-bit indexing range. Execution requires
/// an Oxide-compiled CUDA kernel.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn heads<T: coop::Numeric, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     input: &coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
/// ) -> coop::ThreadData<i32, ITEMS_PER_THREAD, ALIGN>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::discontinuity(block, input, Default::default())
/// }
/// ```
#[inline(always)]
pub fn discontinuity<T, M, Pred, Succ, S, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: DiscontinuityOptions<T, M, Pred, Succ, S>,
) -> M::Output
where
    T: Numeric,
    M: DiscontinuityMode<T, Pred, Succ, ITEMS_PER_THREAD, ALIGN>,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    M::apply(input, options)
}

mod sealed {
    pub trait Mode<T, Pred, Succ> {
        type Uniform: Copy;
        const KIND: u8;
        fn controls(predecessor: Pred, successor: Succ) -> Self::Uniform;
    }
}

macro_rules! mode {
    ($mode:ty, $pred:ty, $succ:ty, $kind:literal,
     |$p:ident, $s:ident| -> $uniform:ty $body:block) => {
        impl<T: Numeric> sealed::Mode<T, $pred, $succ> for $mode {
            type Uniform = $uniform;
            const KIND: u8 = $kind;
            #[inline(always)]
            fn controls($p: $pred, $s: $succ) -> Self::Uniform $body
        }
    };
}
mode!(Heads, (), (), 0, |_p, _s| -> () {});
mode!(Heads, T, (), 1, |p, _s| -> (T,) { (p,) });
mode!(Tails, (), (), 2, |_p, _s| -> () {});
mode!(Tails, (), T, 3, |_p, s| -> (T,) { (s,) });
mode!(HeadsAndTails, (), (), 4, |_p, _s| -> () {});
mode!(HeadsAndTails, T, (), 5, |p, _s| -> (T,) { (p,) });
mode!(HeadsAndTails, (), T, 6, |_p, s| -> (T,) { (s,) });
mode!(HeadsAndTails, T, T, 7, |p, s| -> (T, T) { (p, s) });

macro_rules! single_output {
    ($mode:ty) => {
        impl<T: Numeric, Pred, Succ, const N: usize, const ALIGN: usize>
            DiscontinuityMode<T, Pred, Succ, N, ALIGN> for $mode
        where
            Alignment<ALIGN>: SupportedAlignment,
            Self: sealed::Mode<T, Pred, Succ>,
        {
            type Output = ThreadData<i32, N, ALIGN>;
            #[inline(always)]
            fn apply<S: TempStorageLike>(
                input: &ThreadData<T, N, ALIGN>,
                options: DiscontinuityOptions<T, Self, Pred, Succ, S>,
            ) -> Self::Output {
                boundary::single(input, options)
            }
        }
    };
}
single_output!(Heads);
single_output!(Tails);

impl<T: Numeric, Pred, Succ, const N: usize, const ALIGN: usize>
    DiscontinuityMode<T, Pred, Succ, N, ALIGN> for HeadsAndTails
where
    Alignment<ALIGN>: SupportedAlignment,
    Self: sealed::Mode<T, Pred, Succ>,
{
    type Output = (ThreadData<i32, N, ALIGN>, ThreadData<i32, N, ALIGN>);
    #[inline(always)]
    fn apply<S: TempStorageLike>(
        input: &ThreadData<T, N, ALIGN>,
        options: DiscontinuityOptions<T, Self, Pred, Succ, S>,
    ) -> Self::Output {
        boundary::pair(input, options)
    }
}
