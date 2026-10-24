// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Block, Numeric, SupportedAlignment, TempStorage, TempStorageLike, ThreadData,
    ThreadGroup,
};

mod boundary;

/// Count samples with atomic additions (the default Histogram algorithm).
#[derive(Clone, Copy, Debug, Default)]
pub struct Atomic;

/// Count samples by sorting a private copy, preserving the input payload.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sort;

/// Sealed Histogram input types: u8, i32, u32, i64 and u64.
///
/// Every input value must lie in `[0, BINS)`. The native boundary checks this
/// before CUB reads the samples; it does not clamp or filter invalid values.
pub trait HistogramSample: Numeric + sealed::Sample {}

/// Sealed Histogram counter types: i32, u32, i64 and u64.
///
/// The counter type is independent of the sample type. A fresh histogram has
/// at most `i32::MAX` samples, so every supported counter represents its count.
pub trait HistogramCounter: Numeric + sealed::Counter {}

/// Sealed Histogram algorithms: [`Atomic`] and [`Sort`].
///
/// Both return the same striped counters and preserve input. They use the
/// complete queried scratch layout, including the intermediate counters.
pub trait HistogramAlgorithm: sealed::Algorithm {}

/// Static bin count, independent counter shape, algorithm and scratch policy.
///
/// `BINS` is required. Canonical [`Default`] selects i32 counters, one counter
/// per thread (`K = 1`), [`Atomic`] and implicit scratch. `BINS` and `K` must be
/// positive; the complete Block's `B * K` capacity must cover `BINS`. Bin count,
/// input tile size and output tile capacity must fit signed 32-bit indexing.
///
/// Builders consume options and preserve other choices. `C` is the counter
/// type, `A` the algorithm and `S` the existing storage selection. Input type,
/// extent and alignment are inferred separately from the borrowed payload.
///
/// ```
/// use cuda_coop as coop;
/// let _: coop::HistogramOptions<32> = Default::default();
/// let _: coop::HistogramOptions<65, u64, 2, coop::Sort> =
///     coop::HistogramOptions::<65>::default()
///         .counter_type::<u64>().bins_per_thread::<2>().algorithm(coop::Sort);
/// ```
///
/// Counters cannot use narrow integer or floating-point types:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// let _ = coop::HistogramOptions::<32>::default().counter_type::<u8>();
/// ```
pub struct HistogramOptions<const BINS: usize, C = i32, const K: usize = 1, A = Atomic, S = ()> {
    storage: S,
    selection: PhantomData<(C, A)>,
}

impl<const BINS: usize> Default for HistogramOptions<BINS> {
    fn default() -> Self {
        Self {
            storage: (),
            selection: PhantomData,
        }
    }
}

impl<const BINS: usize, C, const K: usize, A, S> HistogramOptions<BINS, C, K, A, S> {
    /// Select the output counter type independently of the input samples.
    pub fn counter_type<C2: HistogramCounter>(self) -> HistogramOptions<BINS, C2, K, A, S> {
        HistogramOptions {
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Select the number of striped output counters owned by each thread.
    ///
    /// All `K2` slots are initialized, including zero padding beyond `BINS`.
    /// The provider rejects a nonpositive or insufficient output capacity.
    pub fn bins_per_thread<const K2: usize>(self) -> HistogramOptions<BINS, C, K2, A, S> {
        HistogramOptions {
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Select [`Atomic`] or [`Sort`], retaining counter shape and storage.
    pub fn algorithm<A2: HistogramAlgorithm>(
        self,
        _algorithm: A2,
    ) -> HistogramOptions<BINS, C, K, A2, S> {
        HistogramOptions {
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Borrow explicit scratch with its existing sharing and completion policy.
    ///
    /// Manual reuse requires `cuda_device::thread::sync_threads()` after the
    /// call. Auto-sync storage supplies its existing trailing barrier. Options
    /// are consumed, allowing ordinary mutable reborrows on successive calls.
    pub fn temp_storage<
        const CAPACITY: usize,
        const ALIGN: usize,
        const AUTO: bool,
        const EX: bool,
    >(
        self,
        storage: &mut TempStorage<CAPACITY, ALIGN, AUTO, EX>,
    ) -> HistogramOptions<BINS, C, K, A, &mut TempStorage<CAPACITY, ALIGN, AUTO, EX>> {
        HistogramOptions {
            storage,
            selection: PhantomData,
        }
    }
}

/// Count a complete one-dimensional Block's samples into fresh striped counters.
///
/// Every member of `group` participates with `ITEMS_PER_THREAD` initialized
/// samples. Supported samples are [`HistogramSample`]'s five integer types;
/// each value must be in `[0, BINS)`. Unreadable or out-of-range samples trap
/// before native Histogram accesses. There is no partial count or filtering:
/// Load's zero-filled tail samples contribute to bin zero.
///
/// `BINS` is the number of real bins, `C` the independent counter type, and `K`
/// the per-thread output extent. Rank `t`, slot `i` owns bin `t + i * B`, where
/// `B` is the Block size. Every slot is valid; bins beyond `BINS` contain zero.
/// The output has default alignment, including C's natural alignment. Input
/// values and validity are unchanged, including with [`Sort`]. Shared input
/// and mutable reborrows are accepted. No generic-const-expression feature is
/// needed in an ordinary consumer, including generic T/N/ALIGN/C/K helpers.
///
/// Store results with [`crate::Striped`] and an output ownership grain of K
/// to obtain increasing bin order. Input T/N and output C/K are independent.
/// Each call starts from zero. Scratch includes both CUB's temporary storage
/// and intermediate counters, even for Atomic. The internal readiness barrier
/// precedes result reads; later scratch reuse follows the existing implicit or
/// explicit [`TempStorage`] policy. Unrelated global accesses gain no completion
/// guarantee from this operation.
///
/// Unsupported types/groups and invalid static sizes fail compilation. Only
/// complete 1D Blocks are supported, with `BINS`, `B * ITEMS_PER_THREAD` and
/// `B * K` at most `i32::MAX`, and `B * K >= BINS`. Required BINS is specified
/// in options; it cannot be inferred from input. Execution requires an
/// Oxide-compiled CUDA kernel with support for generic registration constants.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn counts<T: coop::HistogramSample, C: coop::HistogramCounter,
///           const ITEMS_PER_THREAD: usize, const ALIGN: usize,
///           const BINS: usize, const K: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     input: &coop::ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
/// ) -> coop::ThreadData<C, K>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::histogram(block, input, coop::HistogramOptions::<BINS>::default()
///         .counter_type::<C>().bins_per_thread::<K>())
/// }
/// ```
///
/// Floating-point samples are unsupported:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(input: &coop::ThreadData<f32, 4>) {
///     coop::histogram(coop::this_block(), input, coop::HistogramOptions::<32>::default());
/// }
/// ```
#[inline(always)]
pub fn histogram<
    T,
    C,
    A,
    S,
    const BINS: usize,
    const ITEMS_PER_THREAD: usize,
    const K: usize,
    const ALIGN: usize,
>(
    _group: ThreadGroup<Block>,
    input: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: HistogramOptions<BINS, C, K, A, S>,
) -> ThreadData<C, K>
where
    T: HistogramSample,
    C: HistogramCounter,
    A: HistogramAlgorithm,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::histogram(input, options)
}

mod sealed {
    pub trait Sample {}
    pub trait Counter {
        const TAG: [u8; 3];
    }
    pub trait Algorithm {
        const SORT: bool;
    }
}

macro_rules! samples {
    ($($ty:ty),+ $(,)?) => {$ (
        impl sealed::Sample for $ty {}
        impl HistogramSample for $ty {}
    )+};
}
samples!(u8, i32, u32, i64, u64);

macro_rules! counters {
    ($(($ty:ty, $tag:literal)),+ $(,)?) => {$ (
        impl sealed::Counter for $ty { const TAG: [u8; 3] = *$tag; }
        impl HistogramCounter for $ty {}
    )+};
}
counters!((i32, b"i32"), (u32, b"u32"), (i64, b"i64"), (u64, b"u64"));

impl sealed::Algorithm for Atomic {
    const SORT: bool = false;
}
impl HistogramAlgorithm for Atomic {}
impl sealed::Algorithm for Sort {
    const SORT: bool = true;
}
impl HistogramAlgorithm for Sort {}
