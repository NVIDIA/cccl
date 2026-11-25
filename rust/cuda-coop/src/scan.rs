// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Block, GroupScope, Numeric, SupportedAlignment, TempStorage, TempStorageLike,
    ThreadData, ThreadGroup, ThreadsWithinWarp, Warp,
};
use cuda_device::library;

mod boundary;

/// Exclusive prefix order: the first result is the initial value.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exclusive;

/// Inclusive prefix order: each result includes the current input item.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inclusive;

macro_rules! operators {
    ($(($name:ident, $kind:literal, $doc:literal)),+ $(,)?) => {$ (
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl sealed::Operator for $name { const KIND: u8 = $kind; }
    )+};
}
operators!(
    (
        Sum,
        0,
        "Addition in the input numeric type; the default Scan and Reduce operator."
    ),
    (Multiplies, 1, "Multiplication in the input numeric type."),
    (Min, 2, "CUB minimum in the input numeric type."),
    (Max, 3, "CUB maximum in the input numeric type."),
    (BitAnd, 4, "Bitwise AND for integer inputs."),
    (BitOr, 5, "Bitwise OR for integer inputs."),
    (BitXor, 6, "Bitwise XOR for integer inputs."),
);

/// A sealed static Scan operator for numeric input type `T`.
///
/// [`Sum`], [`Multiplies`], [`Min`] and [`Max`] accept every [`Numeric`] type.
/// [`BitAnd`], [`BitOr`] and [`BitXor`] accept the eight integer types only.
/// Generic helpers may carry this bound. Custom operators are not yet supported.
pub trait ScanOperator<T: Numeric>: sealed::Operator {}
impl<T: Numeric> ScanOperator<T> for Sum {}
impl<T: Numeric> ScanOperator<T> for Multiplies {}
impl<T: Numeric> ScanOperator<T> for Min {}
impl<T: Numeric> ScanOperator<T> for Max {}
macro_rules! integer_operators {
    ($($t:ty),+ $(,)?) => {$ (
        impl ScanOperator<$t> for BitAnd {}
        impl ScanOperator<$t> for BitOr {}
        impl ScanOperator<$t> for BitXor {}
    )+};
}
integer_operators!(i8, u8, i16, u16, i32, u32, i64, u64);

macro_rules! algorithms {
    ($(($name:ident, $kind:literal, $doc:literal)),+ $(,)?) => {$ (
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl sealed::Algorithm for $name { const KIND: u8 = $kind; }
        impl ScanAlgorithm for $name {}
    )+};
}
algorithms!(
    (
        Raking,
        1,
        "CUB's raking Block algorithm; default for Scan and an explicit choice for Reduce."
    ),
    (RakingMemoize, 2, "CUB's memoized raking Block Scan."),
    (
        WarpScans,
        3,
        "CUB's warp-based Block Scan; requires a block size divisible by 32."
    ),
);

/// A sealed Block Scan algorithm or omitted selection `()`.
///
/// Block resolves omission to [`Raking`]. Warp accepts only omission, so even
/// explicit `Raking` is rejected for Warp. This trait is separate from the
/// Load/Store [`Algorithm`](crate::Algorithm) selectors.
pub trait ScanAlgorithm: sealed::Algorithm {}
impl sealed::Algorithm for () {
    const KIND: u8 = 0;
}
impl ScanAlgorithm for () {}

/// A sealed supported combination of Scan mode, operator and initial value.
///
/// `()` and [`Exclusive`] allow an omitted seed `()` only for [`Sum`], selecting
/// typed zero. A supplied seed has exactly type `T` and is allowed for every
/// supported operator. [`Inclusive`] allows only an omitted seed. Mode `()` is
/// the omitted selection used by default options and the fixed-name functions.
/// Runtime seed uniformity is checked by the device compiler, not this trait.
pub trait ScanMode<T: Numeric, Op: ScanOperator<T>, Seed>: sealed::Mode<T, Op, Seed> {}
macro_rules! exclusive_modes {
    ($($mode:ty),+) => {$ (
        impl<T: Numeric> sealed::Mode<T, Sum, ()> for $mode {
            type Uniform = ();
            const KIND: u8 = 0;
            #[inline(always)] fn controls(_: ()) {}
        }
        impl<T: Numeric> ScanMode<T, Sum, ()> for $mode {}
        impl<T: Numeric, Op: ScanOperator<T>> sealed::Mode<T, Op, T> for $mode {
            type Uniform = (T,);
            const KIND: u8 = 1 + Op::KIND;
            #[inline(always)] fn controls(seed: T) -> (T,) { (seed,) }
        }
        impl<T: Numeric, Op: ScanOperator<T>> ScanMode<T, Op, T> for $mode {}
    )+};
}
exclusive_modes!((), Exclusive);
impl<T: Numeric, Op: ScanOperator<T>> sealed::Mode<T, Op, ()> for Inclusive {
    type Uniform = ();
    const KIND: u8 = 8 + Op::KIND;
    #[inline(always)]
    fn controls(_: ()) {}
}
impl<T: Numeric, Op: ScanOperator<T>> ScanMode<T, Op, ()> for Inclusive {}

/// A sealed group supporting a Scan algorithm and storage choice.
///
/// Complete Blocks support scalar and payload Scan, with implicit or explicit
/// storage. Physical [`Warp`] and [`ThreadsWithinWarp`] widths 1/2/4/8/16/32
/// support scalar Scan with omitted algorithm and implicit storage only. Warp
/// requires complete physical warps in the enclosing block. Membership,
/// participation and exact launch geometry are enforced by the compiler.
///
/// Explicit Block algorithms are not accepted for Warp:
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(value: i32) {
///     coop::scan(coop::this_warp(), value,
///         coop::ScanOptions::default().algorithm(coop::Raking));
/// }
/// ```
pub trait ScanGroup<A: ScanAlgorithm, S: TempStorageLike>: GroupScope + sealed::Group {}
impl<A: ScanAlgorithm, S: TempStorageLike> ScanGroup<A, S> for Block {}
impl ScanGroup<(), ()> for Warp {}
impl sealed::Group for Block {
    const WARP: bool = false;
    #[inline(always)]
    unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, S, U: Copy, P: Copy, const I: usize>(
        registration: R,
        storage: S,
        uniform: U,
        payload: P,
    ) {
        // SAFETY: the private boundary supplies live typed payloads and the exact recipe.
        unsafe {
            library::checked_block_payload_call_v1::<T, _, _, _, _, I>(
                registration,
                storage,
                uniform,
                payload,
            )
        }
    }
}
macro_rules! warp_group {
    ($scope:ty, $width:expr $(, $e:ident)?) => {
        impl<$(const $e: bool)?> sealed::Group for $scope {
            const WARP: bool = true;
            #[inline(always)]
            unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, S, U: Copy, P: Copy, const I: usize>(
                registration: R, _storage: S, uniform: U, payload: P,
            ) {
                // SAFETY: ScanGroup permits implicit storage only; the compiler enforces the canonical width.
                unsafe { library::checked_warp_payload_call_v1::<T, _, _, _, $width, I>(registration, uniform, payload) }
            }
        }
    };
}
warp_group!(Warp, 32);
macro_rules! logical_groups {
    ($($width:literal),+) => {$ (
        impl<const E: bool> ScanGroup<(), ()> for ThreadsWithinWarp<$width, E> {}
        warp_group!(ThreadsWithinWarp<$width, E>, $width, E);
    )+};
}
logical_groups!(1, 2, 4, 8, 16, 32);

/// Static selections and runtime seed/storage for [`scan`] and its named forms.
///
/// `T` is inferred from the input. `M`, `Op` and `A` select mode, operator and
/// algorithm; `Seed` is either `()` or exactly `T`; `S` is implicit `()` or a
/// mutable [`TempStorage`] borrow. Defaults are omitted mode (exclusive for
/// `scan`), [`Sum`], omitted algorithm (Block [`Raking`]), zero seed and managed
/// storage. Only this canonical specialization implements Default, keeping
/// `Default::default()` unambiguous at all five entry points.
///
/// Builders consume options and preserve other fields. Fixed-name functions
/// select their own mode and accept only omitted M. Warp rejects an explicit
/// algorithm or storage. Inclusive forms reject seeds; non-sum exclusive forms
/// require one. A seed must be uniform within the participating group.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn prefixes<const ITEMS_PER_THREAD: usize>(
///     values: &coop::ThreadData<i32, ITEMS_PER_THREAD>, seed: i32,
/// ) -> coop::ThreadData<i32, ITEMS_PER_THREAD> {
///     coop::exclusive_scan(coop::this_block(), values,
///         coop::ScanOptions::default().scan_op(coop::Max).initial_value(seed))
/// }
/// ```
///
/// All fixed-name defaults preserve a generic scalar's type:
/// ```no_run
/// use cuda_coop as coop;
/// fn scalar<T: coop::Numeric>(value: T) {
///     let group = coop::this_warp();
///     let _: T = coop::scan(group, value, Default::default());
///     let _: T = coop::exclusive_sum(group, value, Default::default());
///     let _: T = coop::inclusive_sum(group, value, Default::default());
///     let _: T = coop::exclusive_scan(group, value, Default::default());
///     let _: T = coop::inclusive_scan(group, value, Default::default());
/// }
/// ```
///
/// Non-sum exclusive Scan requires a seed:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(value: i32) {
///     coop::scan(coop::this_block(), value,
///         coop::ScanOptions::default().scan_op(coop::Max));
/// }
/// ```
/// Inclusive Scan forbids a seed:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(value: i32) {
///     coop::scan(coop::this_block(), value,
///         coop::ScanOptions::default().mode(coop::Inclusive).initial_value(0));
/// }
/// ```
/// Bitwise operators require integer inputs:
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(value: f32) {
///     coop::inclusive_scan(coop::this_block(), value,
///         coop::ScanOptions::default().scan_op(coop::BitAnd));
/// }
/// ```
/// Seeds preserve the input type:
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(value: i16) {
///     coop::exclusive_scan(coop::this_block(), value,
///         coop::ScanOptions::default().initial_value(0_i32));
/// }
/// ```
pub struct ScanOptions<T, M = (), Op = Sum, A = (), Seed = (), S = ()> {
    scan_op: Op,
    initial_value: Seed,
    storage: S,
    selection: PhantomData<(T, M, A)>,
}
impl<T: Numeric> Default for ScanOptions<T> {
    fn default() -> Self {
        Self {
            scan_op: Sum,
            initial_value: (),
            storage: (),
            selection: PhantomData,
        }
    }
}
impl<T: Numeric, M, Op, A, Seed, S> ScanOptions<T, M, Op, A, Seed, S> {
    /// Select [`Exclusive`] or [`Inclusive`] for [`scan`].
    pub fn mode<M2>(self, _mode: M2) -> ScanOptions<T, M2, Op, A, Seed, S> {
        ScanOptions {
            scan_op: self.scan_op,
            initial_value: self.initial_value,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Select a built-in operator, retaining its actual generic value.
    pub fn scan_op<Op2: ScanOperator<T>>(self, op: Op2) -> ScanOptions<T, M, Op2, A, Seed, S> {
        ScanOptions {
            scan_op: op,
            initial_value: self.initial_value,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Select a Block algorithm. Warp accepts only omitted selection `()`.
    pub fn algorithm<A2: ScanAlgorithm>(
        self,
        _algorithm: A2,
    ) -> ScanOptions<T, M, Op, A2, Seed, S> {
        ScanOptions {
            scan_op: self.scan_op,
            initial_value: self.initial_value,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Supply the exact-T, group-uniform seed required by a non-sum exclusive Scan.
    pub fn initial_value(self, value: T) -> ScanOptions<T, M, Op, A, T, S> {
        ScanOptions {
            scan_op: self.scan_op,
            initial_value: value,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Borrow explicit Block scratch with its existing reuse/completion policy.
    ///
    /// Warp rejects this choice. The operation consumes this options value,
    /// permitting a fresh mutable reborrow for the next operation.
    pub fn temp_storage<const C: usize, const L: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, L, AUTO, EX>,
    ) -> ScanOptions<T, M, Op, A, Seed, &mut TempStorage<C, L, AUTO, EX>> {
        ScanOptions {
            scan_op: self.scan_op,
            initial_value: self.initial_value,
            storage,
            selection: PhantomData,
        }
    }
}

/// Options shared by [`exclusive_sum`] and [`inclusive_sum`].
pub type SumScanOptions<T, A = (), S = ()> = ScanOptions<T, (), Sum, A, (), S>;
/// Operator, seed, algorithm and storage options for [`exclusive_scan`].
pub type ExclusiveScanOptions<T, Op = Sum, A = (), Seed = (), S = ()> =
    ScanOptions<T, (), Op, A, Seed, S>;
/// Operator, algorithm and storage options for [`inclusive_scan`]; no seed.
pub type InclusiveScanOptions<T, Op = Sum, A = (), S = ()> = ScanOptions<T, (), Op, A, (), S>;

/// Scan numeric values in group order, returning a fresh fully initialized result.
///
/// `T` and the result shape are inferred from `value`. A scalar returns T.
/// A shared or mutable Block payload borrow returns a fresh payload with the
/// same dtype, extent and alignment; input values and validity are preserved.
/// Block order is x-fastest thread rank, then each thread's items in index order.
/// Warp accepts a scalar only, with prefixes restarting within each logical group.
/// Every required input slot must be initialized; no valid-count option exists.
///
/// Defaults select exclusive sum with typed-zero seed and Block Raking. Inclusive
/// mode includes the current item and forbids a seed. Non-sum exclusive mode
/// requires an exact-T, group-uniform seed. All ten [`Numeric`] widths are
/// supported, except bitwise operators require integers. CUB may regroup
/// associative arithmetic; overflow and floating behavior follow the input
/// dtype and underlying primitive, not sequential host arithmetic guarantees.
///
/// Every member must reach matching operations. Oxide enforces the exact launch
/// geometry, canonical group participation and seed uniformity. Block supports
/// all three algorithms and implicit or explicit storage; Warp supports only
/// omitted algorithm/storage. [`WarpScans`] requires a block size divisible by 32.
/// Implicit scratch inserts completion before dirty reuse; it is not a trailing
/// completion guarantee. Explicit manual scratch requires completion before
/// conflicting reuse, and auto-sync storage inserts a trailing block barrier.
/// Scan alone is not completion of prior global-memory accesses. The operation
/// uses the existing checked native provider and requires the pinned Oxide toolchain.
///
/// # Panics
/// An unwritten required input traps before its numeric read. Unsupported
/// type/options, group participation, geometry, or scratch reuse fail compilation.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn prefixes<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
///     values: &coop::ThreadData<T, ITEMS_PER_THREAD>,
/// ) -> coop::ThreadData<T, ITEMS_PER_THREAD> {
///     coop::scan(coop::this_block(), values, Default::default())
/// }
/// ```
#[inline(always)]
pub fn scan<
    T: Numeric,
    V: ScanValue<T, G>,
    G: ScanGroup<A, S>,
    M: ScanMode<T, Op, Seed>,
    Op: ScanOperator<T>,
    A: ScanAlgorithm,
    Seed,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: ScanOptions<T, M, Op, A, Seed, S>,
) -> V::Output {
    value.scan(group, options)
}

/// Exclusive sum with typed-zero seed; input/result and completion follow [`scan`].
///
/// Scalars return scalars; borrowed Block payloads return fresh payloads. Options
/// select only Block algorithm/storage. Default options also support scalar Warp.
#[inline(always)]
pub fn exclusive_sum<
    T: Numeric,
    V: ScanValue<T, G>,
    G: ScanGroup<A, S>,
    A: ScanAlgorithm,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: SumScanOptions<T, A, S>,
) -> V::Output {
    scan(group, value, options)
}

/// Inclusive sum, including the current item; ownership and completion follow [`scan`].
///
/// No seed or replacement operator is accepted. Default options support Block
/// scalar/payload inputs and scalar Warp inputs.
#[inline(always)]
pub fn inclusive_sum<
    T: Numeric,
    V: ScanValue<T, G>,
    G: ScanGroup<A, S>,
    A: ScanAlgorithm,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: SumScanOptions<T, A, S>,
) -> V::Output {
    scan(group, value, options.mode(Inclusive))
}

/// Exclusive Scan with a static operator; result ownership and completion follow [`scan`].
///
/// Defaults select sum/typed zero. Every other operator requires an exact-T,
/// group-uniform [`initial_value`](ScanOptions::initial_value). The first result
/// is the seed and every later result combines it with preceding inputs.
#[inline(always)]
pub fn exclusive_scan<
    T: Numeric,
    V: ScanValue<T, G>,
    G: ScanGroup<A, S>,
    Op: ScanOperator<T>,
    A: ScanAlgorithm,
    Seed,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: ExclusiveScanOptions<T, Op, A, Seed, S>,
) -> V::Output
where
    (): ScanMode<T, Op, Seed>,
{
    scan(group, value, options)
}

/// Inclusive Scan with a static operator and no seed; ownership/completion follow [`scan`].
///
/// Defaults select sum. The result includes the current item, and every output
/// slot is defined. Block algorithm/storage choices and scalar Warp restrictions
/// are the same as for [`scan`].
#[inline(always)]
pub fn inclusive_scan<
    T: Numeric,
    V: ScanValue<T, G>,
    G: ScanGroup<A, S>,
    Op: ScanOperator<T>,
    A: ScanAlgorithm,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: InclusiveScanOptions<T, Op, A, S>,
) -> V::Output {
    scan(group, value, options.mode(Inclusive))
}

/// A sealed Scan input with its inferred fresh output type.
///
/// Numeric scalars return T. Shared/mutable payload borrows are supported only
/// for Block and return `ThreadData<T, ITEMS, ALIGN>` without changing the input.
/// Warp payload inputs and owned-payload-consuming forms are not provided.
///
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(values: &coop::ThreadData<i32, 4>) {
///     coop::scan(coop::this_warp(), values, Default::default());
/// }
/// ```
pub trait ScanValue<T: Numeric, G: GroupScope>: sealed::Value<T, G> {
    /// Scalar T or a fresh payload with the input's dtype, extent and alignment.
    type Output;
    #[doc(hidden)]
    fn scan<
        M: ScanMode<T, Op, Seed>,
        Op: ScanOperator<T>,
        A: ScanAlgorithm,
        Seed,
        S: TempStorageLike,
    >(
        self,
        group: ThreadGroup<G>,
        options: ScanOptions<T, M, Op, A, Seed, S>,
    ) -> Self::Output
    where
        G: ScanGroup<A, S>;
}
impl<T: Numeric, G: GroupScope> sealed::Value<T, G> for T {}
impl<T: Numeric, G: GroupScope> ScanValue<T, G> for T {
    type Output = T;
    #[inline(always)]
    fn scan<
        M: ScanMode<T, Op, Seed>,
        Op: ScanOperator<T>,
        A: ScanAlgorithm,
        Seed,
        S: TempStorageLike,
    >(
        self,
        _group: ThreadGroup<G>,
        options: ScanOptions<T, M, Op, A, Seed, S>,
    ) -> T
    where
        G: ScanGroup<A, S>,
    {
        let mut input = ThreadData::<_, 1>::new();
        input.write(0, self);
        let output = boundary::scan::<T, 1, 1, G, M, Op, A, Seed, S>(&input, options);
        let (pointer, _) = output.block_io_read_parts();
        // SAFETY: the synchronous selected recipe initializes every output before returning.
        unsafe { pointer.read() }
    }
}
impl<T: Numeric, const I: usize, const ALIGN: usize> sealed::Value<T, Block>
    for &ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}
impl<T: Numeric, const I: usize, const ALIGN: usize> ScanValue<T, Block>
    for &ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    type Output = ThreadData<T, I, ALIGN>;
    #[inline(always)]
    fn scan<
        M: ScanMode<T, Op, Seed>,
        Op: ScanOperator<T>,
        A: ScanAlgorithm,
        Seed,
        S: TempStorageLike,
    >(
        self,
        _group: ThreadGroup<Block>,
        options: ScanOptions<T, M, Op, A, Seed, S>,
    ) -> Self::Output {
        boundary::scan::<T, I, ALIGN, Block, M, Op, A, Seed, S>(self, options)
    }
}
impl<T: Numeric, const I: usize, const ALIGN: usize> sealed::Value<T, Block>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}
impl<T: Numeric, const I: usize, const ALIGN: usize> ScanValue<T, Block>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    type Output = ThreadData<T, I, ALIGN>;
    #[inline(always)]
    fn scan<
        M: ScanMode<T, Op, Seed>,
        Op: ScanOperator<T>,
        A: ScanAlgorithm,
        Seed,
        S: TempStorageLike,
    >(
        self,
        group: ThreadGroup<Block>,
        options: ScanOptions<T, M, Op, A, Seed, S>,
    ) -> Self::Output {
        (&*self).scan(group, options)
    }
}

mod sealed {
    pub trait Operator {
        const KIND: u8;
    }
    pub trait Algorithm {
        const KIND: u8;
    }
    pub trait Mode<T, Op, Seed> {
        type Uniform: Copy;
        const KIND: u8;
        fn controls(seed: Seed) -> Self::Uniform;
    }
    pub trait Value<T, G> {}
    pub trait Group {
        const WARP: bool;
        unsafe fn call<
            T: Copy,
            R: Fn() -> &'static str + Copy,
            S,
            U: Copy,
            P: Copy,
            const I: usize,
        >(
            registration: R,
            storage: S,
            uniform: U,
            payload: P,
        );
    }
}
