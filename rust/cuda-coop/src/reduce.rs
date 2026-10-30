// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, BitAnd, BitOr, BitXor, Block, GroupScope, Max, Min, Multiplies, Numeric, Raking,
    Sum, SupportedAlignment, TempStorage, TempStorageLike, ThreadData, ThreadGroup,
    ThreadsWithinWarp, Warp,
};
use cuda_device::library;

mod boundary;

macro_rules! operators {
    ($(($name:ty, $kind:literal)),+ $(,)?) => {$ (
        impl sealed::Operator for $name { const KIND: u8 = $kind; }
        impl<T: Numeric> sealed::Dispatch<T> for $name {
            #[inline(always)]
            fn reduce<
                const ITEMS: usize, const ALIGN: usize, G: ReduceGroup<A, S>,
                A: ReduceAlgorithm, C: sealed::Count, S: TempStorageLike,
                const PAYLOAD: bool,
            >(
                input: &ThreadData<T, ITEMS, ALIGN>,
                options: ReduceOptions<T, Self, A, C, S>,
            ) -> Option<T>
            where Alignment<ALIGN>: SupportedAlignment {
                boundary::reduce::<T, ITEMS, ALIGN, G, Self, A, C, S, PAYLOAD>(input, options)
            }
        }
    )+};
}
operators!(
    (Sum, 0),
    (Multiplies, 1),
    (Min, 2),
    (Max, 3),
    (BitAnd, 4),
    (BitOr, 5),
    (BitXor, 6)
);

/// A sealed static Reduce operator for numeric input type `T`.
///
/// [`Sum`], [`Multiplies`], [`Min`] and [`Max`] accept all ten [`Numeric`] types.
/// Bitwise operators accept only the eight integer types. Generic helpers may
/// carry this bound and pass the actual operator value. These built-in
/// operators permit the commutative algorithm.
///
/// Function items and closures implementing `Fn(i32, i32) -> i32 + Copy`
/// support full-input Block reduction with explicit [`Raking`]. The operator
/// must be associative. A closure may capture one `i32` by value when the compiler
/// proves that its value is identical for every thread in the block at the call.
/// Other captures, function pointers, other input types, other algorithms, Warp
/// groups and partial counts are not supported.
pub trait ReduceOperator<T: Numeric>: sealed::Operator + sealed::Dispatch<T> {}
impl<T: Numeric> ReduceOperator<T> for Sum {}
impl<T: Numeric> ReduceOperator<T> for Multiplies {}
impl<T: Numeric> ReduceOperator<T> for Min {}
impl<T: Numeric> ReduceOperator<T> for Max {}
macro_rules! integer_operators {
    ($($t:ty),+ $(,)?) => {$ (
        impl ReduceOperator<$t> for BitAnd {}
        impl ReduceOperator<$t> for BitOr {}
        impl ReduceOperator<$t> for BitXor {}
    )+};
}
integer_operators!(i8, u8, i16, u16, i32, u32, i64, u64);

impl<F: Fn(i32, i32) -> i32 + Copy> ReduceOperator<i32> for F {}
impl<F: Fn(i32, i32) -> i32 + Copy> sealed::Operator for F {
    const KIND: u8 = panic!("callable Reduce does not support batched reduction");
}

struct CallableBounds<G, A, C>(PhantomData<(G, A, C)>);
impl<G: sealed::Group, A: sealed::Algorithm, C: sealed::Count> CallableBounds<G, A, C> {
    const VALID: () = {
        assert!(!G::WARP, "callable Reduce requires Block");
        assert!(A::KIND == 2, "callable Reduce requires explicit Raking");
        assert!(C::KIND == 0, "callable Reduce requires full input");
    };
}
impl<F: Fn(i32, i32) -> i32 + Copy> sealed::Dispatch<i32> for F {
    #[inline(always)]
    fn reduce<
        const ITEMS: usize,
        const ALIGN: usize,
        G: ReduceGroup<A, S>,
        A: ReduceAlgorithm,
        C: sealed::Count,
        S: TempStorageLike,
        const PAYLOAD: bool,
    >(
        input: &ThreadData<i32, ITEMS, ALIGN>,
        options: ReduceOptions<i32, Self, A, C, S>,
    ) -> Option<i32>
    where
        Alignment<ALIGN>: SupportedAlignment,
    {
        let () = CallableBounds::<G, A, C>::VALID;
        boundary::callable(input, options.binary_op, options.storage)
    }
}

/// CUB's raking Block reduction specialized for commutative operators.
#[derive(Clone, Copy, Debug, Default)]
pub struct RakingCommutativeOnly;
/// CUB's warp-based Block reduction; selected when the algorithm is omitted.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarpReductions;

/// A sealed Block reduction algorithm or omitted selection `()`.
///
/// Block resolves omission to [`WarpReductions`]. Explicit [`Raking`],
/// [`RakingCommutativeOnly`] and [`WarpReductions`] are Block-only. Warp accepts
/// only the omitted selection, using its own CUB Warp reduction.
pub trait ReduceAlgorithm: sealed::Algorithm {}
macro_rules! algorithms {
    ($(($name:ty, $kind:literal)),+ $(,)?) => {$ (
        impl sealed::Algorithm for $name { const KIND: u8 = $kind; }
        impl ReduceAlgorithm for $name {}
    )+};
}
algorithms!(
    ((), 0),
    (RakingCommutativeOnly, 1),
    (Raking, 2),
    (WarpReductions, 3)
);

/// A sealed group supporting a Reduce algorithm and storage choice.
///
/// Complete Blocks support scalar or payload input with implicit or explicit
/// storage. Physical [`Warp`] and logical widths 1/2/4/8/16/32 support both
/// input forms with omitted algorithm and implicit storage only. Non-exhaustive
/// logical widths 17 through 31 support the low-width prefix of each physical
/// warp, inside a genuine [`ThreadGroup::is_member`] guard. Warp operations
/// require complete physical warps in the enclosing block.
///
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(value: i32) {
///     coop::sum(coop::this_warp(), value,
///         coop::ReduceOptions::default().algorithm(coop::Raking));
/// }
/// ```
pub trait ReduceGroup<A: ReduceAlgorithm, S: TempStorageLike>: GroupScope + sealed::Group {}
impl<A: ReduceAlgorithm, S: TempStorageLike> ReduceGroup<A, S> for Block {}
impl ReduceGroup<(), ()> for Warp {}
impl sealed::Group for Block {
    const WARP: bool = false;
    #[inline(always)]
    fn count() -> usize {
        crate::this_block().count() as usize
    }
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
            #[inline(always)] fn count() -> usize { $width }
            #[inline(always)]
            unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, S, U: Copy, P: Copy, const I: usize>(
                registration: R, _storage: S, uniform: U, payload: P,
            ) {
                // SAFETY: ReduceGroup permits implicit storage only; the compiler checks membership and width.
                unsafe { library::checked_warp_payload_call_v1::<T, _, _, _, $width, I>(registration, uniform, payload) }
            }
        }
    };
}
warp_group!(Warp, 32);
macro_rules! logical_groups {
    ($($width:literal),+) => {$ (
        impl<const E: bool> ReduceGroup<(), ()> for ThreadsWithinWarp<$width, E> {}
        warp_group!(ThreadsWithinWarp<$width, E>, $width, E);
    )+};
}
logical_groups!(1, 2, 4, 8, 16, 32);
macro_rules! prefix_groups {
    ($($width:literal),+) => {$ (
        impl ReduceGroup<(), ()> for ThreadsWithinWarp<$width, false> {}
        warp_group!(ThreadsWithinWarp<$width, false>, $width);
    )+};
}
prefix_groups!(17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

/// Static operator/algorithm and runtime count/storage for [`reduce`] and [`sum`].
///
/// `T` is inferred from the input. Defaults select [`Sum`], omitted algorithm
/// (Block [`WarpReductions`]), every input and managed storage. Only this
/// canonical specialization implements Default, so `Default::default()` is
/// unambiguous. Builders consume options and preserve their other fields.
///
/// A valid count is supported only for scalar input, including when a payload
/// happens to contain one item. Warp rejects explicit algorithm and storage.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn total<T: coop::Numeric, Op: coop::ReduceOperator<T>, const ITEMS_PER_THREAD: usize>(
///     values: &coop::ThreadData<T, ITEMS_PER_THREAD>, op: Op,
/// ) -> Option<T> {
///     coop::reduce(coop::this_block(), values,
///         coop::ReduceOptions::default().binary_op(op))
/// }
/// ```
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(values: &coop::ThreadData<i32, 1>) {
///     coop::sum(coop::this_block(), values,
///         coop::ReduceOptions::default().valid_items(1));
/// }
/// ```
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(value: f32) {
///     coop::reduce(coop::this_block(), value,
///         coop::ReduceOptions::default().binary_op(coop::BitAnd));
/// }
/// ```
pub struct ReduceOptions<T, Op = Sum, A = (), Count = (), S = ()> {
    binary_op: Op,
    valid_items: Count,
    storage: S,
    selection: PhantomData<(T, A)>,
}
impl<T: Numeric> Default for ReduceOptions<T> {
    fn default() -> Self {
        Self {
            binary_op: Sum,
            valid_items: (),
            storage: (),
            selection: PhantomData,
        }
    }
}
impl<T: Numeric, Op, A, Count, S> ReduceOptions<T, Op, A, Count, S> {
    /// Select an operator, retaining its actual generic value.
    ///
    /// `i32` callables require Block, explicit [`Raking`] and full input.
    /// See [`ReduceOperator`] for supported captures and their uniformity rule.
    pub fn binary_op<Op2: ReduceOperator<T>>(self, op: Op2) -> ReduceOptions<T, Op2, A, Count, S> {
        ReduceOptions {
            binary_op: op,
            valid_items: self.valid_items,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Select a Block algorithm. Warp accepts only the omitted selection `()`.
    pub fn algorithm<A2: ReduceAlgorithm>(
        self,
        _algorithm: A2,
    ) -> ReduceOptions<T, Op, A2, Count, S> {
        ReduceOptions {
            binary_op: self.binary_op,
            valid_items: self.valid_items,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Reduce the first `count` scalar inputs in group rank order.
    ///
    /// Requires `1 <= count <= group.count()` and a group-uniform count. All
    /// members still participate; excluded scalar inputs do not contribute.
    /// Payload inputs reject this option, even for a one-item payload.
    pub fn valid_items(self, count: usize) -> ReduceOptions<T, Op, A, usize, S> {
        ReduceOptions {
            binary_op: self.binary_op,
            valid_items: count,
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Borrow explicit Block scratch with its existing reuse/completion policy.
    ///
    /// Warp rejects this choice. Consuming the options permits ordinary mutable
    /// reborrows for later operations, without requiring storage to be Copy.
    pub fn temp_storage<const C: usize, const L: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, L, AUTO, EX>,
    ) -> ReduceOptions<T, Op, A, Count, &mut TempStorage<C, L, AUTO, EX>> {
        ReduceOptions {
            binary_op: self.binary_op,
            valid_items: self.valid_items,
            storage,
            selection: PhantomData,
        }
    }
}

/// Algorithm, scalar count and storage options for [`sum`]; fixes the operator to [`Sum`].
pub type SumOptions<T, A = (), Count = (), S = ()> = ReduceOptions<T, Sum, A, Count, S>;

/// Reduce numeric values, returning `Some(total)` only at group rank zero.
///
/// Scalars and shared/mutable payload borrows all return `Option<T>`. Every
/// other participating thread gets `None`; no undefined numeric slot is read,
/// broadcast or replaced by an implicit identity. Input values and validity
/// are preserved. Block order is x-fastest thread rank followed by each
/// thread's item order. Warp uses the corresponding order within each group.
///
/// Defaults select sum, all inputs and Block [`WarpReductions`]. Scalar-only
/// [`valid_items`](ReduceOptions::valid_items) restricts contributors to a
/// nonempty prefix; every member must still reach the operation with the same
/// count. Payload reduction requires every slot initialized. A tail must be
/// filled explicitly before reduction, for example by a default-filled Load.
/// CUB may regroup associative arithmetic; overflow and floating behavior
/// follow the input dtype and primitive, not a sequential host-order guarantee.
///
/// All members must participate. [`ReduceGroup`] describes supported groups;
/// prefix widths 17–31 require a genuine membership guard. A result's `Some`
/// branch grants no collective participation or global-memory ownership.
/// For a per-block output, obtain a checked [`cuda_device::thread::BlockIndex`]
/// witness inside the terminal result branch and use a [`crate::DisjointSlice`].
///
/// Managed scratch completes before dirty reuse, without a trailing completion
/// promise. Explicit manual scratch needs completion before conflicting reuse;
/// auto-sync storage adds a trailing Block barrier. The operation alone is not
/// completion of prior global-memory accesses. Execution requires pinned Oxide.
///
/// # Panics
/// An uninitialized required input or a scalar count outside `1..=group.count()`
/// traps before reduction. Invalid participation, options, geometry or scratch
/// reuse fail compilation.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn total<T: coop::Numeric, const ITEMS_PER_THREAD: usize>(
///     values: &coop::ThreadData<T, ITEMS_PER_THREAD>,
/// ) -> Option<T> {
///     coop::reduce(coop::this_block(), values, Default::default())
/// }
/// ```
#[inline(always)]
pub fn reduce<
    T: Numeric,
    V: ReduceValue<T, G, Count>,
    G: ReduceGroup<A, S>,
    Op: ReduceOperator<T>,
    A: ReduceAlgorithm,
    Count,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: ReduceOptions<T, Op, A, Count, S>,
) -> Option<T> {
    value.reduce(group, options)
}

/// Sum numeric values; root-only validity, input preservation and completion follow [`reduce`].
///
/// The operator is fixed to [`Sum`]. Options select Block algorithm/storage and
/// an optional scalar count. Default options infer `T` and payload extent.
#[inline(always)]
pub fn sum<
    T: Numeric,
    V: ReduceValue<T, G, Count>,
    G: ReduceGroup<A, S>,
    A: ReduceAlgorithm,
    Count,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    value: V,
    options: SumOptions<T, A, Count, S>,
) -> Option<T> {
    reduce(group, value, options)
}

/// A sealed scalar or borrowed payload input with an inferred root-only result.
///
/// Numeric scalars permit omitted or `usize` count. Shared/mutable
/// `ThreadData<T, ITEMS, ALIGN>` borrows permit only omitted count. Both input
/// forms support every [`ReduceGroup`], preserve input and return `Option<T>`.
/// Owned-payload-consuming forms are not provided.
pub trait ReduceValue<T: Numeric, G: GroupScope, Count>: sealed::Value<T, G, Count> {
    #[doc(hidden)]
    fn reduce<Op: ReduceOperator<T>, A: ReduceAlgorithm, S: TempStorageLike>(
        self,
        group: ThreadGroup<G>,
        options: ReduceOptions<T, Op, A, Count, S>,
    ) -> Option<T>
    where
        G: ReduceGroup<A, S>;
}
impl<T: Numeric, G: GroupScope, C: sealed::Count> sealed::Value<T, G, C> for T {}
impl<T: Numeric, G: GroupScope, C: sealed::Count> ReduceValue<T, G, C> for T {
    #[inline(always)]
    fn reduce<Op: ReduceOperator<T>, A: ReduceAlgorithm, S: TempStorageLike>(
        self,
        _group: ThreadGroup<G>,
        options: ReduceOptions<T, Op, A, C, S>,
    ) -> Option<T>
    where
        G: ReduceGroup<A, S>,
    {
        let mut input = ThreadData::<_, 1>::new();
        input.write(0, self);
        Op::reduce::<1, 1, G, A, C, S, false>(&input, options)
    }
}
impl<T: Numeric, G: GroupScope, const I: usize, const ALIGN: usize> sealed::Value<T, G, ()>
    for &ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}
impl<T: Numeric, G: GroupScope, const I: usize, const ALIGN: usize> ReduceValue<T, G, ()>
    for &ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline(always)]
    fn reduce<Op: ReduceOperator<T>, A: ReduceAlgorithm, S: TempStorageLike>(
        self,
        _group: ThreadGroup<G>,
        options: ReduceOptions<T, Op, A, (), S>,
    ) -> Option<T>
    where
        G: ReduceGroup<A, S>,
    {
        Op::reduce::<I, ALIGN, G, A, (), S, true>(self, options)
    }
}
impl<T: Numeric, G: GroupScope, const I: usize, const ALIGN: usize> sealed::Value<T, G, ()>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}
impl<T: Numeric, G: GroupScope, const I: usize, const ALIGN: usize> ReduceValue<T, G, ()>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline(always)]
    fn reduce<Op: ReduceOperator<T>, A: ReduceAlgorithm, S: TempStorageLike>(
        self,
        group: ThreadGroup<G>,
        options: ReduceOptions<T, Op, A, (), S>,
    ) -> Option<T>
    where
        G: ReduceGroup<A, S>,
    {
        (&*self).reduce(group, options)
    }
}

mod sealed {
    use super::*;

    pub trait Dispatch<T: Numeric>: Sized {
        fn reduce<
            const ITEMS: usize,
            const ALIGN: usize,
            G: ReduceGroup<A, S>,
            A: ReduceAlgorithm,
            C: Count,
            S: TempStorageLike,
            const PAYLOAD: bool,
        >(
            input: &ThreadData<T, ITEMS, ALIGN>,
            options: ReduceOptions<T, Self, A, C, S>,
        ) -> Option<T>
        where
            Alignment<ALIGN>: SupportedAlignment;
    }
    pub trait Operator {
        const KIND: u8;
    }
    pub trait Algorithm {
        const KIND: u8;
    }
    pub trait Value<T, G, Count> {}
    pub trait Count {
        type Uniform: Copy;
        const KIND: u8;
        fn controls<G: Group>(self) -> Self::Uniform;
    }
    impl Count for () {
        type Uniform = ();
        const KIND: u8 = 0;
        #[inline(always)]
        fn controls<G: Group>(self) {}
    }
    impl Count for usize {
        type Uniform = (i32,);
        const KIND: u8 = 1;
        #[inline(always)]
        fn controls<G: Group>(self) -> (i32,) {
            assert!(self > 0);
            assert!(self <= G::count());
            // CUDA group sizes are bounded by the maximum threads in a block.
            (self as i32,)
        }
    }
    pub trait Group {
        const WARP: bool;
        fn count() -> usize;
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
