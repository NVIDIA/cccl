// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Ascending, Block, Descending, GroupScope, Numeric, SupportedAlignment, TempStorage,
    TempStorageLike, ThreadData, ThreadGroup, ThreadsWithinWarp, Warp,
};

use cuda_device::library;

mod boundary;

/// A sealed Merge Sort order or stateless i32 key comparator.
///
/// [`Ascending`] and [`Descending`] compare keys in their original numeric type.
/// Full-input Block i32 Keys also accepts function items and zero-capture
/// `Fn(i32, i32) -> bool + Copy` closures. True means the left key precedes the
/// right; the comparator must define a strict weak ordering. Merge Sort does
/// not guarantee the relative order of equal keys.
pub trait MergeSortOrder: sealed::Order {}

/// Sealed associated-value types for [`merge_sort_pairs`].
///
/// All ten [`Numeric`] types are supported independently of the key type.
/// Values follow their keys unchanged, including floating-point bit patterns;
/// they do not participate in comparisons.
pub trait MergeSortValue: Numeric + sealed::Value {}

/// A sealed full or partial tile selection for [`MergeSortOptions`].
///
/// Omitted `()` selects the full tile. The [`MergeSortOptions::valid_items`]
/// builder selects `(i64, T)`, keeping a uniform count and key-typed sentinel
/// together. This type bound alone does not prove their uniformity.
pub trait MergeSortPrefix<T: Numeric>: sealed::Prefix<T> {}

/// A sealed Merge Sort group and storage combination.
///
/// Complete power-of-two Blocks accept implicit or explicit storage. Physical
/// [`Warp`] and logical widths 1/2/4/8/16/32 accept implicit storage only and
/// require complete physical warps in the enclosing Block.
///
/// ```compile_fail,E0308
/// use cuda_coop as coop;
/// fn invalid(keys: &coop::ThreadData<i32, 4>, scratch: &mut coop::TempStorage) {
///     coop::merge_sort_keys(coop::this_warp(), keys,
///         coop::MergeSortOptions::default().temp_storage(scratch));
/// }
/// ```
/// ```compile_fail,E0277
/// use cuda_coop as coop;
/// fn invalid(keys: &coop::ThreadData<i32, 4>) {
///     let group = coop::this_warp().group_by_non_exhaustive::<17>();
///     coop::merge_sort_keys(group, keys, Default::default());
/// }
/// ```
pub trait MergeSortGroup<S: TempStorageLike>: GroupScope + sealed::Group {}
impl<S: TempStorageLike> MergeSortGroup<S> for Block {}
impl MergeSortGroup<()> for Warp {}
impl sealed::Group for Block {
    const WARP: bool = false;
    #[inline(always)]
    unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, S, U: Copy, P: Copy, const N: usize>(
        registration: R,
        storage: S,
        uniform: U,
        payload: P,
    ) {
        // SAFETY: the private boundary supplies live typed payloads and the exact recipe.
        unsafe {
            library::checked_block_payload_call_v1::<T, _, _, _, _, N>(
                registration,
                storage,
                uniform,
                payload,
            )
        }
    }
}
macro_rules! warp_group {
    ($scope:ty, $width:literal $(, $e:ident)?) => {
        impl<$(const $e: bool)?> sealed::Group for $scope {
            const WARP: bool = true;
            #[inline(always)]
            unsafe fn call<T: Copy, R: Fn() -> &'static str + Copy, S, U: Copy, P: Copy, const N: usize>(
                registration: R, _storage: S, uniform: U, payload: P,
            ) {
                // SAFETY: MergeSortGroup permits only implicit Warp storage;
                // the genuine marker checks participation and the exact width.
                unsafe { library::checked_warp_payload_call_v1::<T, _, _, _, $width, N>(registration, uniform, payload) }
            }
        }
    };
}
warp_group!(Warp, 32);
macro_rules! logical_groups {
    ($($width:literal),+) => {$ (
        impl<const E: bool> MergeSortGroup<()> for ThreadsWithinWarp<$width, E> {}
        warp_group!(ThreadsWithinWarp<$width, E>, $width, E);
    )+};
}
logical_groups!(1, 2, 4, 8, 16, 32);

/// Order and optional count/sentinel for [`merge_sort_keys`] or [`merge_sort_pairs`].
///
/// `T` is inferred from the input. Only canonical [`Default`] is implemented:
/// ascending order, every input and implicit scratch. Builders consume options
/// and preserve the other selections, including the actual order value and storage.
///
/// ```
/// use cuda_coop as coop;
/// let _: coop::MergeSortOptions<i64> = Default::default();
/// let _ = coop::MergeSortOptions::<i64>::default()
///     .order(coop::Descending).valid_items(7, i64::MIN);
/// let _ = coop::MergeSortOptions::<i64>::default()
///     .valid_items(7, i64::MIN).order(coop::Descending);
/// ```
pub struct MergeSortOptions<T, O = Ascending, Prefix = (), S = ()> {
    order: O,
    prefix: Prefix,
    storage: S,
    selection: PhantomData<T>,
}

impl<T: Numeric> Default for MergeSortOptions<T> {
    fn default() -> Self {
        Self {
            order: Ascending,
            prefix: (),
            storage: (),
            selection: PhantomData,
        }
    }
}

impl<T: Numeric, O, P, S> MergeSortOptions<T, O, P, S> {
    /// Select a built-in order or stateless i32 Keys comparator, retaining other options.
    ///
    /// Closures that call methods on their keys may need explicit `i32` parameter types.
    pub fn order<O2: MergeSortOrder>(self, order: O2) -> MergeSortOptions<T, O2, P, S> {
        MergeSortOptions {
            order,
            prefix: self.prefix,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Sort this blocked prefix using a sentinel of exactly the key type.
    ///
    /// Count and sentinel must be uniform across the selected group. Count
    /// must be in `0..=group_size * ITEMS_PER_THREAD`; it is checked as signed
    /// `i64` before conversion to CUB's int. Widen smaller integer counts
    /// losslessly. The sentinel must be ordered after every valid key in the
    /// selected order. A literal sentinel is inferred from the input key type.
    ///
    /// Every input slot must still be initialized, including at count zero.
    /// The result has initialized slots only within this prefix. Calling this
    /// builder again replaces both count and sentinel.
    ///
    /// ```compile_fail,E0308
    /// use cuda_coop as coop;
    /// let _ = coop::MergeSortOptions::<i64>::default().valid_items(7, 0_i32);
    /// ```
    pub fn valid_items(self, count: i64, sentinel: T) -> MergeSortOptions<T, O, (i64, T), S> {
        MergeSortOptions {
            order: self.order,
            prefix: (count, sentinel),
            storage: self.storage,
            selection: PhantomData,
        }
    }
    /// Borrow existing Block scratch, retaining order and paired controls.
    ///
    /// Manual scratch requires Block completion before conflicting reuse.
    /// Auto-sync storage inserts its existing trailing barrier. Warp rejects
    /// this option; consumed options permit later ordinary mutable reborrows.
    pub fn temp_storage<const C: usize, const A: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, A, AUTO, EX>,
    ) -> MergeSortOptions<T, O, P, &mut TempStorage<C, A, AUTO, EX>> {
        MergeSortOptions {
            order: self.order,
            prefix: self.prefix,
            storage,
            selection: PhantomData,
        }
    }
}

/// Merge-sort a group's keys into a fresh blocked payload.
///
/// Supports all ten [`Numeric`] types and a positive `ITEMS_PER_THREAD`. The
/// result preserves the input type, extent and minimum alignment. Every input
/// slot must be initialized, even outside a selected prefix or at count zero;
/// input values and validity bits remain unchanged through shared borrows.
///
/// All members of the selected group participate. Complete XYZ Blocks require a
/// power-of-two total; Warp supports physical and logical widths described by
/// [`MergeSortGroup`]. Input/output index is `rank * ITEMS_PER_THREAD + item`,
/// using x-fastest Block rank or rank within each Warp group. The provider checks
/// tile capacity and CUB's
/// padded scratch extent before converting dimensions to int.
///
/// Default options sort the full tile ascending and initialize every result.
/// A partial count and typed sentinel are selected together with
/// [`MergeSortOptions::valid_items`]. Only the result's blocked prefix is then
/// initialized; [`ThreadData::get`] returns `None` for the suffix. Count zero
/// still participates, validates the full input and returns an empty result.
/// Invalid runtime counts or unreadable inputs trap before native numeric reads.
///
/// Equal keys have no stable-tie guarantee. Floating keys must obey the
/// comparison's strict weak ordering requirements; no NaN total order is
/// supplied. The sentinel must follow valid keys in the chosen order.
///
/// Implicit scratch is compiler-planned, with synchronization before dirty reuse.
/// Explicit Block storage retains its manual/auto-sync/exclusive policy. CUB
/// synchronization does not complete unrelated global accesses; use
/// `cuda_device::thread::sync_threads()` when needed. Full-input Block i32 Keys
/// also accepts the stateless comparators described by [`MergeSortOrder`].
/// Execution requires an
/// Oxide-compiled CUDA kernel. Generic T/N/ALIGN callers need no
/// `generic_const_exprs` feature.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn sorted<T: coop::Numeric, const N: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     keys: &coop::ThreadData<T, N, ALIGN>,
/// ) -> coop::ThreadData<T, N, ALIGN>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::merge_sort_keys(block, keys, Default::default())
/// }
/// ```
#[inline(always)]
pub fn merge_sort_keys<T, G, O, P, S, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<G>,
    keys: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: MergeSortOptions<T, O, P, S>,
) -> ThreadData<T, ITEMS_PER_THREAD, ALIGN>
where
    T: Numeric,
    O: MergeSortOrder,
    P: MergeSortPrefix<T>,
    S: TempStorageLike,
    G: MergeSortGroup<S>,
    Alignment<ALIGN>: SupportedAlignment,
{
    O::keys::<T, G, P, S, ITEMS_PER_THREAD, ALIGN>(keys, options)
}

/// Merge-sort associated keys and values into two fresh blocked payloads.
///
/// Keys support all ten [`Numeric`] types, and values independently support
/// all ten [`MergeSortValue`] types. Both inputs share `ITEMS_PER_THREAD`;
/// their minimum alignments `KEY_ALIGN` and `VALUE_ALIGN` are independent.
/// Each output keeps its corresponding input's type, extent and alignment.
/// Both borrowed inputs and their validity maps remain unchanged.
///
/// The group, built-in order, count/sentinel, storage and completion rules
/// are the same as [`merge_sort_keys`]. Default options sort the complete tile
/// ascending with implicit scratch. Partial options take a uniform signed64
/// count and a sentinel of the key type. ALL keys and values must be initialized,
/// even beyond the prefix or at count zero. Invalid counts or either missing
/// input bit trap before numeric reads. Both outputs have the same valid
/// prefix; suffix [`ThreadData::get`] calls return `None`.
/// Callable comparator orders are rejected at compile time for Pairs.
///
/// Each value follows its associated key without numeric conversion. Equal
/// keys have no stable-tie guarantee. Floating keys must satisfy strict weak
/// ordering; no NaN total order is supplied. Values are not compared and may
/// contain any numeric bit pattern. Scratch is queried for the pair primitive,
/// including its padded value storage, and cannot be inferred from key size.
/// Execution requires an Oxide-compiled CUDA kernel; no `generic_const_exprs`
/// feature is needed for generic K/V/N/alignment callers.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn sorted<K: coop::Numeric, V: coop::MergeSortValue,
///           const N: usize, const KA: usize, const VA: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     keys: &coop::ThreadData<K, N, KA>, values: &coop::ThreadData<V, N, VA>,
/// ) -> (coop::ThreadData<K, N, KA>, coop::ThreadData<V, N, VA>)
/// where coop::Alignment<KA>: coop::SupportedAlignment,
///       coop::Alignment<VA>: coop::SupportedAlignment {
///     coop::merge_sort_pairs(block, keys, values, Default::default())
/// }
/// ```
#[inline(always)]
pub fn merge_sort_pairs<
    K,
    V,
    G,
    O,
    P,
    S,
    const ITEMS_PER_THREAD: usize,
    const KEY_ALIGN: usize,
    const VALUE_ALIGN: usize,
>(
    _group: ThreadGroup<G>,
    keys: &ThreadData<K, ITEMS_PER_THREAD, KEY_ALIGN>,
    values: &ThreadData<V, ITEMS_PER_THREAD, VALUE_ALIGN>,
    options: MergeSortOptions<K, O, P, S>,
) -> (
    ThreadData<K, ITEMS_PER_THREAD, KEY_ALIGN>,
    ThreadData<V, ITEMS_PER_THREAD, VALUE_ALIGN>,
)
where
    K: Numeric,
    V: MergeSortValue,
    O: MergeSortOrder,
    P: MergeSortPrefix<K>,
    S: TempStorageLike,
    G: MergeSortGroup<S>,
    Alignment<KEY_ALIGN>: SupportedAlignment,
    Alignment<VALUE_ALIGN>: SupportedAlignment,
{
    boundary::merge_sort_pairs::<K, V, G, O, P, S, ITEMS_PER_THREAD, KEY_ALIGN, VALUE_ALIGN>(
        keys, values, options,
    )
}

mod sealed {
    use super::*;
    pub trait Value {
        const ASCENDING_FULL_REGISTRATION: &'static str;
        const DESCENDING_FULL_REGISTRATION: &'static str;
        const ASCENDING_PARTIAL_REGISTRATION: &'static str;
        const DESCENDING_PARTIAL_REGISTRATION: &'static str;
        const WARP_ASCENDING_FULL_REGISTRATION: &'static str;
        const WARP_DESCENDING_FULL_REGISTRATION: &'static str;
        const WARP_ASCENDING_PARTIAL_REGISTRATION: &'static str;
        const WARP_DESCENDING_PARTIAL_REGISTRATION: &'static str;
    }
    pub trait Group {
        const WARP: bool;
        unsafe fn call<
            T: Copy,
            R: Fn() -> &'static str + Copy,
            S,
            U: Copy,
            P: Copy,
            const N: usize,
        >(
            registration: R,
            storage: S,
            uniform: U,
            payload: P,
        );
    }
    pub trait Order {
        const DESCENDING: bool;
        fn keys<T, G, P, S, const N: usize, const ALIGN: usize>(
            input: &ThreadData<T, N, ALIGN>,
            options: MergeSortOptions<T, Self, P, S>,
        ) -> ThreadData<T, N, ALIGN>
        where
            Self: Sized,
            T: Numeric,
            P: MergeSortPrefix<T>,
            S: TempStorageLike,
            G: MergeSortGroup<S>,
            Alignment<ALIGN>: SupportedAlignment;
    }
    pub trait Prefix<T>: Copy {
        const PARTIAL: bool;
    }
}

impl sealed::Order for Ascending {
    const DESCENDING: bool = false;
    fn keys<T, G, P, S, const N: usize, const ALIGN: usize>(
        input: &ThreadData<T, N, ALIGN>,
        options: MergeSortOptions<T, Self, P, S>,
    ) -> ThreadData<T, N, ALIGN>
    where
        T: Numeric,
        P: MergeSortPrefix<T>,
        S: TempStorageLike,
        G: MergeSortGroup<S>,
        Alignment<ALIGN>: SupportedAlignment,
    {
        boundary::merge_sort_keys::<T, G, Self, P, S, N, ALIGN>(input, options)
    }
}
impl MergeSortOrder for Ascending {}
impl sealed::Order for Descending {
    const DESCENDING: bool = true;
    fn keys<T, G, P, S, const N: usize, const ALIGN: usize>(
        input: &ThreadData<T, N, ALIGN>,
        options: MergeSortOptions<T, Self, P, S>,
    ) -> ThreadData<T, N, ALIGN>
    where
        T: Numeric,
        P: MergeSortPrefix<T>,
        S: TempStorageLike,
        G: MergeSortGroup<S>,
        Alignment<ALIGN>: SupportedAlignment,
    {
        boundary::merge_sort_keys::<T, G, Self, P, S, N, ALIGN>(input, options)
    }
}
impl MergeSortOrder for Descending {}

struct CallableBounds<T, G, P, F>(PhantomData<(T, G, P, F)>);
impl<T, G: sealed::Group, P: sealed::Prefix<T>, F> CallableBounds<T, G, P, F> {
    const VALID: () = {
        assert!(!G::WARP, "callable Merge Sort requires Block");
        assert!(!P::PARTIAL, "callable Merge Sort requires full input");
        assert!(
            core::mem::size_of::<F>() == 0,
            "callable Merge Sort requires a stateless operator"
        );
    };
}
impl<F: Fn(i32, i32) -> bool + Copy> sealed::Order for F {
    const DESCENDING: bool = panic!("callable Merge Sort Pairs is unsupported");
    fn keys<T, G, P, S, const N: usize, const ALIGN: usize>(
        input: &ThreadData<T, N, ALIGN>,
        options: MergeSortOptions<T, Self, P, S>,
    ) -> ThreadData<T, N, ALIGN>
    where
        T: Numeric,
        P: MergeSortPrefix<T>,
        S: TempStorageLike,
        G: MergeSortGroup<S>,
        Alignment<ALIGN>: SupportedAlignment,
    {
        let () = CallableBounds::<T, G, P, F>::VALID;
        boundary::callable(input, options.order, options.storage)
    }
}
impl<F: Fn(i32, i32) -> bool + Copy> MergeSortOrder for F {}
impl<T: Numeric> sealed::Prefix<T> for () {
    const PARTIAL: bool = false;
}
impl<T: Numeric> MergeSortPrefix<T> for () {}
impl<T: Numeric> sealed::Prefix<T> for (i64, T) {
    const PARTIAL: bool = true;
}
impl<T: Numeric> MergeSortPrefix<T> for (i64, T) {}

macro_rules! pair_values {
    ($(($ty:ty, $ascending_full:literal, $descending_full:literal, $ascending_partial:literal, $descending_partial:literal, $warp_ascending_full:literal, $warp_descending_full:literal, $warp_ascending_partial:literal, $warp_descending_partial:literal)), + $(,)?) => {$ (
        impl sealed::Value for $ty {
            const ASCENDING_FULL_REGISTRATION: &'static str = library::registration!($ascending_full);
            const DESCENDING_FULL_REGISTRATION: &'static str = library::registration!($descending_full);
            const ASCENDING_PARTIAL_REGISTRATION: &'static str = library::registration!($ascending_partial);
            const DESCENDING_PARTIAL_REGISTRATION: &'static str = library::registration!($descending_partial);
            const WARP_ASCENDING_FULL_REGISTRATION: &'static str = library::registration!($warp_ascending_full);
            const WARP_DESCENDING_FULL_REGISTRATION: &'static str = library::registration!($warp_descending_full);
            const WARP_ASCENDING_PARTIAL_REGISTRATION: &'static str = library::registration!($warp_ascending_partial);
            const WARP_DESCENDING_PARTIAL_REGISTRATION: &'static str = library::registration!($warp_descending_partial);
        }
        impl MergeSortValue for $ty {}
    )+};
}
pair_values!(
    (
        i8,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_I8",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_I8",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_I8",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_I8",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_I8",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_I8",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_I8",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_I8"
    ),
    (
        u8,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_U8",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_U8",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_U8",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_U8",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_U8",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_U8",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_U8",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_U8"
    ),
    (
        i16,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_I16",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_I16",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_I16",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_I16",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_I16",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_I16",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_I16",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_I16"
    ),
    (
        u16,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_U16",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_U16",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_U16",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_U16",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_U16",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_U16",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_U16",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_U16"
    ),
    (
        i32,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_I32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_I32",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_I32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_I32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_I32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_I32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_I32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_I32"
    ),
    (
        u32,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_U32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_U32",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_U32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_U32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_U32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_U32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_U32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_U32"
    ),
    (
        i64,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_I64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_I64",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_I64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_I64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_I64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_I64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_I64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_I64"
    ),
    (
        u64,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_U64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_U64",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_U64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_U64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_U64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_U64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_U64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_U64"
    ),
    (
        f32,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_F32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_F32",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_F32",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_F32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_F32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_F32",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_F32",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_F32"
    ),
    (
        f64,
        "COOP_MERGE_SORT_PAIRS_ASCENDING_FULL_F64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_FULL_F64",
        "COOP_MERGE_SORT_PAIRS_ASCENDING_PARTIAL_F64",
        "COOP_MERGE_SORT_PAIRS_DESCENDING_PARTIAL_F64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_FULL_F64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_FULL_F64",
        "COOP_MERGE_SORT_PAIRS_WARP_ASCENDING_PARTIAL_F64",
        "COOP_MERGE_SORT_PAIRS_WARP_DESCENDING_PARTIAL_F64"
    ),
);
