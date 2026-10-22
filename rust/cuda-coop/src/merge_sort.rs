// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Ascending, Block, Descending, GroupScope, Numeric, SupportedAlignment, TempStorage,
    TempStorageLike, ThreadData, ThreadGroup, ThreadsWithinWarp, Warp,
};

use cuda_device::library;

mod boundary;

/// A sealed built-in Merge Sort order: [`Ascending`] or [`Descending`].
///
/// Both orders compare keys in their original numeric type. Merge Sort does
/// not guarantee the relative order of equal keys.
pub trait MergeSortOrder: sealed::Order {}

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

/// Built-in order and optional count/sentinel for [`merge_sort_keys`].
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
    /// Select ascending or descending order, preserving the count and sentinel.
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
/// `cuda_device::thread::sync_threads()` when needed. No custom comparator is
/// supplied. Execution requires an
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
    boundary::merge_sort_keys::<T, G, O, P, S, ITEMS_PER_THREAD, ALIGN>(keys, options)
}

mod sealed {
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
    }
    pub trait Prefix<T>: Copy {
        const PARTIAL: bool;
    }
}

impl sealed::Order for Ascending {
    const DESCENDING: bool = false;
}
impl MergeSortOrder for Ascending {}
impl sealed::Order for Descending {
    const DESCENDING: bool = true;
}
impl MergeSortOrder for Descending {}
impl<T: Numeric> sealed::Prefix<T> for () {
    const PARTIAL: bool = false;
}
impl<T: Numeric> MergeSortPrefix<T> for () {}
impl<T: Numeric> sealed::Prefix<T> for (i64, T) {
    const PARTIAL: bool = true;
}
impl<T: Numeric> MergeSortPrefix<T> for (i64, T) {}
