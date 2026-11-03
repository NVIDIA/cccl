// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{
    Alignment, Block, Numeric, SupportedAlignment, TempStorage, TempStorageLike, ThreadData,
    ThreadGroup,
};

mod boundary;

/// A static TopK count. Use [`TopKOptions::k_const`] or
/// [`TopKOptions::valid_items_const`] to select it.
///
/// Counts must be nonnegative and fit CUB int capacity. The provider also
/// checks the concrete Block tile bound; no generic constant expressions are needed.
#[derive(Clone, Copy, Debug)]
pub struct TopKCount<const COUNT: i64>;

/// A sealed required selection count: signed64 runtime or [`TopKCount`].
pub trait TopKSelectionCount: sealed::Count {}
/// A sealed valid-input selection, including `()` for the full tile.
pub trait TopKValidItems: sealed::Valid {}
/// Independently typed associated values for TopK pairs.
///
/// All ten [`Numeric`] types are supported. Values move with their keys without
/// numeric conversion, including floating-point NaN and signed-zero bit patterns.
pub trait TopKValue: Numeric + sealed::Value {}

/// Required k, optional valid prefix and optional existing Block scratch.
///
/// Default options omit k; choose [`Self::k`] or [`Self::k_const`] before calling
/// a TopK operation. Omitted valid items means the complete tile. Builders
/// preserve all other selections and can be called in either order.
///
/// ```
/// use cuda_coop as coop;
/// let _ = coop::TopKOptions::default().k_const::<17>().valid_items(31);
/// let _ = coop::TopKOptions::default().valid_items_const::<31>().k(17);
/// ```
pub struct TopKOptions<Count = (), Valid = (), S = ()> {
    count: Count,
    valid: Valid,
    storage: S,
}
impl Default for TopKOptions {
    fn default() -> Self {
        Self {
            count: (),
            valid: (),
            storage: (),
        }
    }
}
impl<C, V, S> TopKOptions<C, V, S> {
    /// Select a uniform signed64 k, checked in `0..=block_size * N` before narrowing.
    /// Widen smaller integer controls losslessly. k may exceed valid_items;
    /// then all valid inputs are selected. The output is not sorted.
    pub fn k(self, count: i64) -> TopKOptions<i64, V, S> {
        TopKOptions {
            count,
            valid: self.valid,
            storage: self.storage,
        }
    }
    /// Select a static k, checked against the concrete Block tile by the provider.
    pub fn k_const<const COUNT: i64>(self) -> TopKOptions<TopKCount<COUNT>, V, S> {
        TopKOptions {
            count: TopKCount,
            valid: self.valid,
            storage: self.storage,
        }
    }
    /// Select a uniform signed64 valid prefix, checked before narrowing.
    ///
    /// Every slot in every input must still be initialized, even beyond this
    /// prefix or when k or valid_items is zero. Explicit full counts retain
    /// the partial-tile recipe; omitting this option selects the full recipe.
    pub fn valid_items(self, valid: i64) -> TopKOptions<C, i64, S> {
        TopKOptions {
            count: self.count,
            valid,
            storage: self.storage,
        }
    }
    /// Select a static valid prefix, checked against the concrete Block tile.
    pub fn valid_items_const<const COUNT: i64>(self) -> TopKOptions<C, TopKCount<COUNT>, S> {
        TopKOptions {
            count: self.count,
            valid: TopKCount,
            storage: self.storage,
        }
    }
    /// Borrow existing Block scratch, retaining k and valid-input selections.
    ///
    /// Manual storage requires Block completion before conflicting reuse;
    /// auto-sync storage retains its existing trailing barrier. Even empty
    /// calls retain collective participation and storage obligations.
    pub fn temp_storage<const C2: usize, const A: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C2, A, AUTO, EX>,
    ) -> TopKOptions<C, V, &mut TempStorage<C2, A, AUTO, EX>> {
        TopKOptions {
            count: self.count,
            valid: self.valid,
            storage,
        }
    }
}

/// Select the smallest keys into a fresh, unsorted blocked prefix.
///
/// All ten [`Numeric`] types are supported, with positive N. Complete 1D
/// Blocks of 1..=1024 threads participate; the provider checks `block_size * N`
/// fits CUB int capacity. The result preserves input type, extent and alignment.
/// Both counts are uniform signed64 values in `0..=block_size * N`. Only the
/// first `min(k, valid_items)` slots are initialized; every other output bit,
/// including bitmap padding, is clear and its numeric slot remains unreadable.
///
/// ALL input slots must be initialized even for empty selection. The input and
/// its map remain unchanged. Invalid counts or unreadable input trap before
/// numeric reads; a uniform empty selection skips CUB after validation.
/// Equal-key choice/order is unspecified, +0/-0 compare as equal, and no NaN
/// key ordering is promised. This operation does not sort the selected keys.
///
/// Implicit scratch uses existing compiler planning. Explicit scratch preserves
/// its manual/auto-sync/exclusive policy; complete unrelated global accesses
/// explicitly with `cuda_device::thread::sync_threads()`. Execution requires an
/// Oxide CUDA kernel; generic key/N/alignment callers need no GCE feature.
#[inline(always)]
pub fn topk_min_keys<K, C, V, S, const N: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<K, N, ALIGN>,
    options: TopKOptions<C, V, S>,
) -> ThreadData<K, N, ALIGN>
where
    K: Numeric,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::keys::<K, C, V, S, N, ALIGN, false>(keys, options)
}

/// Select the smallest associated pairs into two fresh unsorted blocked prefixes.
///
/// The geometry, count, completion and empty-selection rules are those of
/// [`topk_min_keys`]. Keys and values independently support ten Numeric types,
/// share N, and retain independent minimum alignments in their outputs. Both
/// complete input validity maps are checked before reading either numeric array,
/// even at k or valid_items zero. Both borrowed inputs remain unchanged.
///
/// Each value follows its key unchanged. Equal-key tie selection/order is
/// unspecified; no stable sorting is promised. Floating value bits are opaque.
/// Both outputs initialize exactly the same prefix; their other bits are clear.
/// Scratch is queried for the actual key/value specialization.
#[inline(always)]
pub fn topk_min_pairs<K, T, C, V, S, const N: usize, const KA: usize, const VA: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<K, N, KA>,
    values: &ThreadData<T, N, VA>,
    options: TopKOptions<C, V, S>,
) -> (ThreadData<K, N, KA>, ThreadData<T, N, VA>)
where
    K: Numeric,
    T: TopKValue,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<KA>: SupportedAlignment,
    Alignment<VA>: SupportedAlignment,
{
    boundary::pairs::<K, T, C, V, S, N, KA, VA, false>(keys, values, options)
}

/// Select the largest keys into a fresh, unsorted blocked prefix.
///
/// All ten [`Numeric`] types are supported, with positive N. Complete 1D
/// Blocks of 1..=1024 threads participate; the provider checks `block_size * N`
/// fits CUB int capacity. The result preserves input type, extent and alignment.
/// Both counts are uniform signed64 values in `0..=block_size * N`. Only the
/// first `min(k, valid_items)` slots are initialized; every other output bit,
/// including bitmap padding, is clear and its numeric slot remains unreadable.
///
/// ALL input slots must be initialized even for empty selection. The input and
/// its map remain unchanged. Invalid counts or unreadable input trap before
/// numeric reads; a uniform empty selection skips CUB after validation.
/// Equal-key choice/order is unspecified, +0/-0 compare as equal, and no NaN
/// key ordering is promised. This operation does not sort the selected keys.
///
/// Implicit scratch uses existing compiler planning. Explicit scratch preserves
/// its manual/auto-sync/exclusive policy; complete unrelated global accesses
/// explicitly with `cuda_device::thread::sync_threads()`. Execution requires an
/// Oxide CUDA kernel; generic key/N/alignment callers need no GCE feature.
#[inline(always)]
pub fn topk_max_keys<K, C, V, S, const N: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<K, N, ALIGN>,
    options: TopKOptions<C, V, S>,
) -> ThreadData<K, N, ALIGN>
where
    K: Numeric,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::keys::<K, C, V, S, N, ALIGN, true>(keys, options)
}

/// Select the largest associated pairs into two fresh unsorted blocked prefixes.
///
/// The geometry, count, completion and empty-selection rules are those of
/// [`topk_max_keys`]. Keys and values independently support ten Numeric types,
/// share N, and retain independent minimum alignments in their outputs. Both
/// complete input validity maps are checked before reading either numeric array,
/// even at k or valid_items zero. Both borrowed inputs remain unchanged.
///
/// Each value follows its key unchanged. Equal-key tie selection/order is
/// unspecified; no stable sorting is promised. Floating value bits are opaque.
/// Both outputs initialize exactly the same prefix; their other bits are clear.
/// Scratch is queried for the actual key/value specialization.
#[inline(always)]
pub fn topk_max_pairs<K, T, C, V, S, const N: usize, const KA: usize, const VA: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<K, N, KA>,
    values: &ThreadData<T, N, VA>,
    options: TopKOptions<C, V, S>,
) -> (ThreadData<K, N, KA>, ThreadData<T, N, VA>)
where
    K: Numeric,
    T: TopKValue,
    C: TopKSelectionCount,
    V: TopKValidItems,
    S: TempStorageLike,
    Alignment<KA>: SupportedAlignment,
    Alignment<VA>: SupportedAlignment,
{
    boundary::pairs::<K, T, C, V, S, N, KA, VA, true>(keys, values, options)
}

mod sealed {
    pub trait Count: Copy {
        const MODE: u8;
        const STATIC: i64;
        fn value(self) -> i64;
    }
    pub trait Valid: Count {
        type Controls: Copy;
        fn controls(self, k: i64) -> Self::Controls;
    }
    pub trait Value {
        const TAG: [u8; 3];
    }
}
impl sealed::Count for () {
    const MODE: u8 = b'f';
    const STATIC: i64 = 0;
    fn value(self) -> i64 {
        0
    }
}
impl sealed::Valid for () {
    type Controls = (i64,);
    fn controls(self, k: i64) -> Self::Controls {
        (k,)
    }
}
impl TopKValidItems for () {}
impl sealed::Count for i64 {
    const MODE: u8 = b'r';
    const STATIC: i64 = 0;
    fn value(self) -> i64 {
        self
    }
}
impl sealed::Valid for i64 {
    type Controls = (i64, i64);
    fn controls(self, k: i64) -> Self::Controls {
        (k, self)
    }
}
impl TopKSelectionCount for i64 {}
impl TopKValidItems for i64 {}
impl<const COUNT: i64> sealed::Count for TopKCount<COUNT> {
    const MODE: u8 = b's';
    const STATIC: i64 = {
        assert!(
            COUNT >= 0 && COUNT <= i32::MAX as i64,
            "TopK static count must be in 0..=i32::MAX"
        );
        COUNT
    };
    fn value(self) -> i64 {
        Self::STATIC
    }
}
impl<const COUNT: i64> sealed::Valid for TopKCount<COUNT> {
    type Controls = (i64, i64);
    fn controls(self, k: i64) -> Self::Controls {
        (k, <Self as sealed::Count>::STATIC)
    }
}
impl<const COUNT: i64> TopKSelectionCount for TopKCount<COUNT> {}
impl<const COUNT: i64> TopKValidItems for TopKCount<COUNT> {}
macro_rules! values {
    ($(($ty:ty, $tag:literal)),+) => {$ (
        impl sealed::Value for $ty { const TAG: [u8; 3] = *$tag; }
        impl TopKValue for $ty {}
    )+};
}
values!(
    (i8, b"i8_"),
    (u8, b"u8_"),
    (i16, b"i16"),
    (u16, b"u16"),
    (i32, b"i32"),
    (u32, b"u32"),
    (i64, b"i64"),
    (u64, b"u64"),
    (f32, b"f32"),
    (f64, b"f64")
);
