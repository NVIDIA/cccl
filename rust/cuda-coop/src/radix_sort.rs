// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::marker::PhantomData;

use crate::{
    Alignment, Block, Numeric, SupportedAlignment, TempStorage, TempStorageLike, ThreadData,
    ThreadGroup,
};

mod boundary;

/// Select ascending ordering.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ascending;

/// Select descending ordering.
#[derive(Clone, Copy, Debug, Default)]
pub struct Descending;

/// Sealed Radix Sort and Rank key types: i32, u32, i64 and u64.
pub trait RadixKey: Numeric + sealed::Key {}

/// Sealed associated-value types for [`radix_sort_pairs`].
///
/// Supports i8, u8, i16, u16, i32, u32, i64, u64, f32 and f64. Values move
/// with their keys without comparisons or arithmetic, preserving their bits.
pub trait RadixValue: Numeric + sealed::Value {}

/// A sealed static Radix Sort or Rank order: [`Ascending`] or [`Descending`].
pub trait RadixSortOrder: sealed::Order {}

/// A compile-time bit endpoint selected by the const option builders.
///
/// A [`RadixSortOptions::begin_bit_const`] or
/// [`RadixSortOptions::end_bit_const`] endpoint is checked when the operation
/// is instantiated with its key type and other endpoint.
#[derive(Clone, Copy, Debug, Default)]
pub struct RadixBit<const BIT: i64>;

/// A sealed Radix Sort endpoint: omitted `()`, runtime `i64` or [`RadixBit`].
///
/// This trait records option types; it does not establish uniformity. Runtime
/// endpoints must be Block-uniform, as checked by the device compiler.
pub trait RadixSortBit: sealed::Bit {}

/// Order, independent bit endpoints and scratch for Radix Sort keys or pairs.
///
/// Canonical [`Default`] selects [`Ascending`], begin zero, omitted end and
/// implicit scratch. The omitted end is always the key width, even with a
/// nonzero begin. `T` is inferred from input; builders consume options and
/// preserve every other choice, including an explicit storage borrow.
///
/// Value builders accept `i64` and validate the final interval at runtime.
/// Const builders request compilation-time checks at monomorphization. A
/// literal passed to a value builder still uses the runtime route. Mixed
/// static/runtime bounds are supported. Every interval must satisfy
/// `0 <= begin < end <= key_width`; empty intervals are invalid.
///
/// ```
/// use cuda_coop as coop;
/// let _: coop::RadixSortOptions<i64> = Default::default();
/// let _ = coop::RadixSortOptions::<i64>::default()
///     .begin_bit_const::<8>().end_bit(32).order(coop::Descending);
/// ```
pub struct RadixSortOptions<T, O = Ascending, Begin = (), End = (), S = ()> {
    begin: Begin,
    end: End,
    storage: S,
    selection: PhantomData<(T, O)>,
}

impl<T: RadixKey> Default for RadixSortOptions<T> {
    fn default() -> Self {
        Self {
            begin: (),
            end: (),
            storage: (),
            selection: PhantomData,
        }
    }
}

impl<T: RadixKey, O, Begin, End, S> RadixSortOptions<T, O, Begin, End, S> {
    /// Select ascending or descending order, preserving both endpoints.
    pub fn order<O2: RadixSortOrder>(self, _order: O2) -> RadixSortOptions<T, O2, Begin, End, S> {
        RadixSortOptions {
            begin: self.begin,
            end: self.end,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Set a Block-uniform runtime begin, without changing the end.
    ///
    /// An omitted end remains the full key width. Widen smaller integer values
    /// losslessly to `i64`; bounds are checked before conversion to CUB's int.
    pub fn begin_bit(self, begin: i64) -> RadixSortOptions<T, O, i64, End, S> {
        RadixSortOptions {
            begin,
            end: self.end,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Set a Block-uniform runtime end, without changing the begin.
    pub fn end_bit(self, end: i64) -> RadixSortOptions<T, O, Begin, i64, S> {
        RadixSortOptions {
            begin: self.begin,
            end,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Set a static begin, checked when the sort operation is instantiated.
    ///
    /// An omitted end remains the full key width. This replaces any prior begin
    /// selection; validation uses the final options, not intermediate builders.
    pub fn begin_bit_const<const BIT: i64>(self) -> RadixSortOptions<T, O, RadixBit<BIT>, End, S> {
        RadixSortOptions {
            begin: RadixBit,
            end: self.end,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Set a static end, checked when the sort operation is instantiated.
    pub fn end_bit_const<const BIT: i64>(self) -> RadixSortOptions<T, O, Begin, RadixBit<BIT>, S> {
        RadixSortOptions {
            begin: self.begin,
            end: RadixBit,
            storage: self.storage,
            selection: PhantomData,
        }
    }

    /// Borrow scratch with its existing sharing and completion policy.
    ///
    /// Manual reuse requires `cuda_device::thread::sync_threads()` after the
    /// call. Auto-sync storage supplies its trailing barrier. Consuming options
    /// permit ordinary mutable reborrows on successive calls.
    pub fn temp_storage<const C: usize, const A: usize, const AUTO: bool, const EX: bool>(
        self,
        storage: &mut TempStorage<C, A, AUTO, EX>,
    ) -> RadixSortOptions<T, O, Begin, End, &mut TempStorage<C, A, AUTO, EX>> {
        RadixSortOptions {
            begin: self.begin,
            end: self.end,
            storage,
            selection: PhantomData,
        }
    }
}

/// Stably sort a complete Block's keys into a fresh blocked payload.
///
/// Every member participates with positive `ITEMS_PER_THREAD` initialized keys.
/// Rank `r`, item `i` has input/output index `r * ITEMS_PER_THREAD + i`, with
/// x-fastest rank across all three Block dimensions. The checked product of
/// Block size and extent must not exceed 65,535. The result has the same key
/// type, extent and minimum alignment, and every output slot is initialized.
/// Input values and validity bits are preserved, including through a mutable
/// reborrow. Generic T/N/ALIGN consumers need no `generic_const_exprs` feature.
///
/// Keys support the four [`RadixKey`] types. Equal selected digits retain their
/// original blocked order in both directions. Signed keys invert the sign bit
/// before selecting `[begin_bit, end_bit)` and restore the original key
/// representation in the result. Default options sort the full key ascending.
/// An omitted end always means full key width, independently of begin.
///
/// Static endpoints are checked at monomorphization; runtime endpoints are
/// Block-uniform `i64` values. Invalid runtime intervals or unreadable inputs
/// trap before native sorting reads. There is no partial count, fill, scalar,
/// Warp, floating-key or striped-output form.
///
/// Omitting storage requests compiler-planned CUB scratch and synchronization
/// before dirty reuse. Explicit [`TempStorage`] retains its manual/auto-sync
/// policy. Internal CUB synchronization grants no completion of unrelated
/// global accesses; use `cuda_device::thread::sync_threads()` when needed.
/// Execution requires an Oxide-compiled CUDA kernel.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn sorted<T: coop::RadixKey, const N: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     keys: &coop::ThreadData<T, N, ALIGN>,
/// ) -> coop::ThreadData<T, N, ALIGN>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::radix_sort_keys(block, keys, Default::default())
/// }
/// ```
#[inline(always)]
pub fn radix_sort_keys<T, O, Begin, End, S, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: RadixSortOptions<T, O, Begin, End, S>,
) -> ThreadData<T, ITEMS_PER_THREAD, ALIGN>
where
    T: RadixKey,
    O: RadixSortOrder,
    Begin: RadixSortBit,
    End: RadixSortBit,
    S: TempStorageLike,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::radix_sort_keys(keys, options)
}

/// Stably sort a complete Block's key/value pairs into two fresh payloads.
///
/// Keys use the four [`RadixKey`] types; associated values use [`RadixValue`].
/// Both borrowed inputs have the same positive `ITEMS_PER_THREAD`, with
/// independent minimum alignments. Each result preserves its input's type,
/// extent and alignment, and every result slot is initialized. Both inputs
/// and their validity maps remain unchanged, including through mutable
/// reborrows. Generic K/V/N/ALIGN consumers need no `generic_const_exprs`.
///
/// Every member of the complete XYZ Block participates. Rank `r`, item `i`
/// has blocked index `r * ITEMS_PER_THREAD + i`, with x-fastest rank. The
/// checked Block-size/extent product must not exceed 65,535. Both input maps
/// must be fully readable, even when an unreadable value's key is initialized.
///
/// Selected digits order the keys, and values follow their corresponding
/// keys. Equal selected digits retain their original blocked order in either
/// direction. Signed keys invert the sign bit before selecting the interval,
/// then restore their representation. Default options sort all key bits
/// ascending. An omitted end means the key width, independently of begin.
///
/// [`RadixSortOptions`] shares the keys-only static/runtime endpoint and
/// storage rules: `0 <= begin < end <= key_width`, runtime endpoints are
/// Block-uniform i64, and invalid bounds or unreadable inputs trap before
/// native numeric reads. Manual scratch reuse needs an explicit Block
/// completion; auto-sync storage supplies it. Internal sorting does not
/// complete unrelated global accesses. There is no Warp, partial-input,
/// floating-key or striped-output form. Execution needs an Oxide CUDA kernel.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn sorted<K: coop::RadixKey, V: coop::RadixValue,
///           const N: usize, const KA: usize, const VA: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     keys: &coop::ThreadData<K, N, KA>,
///     values: &coop::ThreadData<V, N, VA>,
/// ) -> (coop::ThreadData<K, N, KA>, coop::ThreadData<V, N, VA>)
/// where coop::Alignment<KA>: coop::SupportedAlignment,
///       coop::Alignment<VA>: coop::SupportedAlignment {
///     coop::radix_sort_pairs(block, keys, values, Default::default())
/// }
/// ```
#[inline(always)]
pub fn radix_sort_pairs<
    K,
    V,
    O,
    Begin,
    End,
    S,
    const ITEMS_PER_THREAD: usize,
    const KEY_ALIGN: usize,
    const VALUE_ALIGN: usize,
>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<K, ITEMS_PER_THREAD, KEY_ALIGN>,
    values: &ThreadData<V, ITEMS_PER_THREAD, VALUE_ALIGN>,
    options: RadixSortOptions<K, O, Begin, End, S>,
) -> (
    ThreadData<K, ITEMS_PER_THREAD, KEY_ALIGN>,
    ThreadData<V, ITEMS_PER_THREAD, VALUE_ALIGN>,
)
where
    K: RadixKey,
    V: RadixValue,
    O: RadixSortOrder,
    Begin: RadixSortBit,
    End: RadixSortBit,
    S: TempStorageLike,
    Alignment<KEY_ALIGN>: SupportedAlignment,
    Alignment<VALUE_ALIGN>: SupportedAlignment,
{
    boundary::radix_sort_pairs(keys, values, options)
}

/// A compile-time digit width selected by [`RadixRankOptions::radix_bits`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RadixRankBits<const BITS: i64>;

/// A sealed Rank width: omitted `()` or explicit [`RadixRankBits`].
pub trait RadixRankWidth: sealed::RankWidth {}

/// Static order and digit width, with a static or Block-uniform runtime begin.
///
/// Canonical [`Default`] selects ascending, begin zero and four radix bits.
/// With a static end and no explicit width, the width is `end - begin`.
/// Every width must be in `1..=8`, and the complete interval must fit the key.
/// A runtime begin requires an explicit width; a literal passed to `begin_bit`
/// still selects the runtime route. An optional end is always static and must
/// equal begin plus width. Builders validate their final combined choices at
/// monomorphization, or at runtime where begin is dynamic.
///
/// `T` is inferred from the input. Rank uses implicit compiler-planned scratch.
///
/// ```
/// use cuda_coop as coop;
/// let _: coop::RadixRankOptions<i64> = Default::default();
/// let _ = coop::RadixRankOptions::<i64>::default()
///     .begin_bit_const::<61>().end_bit_const::<64>();
/// let _ = coop::RadixRankOptions::<i64>::default()
///     .begin_bit(61).radix_bits::<3>().order(coop::Descending);
/// ```
pub struct RadixRankOptions<T, O = Ascending, Begin = (), End = (), Bits = ()> {
    begin: Begin,
    selection: PhantomData<(T, O, End, Bits)>,
}

impl<T: RadixKey> Default for RadixRankOptions<T> {
    fn default() -> Self {
        Self {
            begin: (),
            selection: PhantomData,
        }
    }
}

impl<T: RadixKey, O, Begin, End, Bits> RadixRankOptions<T, O, Begin, End, Bits> {
    /// Select static order, preserving the interval choices.
    pub fn order<O2: RadixSortOrder>(
        self,
        _order: O2,
    ) -> RadixRankOptions<T, O2, Begin, End, Bits> {
        RadixRankOptions {
            begin: self.begin,
            selection: PhantomData,
        }
    }

    /// Set a Block-uniform runtime begin; an explicit radix width is required.
    pub fn begin_bit(self, begin: i64) -> RadixRankOptions<T, O, i64, End, Bits> {
        RadixRankOptions {
            begin,
            selection: PhantomData,
        }
    }

    /// Set a static begin, checked with the final width and optional end.
    pub fn begin_bit_const<const BIT: i64>(
        self,
    ) -> RadixRankOptions<T, O, RadixBit<BIT>, End, Bits> {
        RadixRankOptions {
            begin: RadixBit,
            selection: PhantomData,
        }
    }

    /// Set a static end, deriving width when no explicit width is selected.
    pub fn end_bit_const<const BIT: i64>(
        self,
    ) -> RadixRankOptions<T, O, Begin, RadixBit<BIT>, Bits> {
        RadixRankOptions {
            begin: self.begin,
            selection: PhantomData,
        }
    }

    /// Select a static digit width in `1..=8`, preserving begin and end.
    pub fn radix_bits<const BITS: i64>(
        self,
    ) -> RadixRankOptions<T, O, Begin, End, RadixRankBits<BITS>> {
        RadixRankOptions {
            begin: self.begin,
            selection: PhantomData,
        }
    }
}

/// Rank selected key digits across a complete XYZ Block, preserving the input.
///
/// Each input item receives its position in the stable ascending or descending
/// selected-digit order. Ties receive increasing ranks in original blocked
/// order. The fresh result is `ThreadData<i32, ITEMS_PER_THREAD>` with natural
/// i32 alignment, independently of the key payload's minimum alignment.
/// Every rank slot is initialized; input values and validity bits are unchanged.
///
/// All Block members participate with positive `ITEMS_PER_THREAD` initialized
/// keys. The x-fastest Block-size/extent product must not exceed 65,535. Signed
/// keys invert their sign bit before extracting the selected digit. Defaults
/// rank the low four transformed bits, rather than sorting the whole key.
/// Generic T/N/ALIGN callers need no `generic_const_exprs` feature.
///
/// [`RadixRankOptions`] validates static choices at monomorphization. Dynamic
/// begin values must be Block-uniform; invalid intervals or unreadable inputs
/// trap before numeric reads, with wide bounds checked before narrowing.
/// Scratch and synchronization before dirty reuse are compiler-planned. Rank
/// does not complete unrelated global accesses. There is no explicit storage,
/// prefix-output, Warp or partial-input form. Execution needs an Oxide kernel.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn ranks<T: coop::RadixKey, const N: usize, const ALIGN: usize>(
///     block: coop::ThreadGroup<coop::Block>,
///     keys: &coop::ThreadData<T, N, ALIGN>,
/// ) -> coop::ThreadData<i32, N>
/// where coop::Alignment<ALIGN>: coop::SupportedAlignment {
///     coop::radix_rank_keys(block, keys, Default::default())
/// }
/// ```
#[inline(always)]
pub fn radix_rank_keys<T, O, Begin, End, Bits, const ITEMS_PER_THREAD: usize, const ALIGN: usize>(
    _group: ThreadGroup<Block>,
    keys: &ThreadData<T, ITEMS_PER_THREAD, ALIGN>,
    options: RadixRankOptions<T, O, Begin, End, Bits>,
) -> ThreadData<i32, ITEMS_PER_THREAD>
where
    T: RadixKey,
    O: RadixSortOrder,
    Begin: RadixSortBit,
    End: RadixSortBit,
    Bits: RadixRankWidth,
    Alignment<ALIGN>: SupportedAlignment,
{
    boundary::radix_rank_keys(keys, options)
}

mod sealed {
    pub trait Key {
        const WIDTH: i64;
    }
    pub trait Value {
        const ASCENDING_REGISTRATION: &'static str;
        const DESCENDING_REGISTRATION: &'static str;
    }
    pub trait RankWidth {
        const STATIC: Option<i64>;
    }
    pub trait Order {
        const DESCENDING: bool;
    }
    pub trait Bit {
        const DEFAULT: bool;
        const STATIC: Option<i64>;
        fn value(self, default: i64) -> i64;
    }
}

macro_rules! keys {
    ($(($ty:ty, $width:literal)),+ $(,)?) => {$ (
        impl sealed::Key for $ty { const WIDTH: i64 = $width; }
        impl RadixKey for $ty {}
    )+};
}
keys!((i32, 32), (u32, 32), (i64, 64), (u64, 64));

macro_rules! values {
    ($(($ty:ty, $ascending:literal, $descending:literal)),+ $(,)?) => {$ (
        impl sealed::Value for $ty {
            const ASCENDING_REGISTRATION: &'static str =
                cuda_device::library::registration!($ascending);
            const DESCENDING_REGISTRATION: &'static str =
                cuda_device::library::registration!($descending);
        }
        impl RadixValue for $ty {}
    )+};
}
values!(
    (
        i8,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_I8",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_I8"
    ),
    (
        u8,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_U8",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_U8"
    ),
    (
        i16,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_I16",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_I16"
    ),
    (
        u16,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_U16",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_U16"
    ),
    (
        i32,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_I32",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_I32"
    ),
    (
        u32,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_U32",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_U32"
    ),
    (
        i64,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_I64",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_I64"
    ),
    (
        u64,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_U64",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_U64"
    ),
    (
        f32,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_F32",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_F32"
    ),
    (
        f64,
        "COOP_RADIX_SORT_PAIRS_ASCENDING_F64",
        "COOP_RADIX_SORT_PAIRS_DESCENDING_F64"
    ),
);

impl sealed::Order for Ascending {
    const DESCENDING: bool = false;
}
impl RadixSortOrder for Ascending {}
impl sealed::Order for Descending {
    const DESCENDING: bool = true;
}
impl RadixSortOrder for Descending {}

impl sealed::Bit for () {
    const DEFAULT: bool = true;
    const STATIC: Option<i64> = None;
    #[inline(always)]
    fn value(self, default: i64) -> i64 {
        default
    }
}
impl RadixSortBit for () {}
impl sealed::Bit for i64 {
    const DEFAULT: bool = false;
    const STATIC: Option<i64> = None;
    #[inline(always)]
    fn value(self, _default: i64) -> i64 {
        self
    }
}
impl RadixSortBit for i64 {}
impl<const BIT: i64> sealed::Bit for RadixBit<BIT> {
    const DEFAULT: bool = false;
    const STATIC: Option<i64> = Some(BIT);
    #[inline(always)]
    fn value(self, _default: i64) -> i64 {
        BIT
    }
}
impl<const BIT: i64> RadixSortBit for RadixBit<BIT> {}

impl sealed::RankWidth for () {
    const STATIC: Option<i64> = None;
}
impl RadixRankWidth for () {}
impl<const BITS: i64> sealed::RankWidth for RadixRankBits<BITS> {
    const STATIC: Option<i64> = Some(BITS);
}
impl<const BITS: i64> RadixRankWidth for RadixRankBits<BITS> {}
