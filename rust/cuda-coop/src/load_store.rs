// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{
    Alignment, Block, GroupScope, Numeric, SupportedAlignment, TempStorage, TempStorageLike,
    ThreadData, ThreadGroup, Warp,
};
use core::marker::PhantomData;
use cuda_device::{CollectiveTiles, DisjointSlice};

mod boundary;

/// Direct blocked memory access without shared scratch.
#[derive(Clone, Copy, Debug, Default)]
pub struct Direct;

/// Direct striped access without shared scratch.
///
/// Thread rank `t` and item `i` address `t + group.count() * i`. Load produces
/// this striped arrangement; Store consumes it and preserves the payload.
#[derive(Clone, Copy, Debug, Default)]
pub struct Striped;

/// Blocked access using aligned vector operations for full tiles.
///
/// CUB checks the actual pointer after the element offset and falls back to
/// Direct access when it is not sufficiently aligned. Counts, including a
/// full-tile count, select Direct accesses. No extra alignment precondition or
/// shared scratch is required. Store preserves every payload value and bit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vectorize;

/// Coalesced access through CUB's shared-memory block exchange.
///
/// Load produces blocked payloads. Store permutes payload values and validity
/// into striped order, including slots outside its selected write prefix.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn copy_tile<const ITEMS_PER_THREAD: usize>(
///     input: &[i16],
///     output: &mut coop::DisjointSlice<'_, i16, coop::CollectiveTiles<ITEMS_PER_THREAD>>,
/// ) {
///     let block = coop::this_block();
///     let mut values = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
///     coop::load(block, input, &mut values,
///         coop::LoadOptions::default().algorithm(coop::Transpose));
///     coop::store(block, output, &mut values,
///         coop::StoreOptions::default().algorithm(coop::Transpose));
/// }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Transpose;

/// Coalesced blocked access through a shared-memory exchange within each warp.
///
/// The complete block participates; its total thread count must be divisible
/// by 32. Load produces blocked payloads. Store mutates every value and validity
/// bit into warp-striped order, including when its write count is zero. See
/// [`store`] for the exact permutation and synchronization requirements.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarpTranspose;

/// [`WarpTranspose`] using one warp's CUB exchange storage in successive slices.
///
/// Every thread in the complete block participates, including the block barriers
/// between slices. The block thread count must be divisible by 32. Load, Store
/// mutation and validity follow [`WarpTranspose`]. Counted Store additionally
/// reserves its validity plane for every warp; that plane is not timesliced.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarpTransposeTimesliced;

/// A sealed, compile-time algorithm selected through Load/Store options.
pub trait Algorithm: sealed::Algorithm + Copy {}
impl Algorithm for Direct {}
impl Algorithm for Striped {}
impl Algorithm for Vectorize {}
impl Algorithm for Transpose {}
impl Algorithm for WarpTranspose {}
impl Algorithm for WarpTransposeTimesliced {}
impl sealed::Algorithm for Direct {
    const KIND: u8 = sealed::DIRECT;
}
impl sealed::Algorithm for Striped {
    const KIND: u8 = sealed::STRIPED;
}
impl sealed::Algorithm for Vectorize {
    const KIND: u8 = sealed::VECTORIZE;
}
impl sealed::Algorithm for Transpose {
    const KIND: u8 = sealed::TRANSPOSE;
}
impl sealed::Algorithm for WarpTranspose {
    const KIND: u8 = sealed::WARP_TRANSPOSE;
}
impl sealed::Algorithm for WarpTransposeTimesliced {
    const KIND: u8 = sealed::WARP_TRANSPOSE_TIMESLICED;
}

/// A sealed group supported by a Load/Store algorithm and storage choice.
///
/// Blocks support the existing algorithms and explicit or implicit storage.
/// Physical [`Warp`] supports [`Direct`] with implicit storage `()`. Generic
/// helpers can carry this bound; ordinary calls infer it from the group/options.
pub trait LoadStoreGroup<A: Algorithm, S: TempStorageLike>: GroupScope + sealed::Group {}
impl<A: Algorithm, S: TempStorageLike> LoadStoreGroup<A, S> for Block {}
impl LoadStoreGroup<Direct, ()> for Warp {}
impl sealed::Group for Block {
    const WARP: bool = false;
}
impl sealed::Group for Warp {
    const WARP: bool = true;
}

/// Controls for a [`load`], using [`Direct`] by default.
///
/// `T` is inferred from the source and output. Defaults load the entire group
/// tile, leave no unspecified slots, and use element offset zero. A partial
/// load without an out-of-bounds default makes every slot outside its prefix
/// unwritten, including slots initialized before the call.
///
/// Omitted storage uses compiler-managed reuse synchronization.
/// [`temp_storage`](Self::temp_storage) borrows an explicit descriptor.
/// Controls, including whether a count/default is present, must be uniform
/// across the participating group. They are checked when the operation executes.
#[derive(Clone, Copy)]
pub struct LoadOptions<T, A = Direct, S = ()> {
    storage: S,
    algorithm: PhantomData<A>,
    valid_items: Option<usize>,
    oob_default: Option<T>,
    offset: usize,
}

impl<T> Default for LoadOptions<T, Direct> {
    fn default() -> Self {
        Self {
            storage: (),
            algorithm: PhantomData,
            valid_items: None,
            oob_default: None,
            offset: 0,
        }
    }
}

impl<T, A: Algorithm, S: TempStorageLike> LoadOptions<T, A, S> {
    /// Use this explicit descriptor instead of compiler-managed scratch.
    ///
    /// The options hold a mutable borrow until consumed by the operation.
    /// Construct fresh options with an ordinary reborrow for each call in a loop.
    pub fn temp_storage<
        const C: usize,
        const ALIGN: usize,
        const AUTO_SYNC: bool,
        const EXCLUSIVE: bool,
    >(
        self,
        storage: &mut TempStorage<C, ALIGN, AUTO_SYNC, EXCLUSIVE>,
    ) -> LoadOptions<T, A, &mut TempStorage<C, ALIGN, AUTO_SYNC, EXCLUSIVE>> {
        LoadOptions {
            storage,
            algorithm: PhantomData,
            valid_items: self.valid_items,
            oob_default: self.oob_default,
            offset: self.offset,
        }
    }

    /// Select the compile-time algorithm while preserving the other controls.
    pub fn algorithm<B: Algorithm>(self, _algorithm: B) -> LoadOptions<T, B, S> {
        LoadOptions {
            storage: self.storage,
            algorithm: PhantomData,
            valid_items: self.valid_items,
            oob_default: self.oob_default,
            offset: self.offset,
        }
    }

    /// Load this many elements of the whole group tile, starting at its origin.
    ///
    /// The count must not exceed the tile size or `i32::MAX`. Zero is permitted.
    pub fn valid_items(mut self, count: usize) -> Self {
        self.valid_items = Some(count);
        self
    }

    /// Initialize slots outside the valid prefix with this typed value.
    ///
    /// Requires an explicit [`valid_items`](Self::valid_items) count. The value
    /// must have the source's element type and be uniform across the group.
    pub fn oob_default(mut self, value: T) -> Self {
        self.oob_default = Some(value);
        self
    }

    /// Set an element offset from the start of the source, defaulting to zero.
    ///
    /// Include the block's origin explicitly. Warp adds its x-fastest warp index
    /// times `32 * ITEMS` automatically. The accessed prefix must fit the source
    /// and, for a collective carrier, the group's fixed ownership region.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// Controls for a [`store`], using [`Direct`] by default.
///
/// Defaults write the entire group tile at element offset zero. A partial
/// store leaves destination elements outside its prefix unchanged. Controls,
/// including whether a count is present, must be uniform across the group.
/// Omitted storage uses compiler-managed reuse synchronization;
/// [`temp_storage`](Self::temp_storage) selects explicit storage.
#[derive(Clone, Copy)]
pub struct StoreOptions<A = Direct, S = ()> {
    storage: S,
    algorithm: PhantomData<A>,
    valid_items: Option<usize>,
    offset: usize,
}

impl Default for StoreOptions<Direct> {
    fn default() -> Self {
        Self {
            storage: (),
            algorithm: PhantomData,
            valid_items: None,
            offset: 0,
        }
    }
}

impl<A: Algorithm, S: TempStorageLike> StoreOptions<A, S> {
    /// Use this explicit descriptor instead of compiler-managed scratch.
    ///
    /// The options hold a mutable borrow until consumed by the operation.
    /// Construct fresh options with an ordinary reborrow for each call in a loop.
    pub fn temp_storage<
        const C: usize,
        const ALIGN: usize,
        const AUTO_SYNC: bool,
        const EXCLUSIVE: bool,
    >(
        self,
        storage: &mut TempStorage<C, ALIGN, AUTO_SYNC, EXCLUSIVE>,
    ) -> StoreOptions<A, &mut TempStorage<C, ALIGN, AUTO_SYNC, EXCLUSIVE>> {
        StoreOptions {
            storage,
            algorithm: PhantomData,
            valid_items: self.valid_items,
            offset: self.offset,
        }
    }

    /// Select the compile-time algorithm while preserving the other controls.
    pub fn algorithm<B: Algorithm>(self, _algorithm: B) -> StoreOptions<B, S> {
        StoreOptions {
            storage: self.storage,
            algorithm: PhantomData,
            valid_items: self.valid_items,
            offset: self.offset,
        }
    }

    /// Write this many elements of the whole group tile.
    ///
    /// The count must not exceed the tile size or `i32::MAX`. A scalar payload
    /// has one item per thread. Zero writes no elements.
    pub fn valid_items(mut self, count: usize) -> Self {
        self.valid_items = Some(count);
        self
    }

    /// Set an element offset from the start of the destination, default zero.
    ///
    /// Include the block's origin explicitly. Warp adds its x-fastest warp index
    /// times `32 * ITEMS` automatically. The written prefix must fit the
    /// allocation and the group's fixed ownership region.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// Load a group tile into an existing per-thread payload.
///
/// `T` is inferred from `source`; `ITEMS` is the positive per-thread extent and
/// `ALIGN` is the payload's minimum alignment. The tile has
/// `group.count() * ITEMS` elements. Striped gives each thread items separated
/// by the block size; the other algorithms give it consecutive items.
/// [`LoadSource`] accepts an immutable slice or a mutable borrow of a collective
/// carrier. [`LoadOptions`] supplies the count, typed default and element offset.
/// The operation returns unit. [`Direct`], [`Striped`] and [`Vectorize`] use no
/// shared scratch. The three transposing algorithms use compiler-planned shared
/// storage and return blocked payloads.
///
/// Complete one-, two- and three-dimensional blocks support all six algorithms.
/// [`WarpTranspose`] and [`WarpTransposeTimesliced`] require a total thread count
/// divisible by 32; they still require participation by every block thread.
/// Thread rank is x-fastest: `t = x + X * (y + Y * z)` for shape `[X, Y, Z]`.
/// Slot `i` addresses `t + group.count() * i` for Striped Load and `t * ITEMS + i`
/// for the other algorithms, relative to the selected offset.
/// Compile with the supported Oxide backend and an exact block launch contract.
/// Physical [`Warp`] supports [`Direct`] with implicit storage when the exact
/// block size is a multiple of 32. Its origin is the caller's element offset
/// plus its x-fastest warp index times `32 * ITEMS`. Every participating group
/// thread must reach matching calls with uniform source, controls and
/// participation. Sibling warps may choose different controls. The compiler
/// checks these requirements and resource
/// conflicts; a group value or Rust borrow alone does not prove them.
///
/// Loading a carrier after Store requires block completion, such as
/// `cuda_device::thread::sync_threads()`. Ending a borrow is not completion.
/// Full/filled Load initializes every output slot. Masked Load without a default
/// resets every out-of-prefix slot to unwritten; access through [`ThreadData`]
/// remains checked. No initialized slice is exposed.
///
/// # Panics
///
/// Invalid counts, offsets, bounds, ownership regions, and a default without a
/// count trap on device. Group/control violations are compilation errors.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn copy_tile<const ITEMS_PER_THREAD: usize>(input: &[i16], count: usize) {
///     let mut items = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
///     coop::load(coop::this_block(), input, &mut items,
///         coop::LoadOptions::default().valid_items(count).oob_default(-7_i16));
/// }
/// ```
#[inline(always)]
pub fn load<
    T: Numeric,
    Source: LoadSource<T>,
    G: LoadStoreGroup<A, S>,
    const ITEMS: usize,
    const ALIGN: usize,
    A: Algorithm,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    source: Source,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T, A, S>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    source.load(group, output, options);
}

/// Store a group tile from a per-thread payload.
///
/// `T` is inferred from the destination and value. [`StoreValue`] accepts a
/// shared or mutable borrow of [`ThreadData`] for [`Direct`], [`Striped`] and [`Vectorize`], a
/// mutable borrow for [`Transpose`], [`WarpTranspose`] or [`WarpTransposeTimesliced`],
/// or a numeric scalar (one item per thread)
/// for any algorithm.
/// `CAPACITY` describes the carrier's ownership grain; it need not equal the
/// payload extent. Direct and Vectorize consume blocked items; Striped consumes striped items.
/// These algorithms preserve every value and initialized bit and use no shared scratch.
/// Transpose uses compiler-planned shared storage and permutes all payload slots
/// into striped order, including a zero-count Store. Destination lane/item
/// `(t, i)` receives the prior blocked slot at linear index `group.count()*i+t`.
/// Both warp-transpose algorithms instead use warp-striped index
/// `(t / 32) * (32 * ITEMS) + 32 * i + t % 32`, where `ITEMS` is the payload extent.
/// They also mutate every slot, including a zero-count Store. All three preserve
/// the initialized bit of the moved value, including unwritten slots and slots
/// outside the write prefix. Scalars use a mutable one-item temporary.
///
/// The supported groups, compiler/launch and uniformity requirements are those
/// of [`load`]. The destination is the canonical collective carrier: it grants
/// fixed group regions without exposing duplicate mutable lane references.
/// For Warp, each warp owns `32 * CAPACITY` elements within the block region.
/// When `CAPACITY != ITEMS`, choose an offset whose automatic Warp origin still
/// lies in that fixed region; the carrier is not repartitioned by this call.
/// Every written payload slot must be initialized. Slots outside the written
/// prefix may be unwritten. A following conflicting access to this carrier
/// needs block completion; Drop is not completion. Managed scratch reuse may
/// insert a barrier before a later scratchful call, but is not a trailing
/// completion guarantee. Use an explicit block barrier when completion is needed.
///
/// # Panics
///
/// Invalid counts, offsets, bounds, ownership regions or an unwritten value in
/// the write domain trap on device. Failure does not roll back other lanes'
/// stores. Group/control/resource violations are compilation errors.
///
/// ```no_run
/// use cuda_coop as coop;
/// fn write_scalar(output: &mut coop::DisjointSlice<'_, i16, coop::CollectiveTiles<1>>) {
///     coop::store(coop::this_block(), output, 7_i16, Default::default());
/// }
/// ```
#[inline(always)]
pub fn store<
    T: Numeric,
    V: StoreValue<T, A>,
    G: LoadStoreGroup<A, S>,
    const CAPACITY: usize,
    A: Algorithm,
    S: TempStorageLike,
>(
    group: ThreadGroup<G>,
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: V,
    options: StoreOptions<A, S>,
) {
    value.store(group, destination, options);
}

/// A sealed source accepted by [`load`] for numeric element type `T`.
///
/// Implemented for `&[T]` and `&mut DisjointSlice<T, CollectiveTiles<CAPACITY>>`.
/// The latter supports collective readback after explicit completion of a prior
/// Store. Implementations retain the original allocation witness and bounds;
/// this trait does not expose pointers or grant lane references.
pub trait LoadSource<T: Numeric>: sealed::LoadSource {
    #[doc(hidden)]
    fn load<
        const ITEMS: usize,
        const ALIGN: usize,
        A: Algorithm,
        S: TempStorageLike,
        G: LoadStoreGroup<A, S>,
    >(
        self,
        group: ThreadGroup<G>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A, S>,
    ) where
        Alignment<ALIGN>: SupportedAlignment;
}

impl<T: Numeric> sealed::LoadSource for &[T] {}
impl<T: Numeric> LoadSource<T> for &[T] {
    #[inline(always)]
    fn load<
        const ITEMS: usize,
        const ALIGN: usize,
        A: Algorithm,
        S: TempStorageLike,
        G: LoadStoreGroup<A, S>,
    >(
        self,
        _group: ThreadGroup<G>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A, S>,
    ) where
        Alignment<ALIGN>: SupportedAlignment,
    {
        // SAFETY: the pointer, length and immutable witness describe this slice.
        unsafe {
            boundary::load::<_, _, ITEMS, ALIGN, A, S, G>(
                self,
                self.as_ptr(),
                self.len(),
                output,
                options,
            );
        }
    }
}

impl<T: Numeric, const C: usize> sealed::LoadSource
    for &mut DisjointSlice<'_, T, CollectiveTiles<C>>
{
}
impl<T: Numeric, const C: usize> LoadSource<T> for &mut DisjointSlice<'_, T, CollectiveTiles<C>> {
    #[inline(always)]
    fn load<
        const ITEMS: usize,
        const ALIGN: usize,
        A: Algorithm,
        S: TempStorageLike,
        G: LoadStoreGroup<A, S>,
    >(
        self,
        _group: ThreadGroup<G>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A, S>,
    ) where
        Alignment<ALIGN>: SupportedAlignment,
    {
        let valid = match options.valid_items {
            Some(count) => count,
            None => boundary::group_tile::<ITEMS, G>(),
        };
        let offset = boundary::group_offset::<ITEMS, G>(options.offset);
        boundary::group_containment::<C, G>(offset, valid);
        let pointer = self.as_mut_ptr().cast_const();
        let len = self.len();
        // SAFETY: the pointer and witness share an allocation; containment was
        // checked. The compiler requires completion after conflicting accesses.
        unsafe {
            boundary::load::<_, _, ITEMS, ALIGN, A, S, G>(&*self, pointer, len, output, options);
        }
    }
}

/// A sealed per-thread value accepted by [`store`] for type `T` and algorithm `A`.
///
/// Numeric scalars and mutable [`ThreadData`] borrows support all algorithms.
/// Shared payload borrows support [`Direct`], [`Striped`] and [`Vectorize`]. A scalar contributes
/// one item per thread. These algorithms preserve every value and bit;
/// the transposing algorithms mutate their positions as described by [`store`].
pub trait StoreValue<T: Numeric, A: Algorithm = Direct>: sealed::StoreValue<T, A> {
    #[doc(hidden)]
    fn store<const CAPACITY: usize, S: TempStorageLike, G: LoadStoreGroup<A, S>>(
        self,
        group: ThreadGroup<G>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
        options: StoreOptions<A, S>,
    );
}

macro_rules! shared_store_value {
    ($algorithm:ty) => {
        impl<T: Numeric, const I: usize, const ALIGN: usize> sealed::StoreValue<T, $algorithm>
            for &ThreadData<T, I, ALIGN>
        where
            Alignment<ALIGN>: SupportedAlignment,
        {
        }
        impl<T: Numeric, const I: usize, const ALIGN: usize> StoreValue<T, $algorithm>
            for &ThreadData<T, I, ALIGN>
        where
            Alignment<ALIGN>: SupportedAlignment,
        {
            #[inline(always)]
            fn store<const C: usize, S: TempStorageLike, G: LoadStoreGroup<$algorithm, S>>(
                self,
                _group: ThreadGroup<G>,
                destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
                options: StoreOptions<$algorithm, S>,
            ) {
                boundary::store::<_, I, ALIGN, C, $algorithm, S, G>(destination, self, options);
            }
        }
    };
}
shared_store_value!(Direct);
shared_store_value!(Striped);
shared_store_value!(Vectorize);
impl<T: Numeric, const I: usize, const ALIGN: usize, A: Algorithm> sealed::StoreValue<T, A>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}

impl<T: Numeric, const I: usize, const ALIGN: usize, A: Algorithm> StoreValue<T, A>
    for &mut ThreadData<T, I, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline(always)]
    fn store<const C: usize, S: TempStorageLike, G: LoadStoreGroup<A, S>>(
        self,
        _group: ThreadGroup<G>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions<A, S>,
    ) {
        match A::KIND {
            sealed::DIRECT => {
                boundary::store::<_, I, ALIGN, C, Direct, S, G>(
                    destination,
                    self,
                    options.algorithm(Direct),
                );
            }
            sealed::STRIPED => {
                boundary::store::<_, I, ALIGN, C, Striped, S, G>(
                    destination,
                    self,
                    options.algorithm(Striped),
                );
            }
            sealed::VECTORIZE => {
                boundary::store::<_, I, ALIGN, C, Vectorize, S, G>(
                    destination,
                    self,
                    options.algorithm(Vectorize),
                );
            }
            sealed::TRANSPOSE => {
                boundary::store_permuting(destination, self, options.algorithm(Transpose));
            }
            sealed::WARP_TRANSPOSE => {
                boundary::store_permuting(destination, self, options.algorithm(WarpTranspose));
            }
            sealed::WARP_TRANSPOSE_TIMESLICED => {
                boundary::store_permuting(
                    destination,
                    self,
                    options.algorithm(WarpTransposeTimesliced),
                );
            }
            _ => panic!(),
        }
    }
}
impl<T: Numeric, A: Algorithm> sealed::StoreValue<T, A> for T {}
impl<T: Numeric, A: Algorithm> StoreValue<T, A> for T {
    #[inline(always)]
    fn store<const C: usize, S: TempStorageLike, G: LoadStoreGroup<A, S>>(
        self,
        group: ThreadGroup<G>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions<A, S>,
    ) {
        let mut value = ThreadData::<_, 1>::new();
        value.write(0, self);
        (&mut value).store(group, destination, options);
    }
}

mod sealed {
    pub trait LoadSource {}
    pub trait StoreValue<T, A> {}
    pub trait Group {
        const WARP: bool;
    }
    // Scalar tags let the compiler prune unused native operation arms before
    // planning scratch. Only the sealed selector implementations assign them.
    pub const DIRECT: u8 = 0;
    pub const STRIPED: u8 = 1;
    pub const TRANSPOSE: u8 = 2;
    pub const VECTORIZE: u8 = 3;
    pub const WARP_TRANSPOSE: u8 = 4;
    pub const WARP_TRANSPOSE_TIMESLICED: u8 = 5;
    pub trait Algorithm {
        const KIND: u8;
    }
}
