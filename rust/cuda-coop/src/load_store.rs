// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{Alignment, Block, Numeric, SupportedAlignment, ThreadData, ThreadGroup};
use core::marker::PhantomData;
use cuda_device::{CollectiveTiles, DisjointSlice};

mod boundary;

/// Direct blocked memory access without shared scratch.
#[derive(Clone, Copy, Debug, Default)]
pub struct Direct;

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

/// A sealed, compile-time algorithm selected through Load/Store options.
pub trait Algorithm: sealed::Algorithm + Copy {}
impl Algorithm for Direct {}
impl Algorithm for Transpose {}
impl sealed::Algorithm for Direct {
    const TRANSPOSE: bool = false;
}
impl sealed::Algorithm for Transpose {
    const TRANSPOSE: bool = true;
}

/// Controls for a [`load`], using [`Direct`] by default.
///
/// `T` is inferred from the source and output. Defaults load the entire group
/// tile, leave no unspecified slots, and use element offset zero. A partial
/// load without an out-of-bounds default makes every slot outside its prefix
/// unwritten, including slots initialized before the call.
///
/// Controls, including whether a count/default is present, must be uniform
/// across the block. They are checked when the operation executes.
#[derive(Clone, Copy)]
pub struct LoadOptions<T, A = Direct> {
    algorithm: PhantomData<A>,
    valid_items: Option<usize>,
    oob_default: Option<T>,
    offset: usize,
}

impl<T> Default for LoadOptions<T, Direct> {
    fn default() -> Self {
        Self {
            algorithm: PhantomData,
            valid_items: None,
            oob_default: None,
            offset: 0,
        }
    }
}

impl<T, A: Algorithm> LoadOptions<T, A> {
    /// Select the compile-time algorithm while preserving the other controls.
    pub fn algorithm<B: Algorithm>(self, _algorithm: B) -> LoadOptions<T, B> {
        LoadOptions {
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
    /// must have the source's element type and be uniform across the block.
    pub fn oob_default(mut self, value: T) -> Self {
        self.oob_default = Some(value);
        self
    }

    /// Set an element offset from the start of the source, defaulting to zero.
    ///
    /// Include the block's origin explicitly. The accessed prefix must fit the
    /// source and, for a collective carrier, this block's owned region.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// Controls for a [`store`], using [`Direct`] by default.
///
/// Defaults write the entire group tile at element offset zero. A partial
/// store leaves destination elements outside its prefix unchanged. Controls,
/// including whether a count is present, must be uniform across the block.
#[derive(Clone, Copy)]
pub struct StoreOptions<A = Direct> {
    algorithm: PhantomData<A>,
    valid_items: Option<usize>,
    offset: usize,
}

impl Default for StoreOptions<Direct> {
    fn default() -> Self {
        Self {
            algorithm: PhantomData,
            valid_items: None,
            offset: 0,
        }
    }
}

impl<A: Algorithm> StoreOptions<A> {
    /// Select the compile-time algorithm while preserving the other controls.
    pub fn algorithm<B: Algorithm>(self, _algorithm: B) -> StoreOptions<B> {
        StoreOptions {
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
    /// Include the block's origin explicitly. The written prefix must fit the
    /// allocation and this block's collective ownership region.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }
}

/// Load a block tile in blocked order into an existing per-thread payload.
///
/// `T` is inferred from `source`; `ITEMS` is the positive per-thread extent and
/// `ALIGN` is the payload's minimum alignment. The tile has
/// `group.count() * ITEMS` elements. Each thread receives consecutive items.
/// [`LoadSource`] accepts an immutable slice or a mutable borrow of a collective
/// carrier. [`LoadOptions`] supplies the count, typed default and element offset.
/// The operation returns unit. [`Direct`] uses no shared scratch; [`Transpose`]
/// uses compiler-planned shared storage and returns the same blocked layout.
///
/// Complete one-, two- and three-dimensional blocks support [`Direct`] and
/// [`Transpose`]. Thread rank is x-fastest: `x + X * (y + Y * z)` for shape
/// `[X, Y, Z]`. Each ranked thread receives `ITEMS` consecutive elements.
/// Compile with the supported Oxide backend and an exact block launch contract.
/// Every block thread must reach matching calls with uniform source, controls
/// and participation. The compiler checks these requirements and resource
/// conflicts; a group value or Rust borrow alone does not prove them.
///
/// Loading a carrier after Store requires explicit block completion, such as
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
pub fn load<T: Numeric, S: LoadSource<T>, const ITEMS: usize, const ALIGN: usize, A: Algorithm>(
    group: ThreadGroup<Block>,
    source: S,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T, A>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    source.load(group, output, options);
}

/// Store a block tile from a blocked per-thread payload.
///
/// `T` is inferred from the destination and value. [`StoreValue`] accepts a
/// shared or mutable borrow of [`ThreadData`] for [`Direct`], a mutable borrow
/// for [`Transpose`], or a numeric scalar (one item per thread) for either.
/// `CAPACITY` describes the carrier's ownership grain; it need not equal the
/// payload extent. Direct preserves values and flags and uses no shared scratch.
/// Transpose uses compiler-planned shared storage and permutes all payload slots
/// into striped order, including a zero-count Store. Destination lane/item
/// `(t, i)` receives the prior blocked slot at linear index `group.count()*i+t`.
/// Its initialized bit follows the value, including unwritten slots and slots
/// outside the write prefix. Scalars use a mutable one-item temporary.
///
/// The supported groups, compiler/launch and uniformity requirements are those
/// of [`load`]. The destination is the canonical collective carrier: it grants
/// disjoint block regions without exposing duplicate mutable lane references.
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
pub fn store<T: Numeric, V: StoreValue<T, A>, const CAPACITY: usize, A: Algorithm>(
    group: ThreadGroup<Block>,
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: V,
    options: StoreOptions<A>,
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
    fn load<const ITEMS: usize, const ALIGN: usize, A: Algorithm>(
        self,
        group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A>,
    ) where
        Alignment<ALIGN>: SupportedAlignment;
}

impl<T: Numeric> sealed::LoadSource for &[T] {}
impl<T: Numeric> LoadSource<T> for &[T] {
    #[inline(always)]
    fn load<const ITEMS: usize, const ALIGN: usize, A: Algorithm>(
        self,
        _group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A>,
    ) where
        Alignment<ALIGN>: SupportedAlignment,
    {
        // SAFETY: the pointer, length and immutable witness describe this slice.
        unsafe {
            boundary::load(self, self.as_ptr(), self.len(), output, options);
        }
    }
}

impl<T: Numeric, const C: usize> sealed::LoadSource
    for &mut DisjointSlice<'_, T, CollectiveTiles<C>>
{
}
impl<T: Numeric, const C: usize> LoadSource<T> for &mut DisjointSlice<'_, T, CollectiveTiles<C>> {
    #[inline(always)]
    fn load<const ITEMS: usize, const ALIGN: usize, A: Algorithm>(
        self,
        _group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T, A>,
    ) where
        Alignment<ALIGN>: SupportedAlignment,
    {
        let valid = match options.valid_items {
            Some(count) => count,
            None => boundary::tile::<ITEMS>(),
        };
        boundary::containment::<C>(options.offset, valid);
        let pointer = self.as_mut_ptr().cast_const();
        let len = self.len();
        // SAFETY: the pointer and witness share an allocation; containment was
        // checked. The compiler requires completion after conflicting accesses.
        unsafe {
            boundary::load(&*self, pointer, len, output, options);
        }
    }
}

/// A sealed per-thread value accepted by [`store`] for type `T` and algorithm `A`.
///
/// Numeric scalars and mutable [`ThreadData`] borrows support both algorithms.
/// Shared payload borrows support only [`Direct`]. A scalar contributes one item
/// per thread. Direct preserves every value and bit; [`Transpose`] mutates their
/// positions as described by [`store`].
pub trait StoreValue<T: Numeric, A: Algorithm = Direct>: sealed::StoreValue<T, A> {
    #[doc(hidden)]
    fn store<const CAPACITY: usize>(
        self,
        group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
        options: StoreOptions<A>,
    );
}

impl<T: Numeric, const I: usize, const A: usize> sealed::StoreValue<T, Direct>
    for &ThreadData<T, I, A>
where
    Alignment<A>: SupportedAlignment,
{
}
impl<T: Numeric, const I: usize, const A: usize> StoreValue<T> for &ThreadData<T, I, A>
where
    Alignment<A>: SupportedAlignment,
{
    #[inline(always)]
    fn store<const C: usize>(
        self,
        _group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions,
    ) {
        boundary::store(destination, self, options);
    }
}
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
    fn store<const C: usize>(
        self,
        _group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions<A>,
    ) {
        if A::TRANSPOSE {
            boundary::store_transpose(destination, self, options.algorithm(Transpose));
        } else {
            boundary::store(destination, self, options.algorithm(Direct));
        }
    }
}
impl<T: Numeric, A: Algorithm> sealed::StoreValue<T, A> for T {}
impl<T: Numeric, A: Algorithm> StoreValue<T, A> for T {
    #[inline(always)]
    fn store<const C: usize>(
        self,
        group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions<A>,
    ) {
        let mut value = ThreadData::<_, 1>::new();
        value.write(0, self);
        (&mut value).store(group, destination, options);
    }
}

mod sealed {
    pub trait LoadSource {}
    pub trait StoreValue<T, A> {}
    pub trait Algorithm {
        const TRANSPOSE: bool;
    }
}
