// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{Alignment, Block, Numeric, SupportedAlignment, ThreadData, ThreadGroup};
use cuda_device::{CollectiveTiles, DisjointSlice};

mod boundary;

/// Controls for a Direct [`load`].
///
/// `T` is inferred from the source and output. Defaults load the entire group
/// tile, leave no unspecified slots, and use element offset zero. A partial
/// load without an out-of-bounds default makes every slot outside its prefix
/// unwritten, including slots initialized before the call.
///
/// Controls, including whether a count/default is present, must be uniform
/// across the block. They are checked when the operation executes.
#[derive(Clone, Copy)]
pub struct LoadOptions<T> {
    valid_items: Option<usize>,
    oob_default: Option<T>,
    offset: usize,
}

impl<T> Default for LoadOptions<T> {
    fn default() -> Self {
        Self {
            valid_items: None,
            oob_default: None,
            offset: 0,
        }
    }
}

impl<T> LoadOptions<T> {
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

/// Controls for a Direct [`store`].
///
/// Defaults write the entire group tile at element offset zero. A partial
/// store leaves destination elements outside its prefix unchanged. Controls,
/// including whether a count is present, must be uniform across the block.
#[derive(Clone, Copy, Default)]
pub struct StoreOptions {
    valid_items: Option<usize>,
    offset: usize,
}

impl StoreOptions {
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
/// The operation returns unit and uses no shared scratch.
///
/// Only complete one-dimensional blocks and the Direct algorithm are supported.
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
pub fn load<T: Numeric, S: LoadSource<T>, const ITEMS: usize, const ALIGN: usize>(
    group: ThreadGroup<Block>,
    source: S,
    output: &mut ThreadData<T, ITEMS, ALIGN>,
    options: LoadOptions<T>,
) where
    Alignment<ALIGN>: SupportedAlignment,
{
    source.load(group, output, options);
}

/// Store a block tile in blocked order, preserving the per-thread payload.
///
/// `T` is inferred from the destination and value. [`StoreValue`] accepts a
/// shared or mutable borrow of [`ThreadData`], or a numeric scalar (one item per
/// thread). `CAPACITY` describes the carrier's ownership grain; it need not equal
/// the payload extent. This Direct operation borrows payload values and flags
/// read-only, returns unit, and uses no shared scratch.
///
/// The supported groups, compiler/launch and uniformity requirements are those
/// of [`load`]. The destination is the canonical collective carrier: it grants
/// disjoint block regions without exposing duplicate mutable lane references.
/// Every written payload slot must be initialized. Slots outside the written
/// prefix may be unwritten and are preserved. A following conflicting access
/// to this carrier needs explicit block completion; Drop is not completion.
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
pub fn store<T: Numeric, V: StoreValue<T>, const CAPACITY: usize>(
    group: ThreadGroup<Block>,
    destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
    value: V,
    options: StoreOptions,
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
    fn load<const ITEMS: usize, const ALIGN: usize>(
        self,
        group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T>,
    ) where
        Alignment<ALIGN>: SupportedAlignment;
}

impl<T: Numeric> sealed::LoadSource for &[T] {}
impl<T: Numeric> LoadSource<T> for &[T] {
    #[inline(always)]
    fn load<const ITEMS: usize, const ALIGN: usize>(
        self,
        _group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T>,
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
    fn load<const ITEMS: usize, const ALIGN: usize>(
        self,
        _group: ThreadGroup<Block>,
        output: &mut ThreadData<T, ITEMS, ALIGN>,
        options: LoadOptions<T>,
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

/// A sealed per-thread value accepted by Direct [`store`] for numeric type `T`.
///
/// Implemented for numeric scalars and shared/mutable borrows of [`ThreadData`].
/// A scalar contributes one item per thread. Payload borrows preserve every
/// value and flag, including slots outside the selected write domain.
pub trait StoreValue<T: Numeric>: sealed::StoreValue<T> {
    #[doc(hidden)]
    fn store<const CAPACITY: usize>(
        self,
        group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<CAPACITY>>,
        options: StoreOptions,
    );
}

impl<T: Numeric, const I: usize, const A: usize> sealed::StoreValue<T> for &ThreadData<T, I, A> where
    Alignment<A>: SupportedAlignment
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
impl<T: Numeric, const I: usize, const A: usize> sealed::StoreValue<T> for &mut ThreadData<T, I, A> where
    Alignment<A>: SupportedAlignment
{
}
impl<T: Numeric, const I: usize, const A: usize> StoreValue<T> for &mut ThreadData<T, I, A>
where
    Alignment<A>: SupportedAlignment,
{
    #[inline(always)]
    fn store<const C: usize>(
        self,
        group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions,
    ) {
        (&*self).store(group, destination, options);
    }
}
impl<T: Numeric> sealed::StoreValue<T> for T {}
impl<T: Numeric> StoreValue<T> for T {
    #[inline(always)]
    fn store<const C: usize>(
        self,
        group: ThreadGroup<Block>,
        destination: &mut DisjointSlice<'_, T, CollectiveTiles<C>>,
        options: StoreOptions,
    ) {
        let mut value = ThreadData::<_, 1>::new();
        value.write(0, self);
        (&value).store(group, destination, options);
    }
}

mod sealed {
    pub trait LoadSource {}
    pub trait StoreValue<T> {}
}
