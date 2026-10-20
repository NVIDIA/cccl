// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut};

use crate::{Alignment, SupportedAlignment};

/// A numeric element supported by [`ThreadData`].
///
/// This sealed trait is implemented for exactly `i8`, `u8`, `i16`, `u16`, `i32`,
/// `u32`, `i64`, `u64`, `f32`, and `f64`. Values retain their Rust type and width.
pub trait Numeric: sealed::Numeric + Copy {}

/// An owned, fixed-size payload whose slots are initialized independently.
///
/// The complete signature is `ThreadData<T, const ITEMS: usize,
/// const ALIGN: usize = 1>`, where `T: Numeric` and
/// `Alignment<ALIGN>: SupportedAlignment`. `ITEMS` must be positive. `T` can be
/// inferred from typed writes; [`new`](Self::new) does not initialize values.
/// `ALIGN` specifies the minimum byte alignment of the actual element storage,
/// in addition to `T`'s natural alignment. The default of 1 adds no alignment
/// requirement. See [`Alignment`] for supported values.
///
/// [`write`](Self::write) initializes or replaces a single slot.
/// [`get`](Self::get) and [`get_mut`](Self::get_mut) return `None` for an unwritten
/// or out-of-bounds slot. Indexing requires an initialized, in-bounds slot and
/// panics otherwise; use `write`, not indexed assignment, for initialization.
/// [`len`](Self::len) counts all slots, including unwritten ones. Copies, moves,
/// and selections carry the values and each slot's validity together.
///
/// The values are contiguous, with private validity metadata. There is no
/// unrestricted initialized slice or public foreign-function layout contract.
/// This per-thread value performs no allocation or synchronization.
///
/// ```
/// use cuda_coop as coop;
/// let mut items = coop::ThreadData::<_, 4>::new();
/// items.write(3, 42_u64);
/// assert_eq!(items.get(0), None);
/// assert_eq!(items.get(3), Some(42_u64));
/// assert_eq!(items.len(), 4);
/// let copy = items;
/// assert_eq!(copy.get(3), Some(42_u64));
/// ```
///
/// Unsupported element types and alignments fail to compile:
///
/// ```compile_fail,E0277
/// let _ = cuda_coop::ThreadData::<bool, 4>::new();
/// ```
///
/// ```compile_fail,E0277
/// let _ = cuda_coop::ThreadData::<i32, 4, 3>::new();
/// ```
///
/// ```compile_fail,E0277
/// let _ = cuda_coop::ThreadData::<i32, 4, 0>::new();
/// ```
///
/// ```compile_fail,E0277
/// let _ = cuda_coop::ThreadData::<i32, 4, 1073741824>::new();
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ThreadData<T: Numeric, const ITEMS: usize, const ALIGN: usize = 1>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    // In repr(C), this zero-length field aligns the following values array
    // without moving it away from offset zero.
    alignment: [<Alignment<ALIGN> as crate::alignment::sealed::Alignment>::Marker; 0],
    values: [MaybeUninit<T>; ITEMS],
    // Packed low bits in the first ITEMS.div_ceil(8) bytes. The backing extent
    // avoids generic const expressions and preserves this private layout size.
    initialized: [u8; ITEMS],
}

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    const POSITIVE_EXTENT: () = assert!(ITEMS > 0, "ThreadData requires at least one item");

    /// Construct a payload with every slot unwritten.
    ///
    /// Instantiating this constructor with zero items fails constant evaluation:
    ///
    /// ```compile_fail,E0080
    /// const EMPTY: cuda_coop::ThreadData<i32, 0> = cuda_coop::ThreadData::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        let () = Self::POSITIVE_EXTENT;
        Self {
            alignment: [],
            values: [MaybeUninit::uninit(); ITEMS],
            initialized: [0; ITEMS],
        }
    }

    /// Return the static slot count, including unwritten slots.
    #[inline]
    pub const fn len(&self) -> usize {
        ITEMS
    }

    /// Return whether the payload has zero slots.
    ///
    /// This is always `false` for a successfully constructed payload.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        ITEMS == 0
    }

    /// Initialize or replace slot `index` with `value`, leaving other slots alone.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn write(&mut self, index: usize, value: T) {
        self.values[index].write(value);
        self.initialized[index / 8] |= 1 << (index % 8);
    }

    /// Copy an initialized value, or return `None` if unwritten or out of bounds.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= ITEMS {
            return None;
        }
        if (self.initialized[index / 8] & (1 << (index % 8))) != 0 {
            // SAFETY: Only write sets this bit, after initializing the value.
            // Copies preserve both arrays. Numeric types are Copy.
            Some(unsafe { self.values[index].assume_init() })
        } else {
            None
        }
    }

    /// Borrow an initialized value mutably, or return `None` if unwritten or
    /// out of bounds. The exclusive borrow prevents simultaneous payload access.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < ITEMS && (self.initialized[index / 8] & (1 << (index % 8))) != 0 {
            // SAFETY: The bit proves initialization, and &mut self is exclusive.
            Some(unsafe { self.values[index].assume_init_mut() })
        } else {
            None
        }
    }
}

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> Default for ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> Index<usize>
    for ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        assert!(index < ITEMS);
        assert!((self.initialized[index / 8] & (1 << (index % 8))) != 0);
        // SAFETY: The bounds and initialization bit were checked above.
        unsafe { self.values[index].assume_init_ref() }
    }
}

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> IndexMut<usize>
    for ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(index < ITEMS);
        assert!((self.initialized[index / 8] & (1 << (index % 8))) != 0);
        // SAFETY: The checks prove initialization; &mut self is exclusive.
        unsafe { self.values[index].assume_init_mut() }
    }
}

/// A sealed interface for generic access to fixed-size per-thread payloads.
///
/// Implemented by [`ThreadData`] for every supported element type, extent and
/// alignment. Consumers can write generic helpers without assuming every slot
/// is initialized. User implementations are not supported.
///
/// ```
/// use cuda_coop as coop;
/// fn copy_first<D: coop::ThreadDataLike>(items: &mut D) {
///     if let Some(value) = items.get(0) {
///         items.write(D::EXTENT - 1, value);
///     }
/// }
/// let mut items = coop::ThreadData::<_, 4>::new();
/// items.write(0, 3_i8);
/// copy_first(&mut items);
/// assert_eq!(items.get(3), Some(3_i8));
/// assert_eq!(items.get(1), None);
/// ```
pub trait ThreadDataLike: sealed::ThreadDataLike {
    /// The numeric element type.
    type Item: Numeric;

    /// The compile-time slot count, including unwritten slots.
    const EXTENT: usize;

    /// Return the static slot count, including unwritten slots.
    #[inline]
    fn len(&self) -> usize {
        Self::EXTENT
    }

    /// Return whether the payload has zero slots.
    #[inline]
    fn is_empty(&self) -> bool {
        Self::EXTENT == 0
    }

    /// Copy an initialized value, or return `None` if unwritten or out of bounds.
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Initialize or replace a single slot, preserving all other slots.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    fn write(&mut self, index: usize, value: Self::Item);
}

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> ThreadDataLike
    for ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
    type Item = T;
    const EXTENT: usize = ITEMS;

    #[inline]
    fn get(&self, index: usize) -> Option<T> {
        self.get(index)
    }

    #[inline]
    fn write(&mut self, index: usize, value: T) {
        self.write(index, value);
    }
}

mod sealed {
    pub trait Numeric {}
    pub trait ThreadDataLike {}
}

macro_rules! numeric {
    ($($ty:ty),* $(,)?) => {
        $(
            impl sealed::Numeric for $ty {}
            impl Numeric for $ty {}
        )*
    };
}

numeric!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

impl<T: Numeric, const ITEMS: usize, const ALIGN: usize> sealed::ThreadDataLike
    for ThreadData<T, ITEMS, ALIGN>
where
    Alignment<ALIGN>: SupportedAlignment,
{
}
