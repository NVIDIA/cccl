// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::Numeric;
use cuda_device::library;

/// Compiler-planned shared scratch for complete-block operations.
///
/// Construct this descriptor inside a supported Oxide kernel and pass a mutable
/// borrow through [`crate::LoadOptions::temp_storage`] or
/// [`crate::StoreOptions::temp_storage`]. The default shares one slice between
/// uses and requires explicit block synchronization before reuse. Omitting the
/// descriptor instead requests compiler-managed reuse synchronization.
///
/// `CAPACITY` and `ALIGN` default to zero, meaning inferred. Consuming builders
/// select positive explicit values and synchronization/sharing policies before
/// use. Capacity covers the entire plan, including exclusive slices and padding.
/// Each static constructor site denotes one descriptor, including in a loop.
/// Different sites have independent storage. Exclusive mode distinguishes
/// static use sites, not repeated executions of one site in a loop.
///
/// This is neither a shared-memory reference nor a runtime allocation. All block
/// lanes must agree on the descriptor and reach matching operations; the compiler
/// checks participation, origin and reuse. Dropping it does not synchronize.
#[must_use]
pub struct TempStorage<
    const CAPACITY: usize = 0,
    const ALIGN: usize = 0,
    const AUTO_SYNC: bool = false,
    const EXCLUSIVE: bool = false,
> {
    raw: library::StorageDescriptor<CAPACITY, ALIGN, AUTO_SYNC, EXCLUSIVE>,
}

impl TempStorage {
    /// Request inferred shared storage with automatic synchronization disabled.
    #[inline(always)]
    pub fn new() -> Self {
        // SAFETY: the supported compiler validates uniform construction and
        // retains the actual rooted constructor identity.
        Self {
            raw: unsafe { library::storage_new_v1::<0, 0, false, false>() },
        }
    }
}

impl Default for TempStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl<const C: usize, const A: usize, const S: bool, const E: bool> TempStorage<C, A, S, E> {
    /// Set a positive byte capacity for the complete descriptor plan.
    ///
    /// Undersized capacity, including exclusive-site padding, is a compilation
    /// error. For a scratch-using plan, the compiler reserves the requested
    /// capacity when it is sufficient. Scratch-free uses allocate nothing.
    #[inline(always)]
    pub fn with_capacity<const BYTES: usize>(self) -> TempStorage<BYTES, A, S, E> {
        const { assert!(BYTES > 0 && BYTES <= u32::MAX as usize) };
        // SAFETY: both types are genuine descriptors; the compiler preserves
        // this origin and rejects policy changes after use.
        TempStorage {
            raw: unsafe { library::storage_reconfigure_v1(self.raw) },
        }
    }

    /// Set a positive power-of-two minimum alignment for the descriptor base.
    ///
    /// Each exclusive slice also satisfies its own operation/reservation
    /// alignment. This base request does not change every slice's alignment.
    #[inline(always)]
    pub fn with_alignment<const ALIGN: usize>(self) -> TempStorage<C, ALIGN, S, E> {
        const { assert!(ALIGN.is_power_of_two() && ALIGN <= u32::MAX as usize) };
        // SAFETY: consuming genuine descriptors retains the compiler origin.
        TempStorage {
            raw: unsafe { library::storage_reconfigure_v1(self.raw) },
        }
    }

    /// Insert a trailing block barrier after each scratch-using operation.
    ///
    /// Scratch-free operations such as Direct Load/Store gain no barrier.
    /// Typed reservations are unavailable on an auto-sync descriptor.
    #[inline(always)]
    pub fn auto_sync(self) -> TempStorage<C, A, true, E> {
        // SAFETY: the compiler checks policy changes and actual scratch use.
        TempStorage {
            raw: unsafe { library::storage_reconfigure_v1(self.raw) },
        }
    }

    /// Give distinct static primitive and reservation sites separate slices.
    ///
    /// Repeating one site still reuses its slice and requires completion.
    /// Independent scratch does not complete conflicting global-memory accesses.
    #[inline(always)]
    pub fn exclusive(self) -> TempStorage<C, A, S, true> {
        // SAFETY: the compiler preserves the origin and plans distinct sites.
        TempStorage {
            raw: unsafe { library::storage_reconfigure_v1(self.raw) },
        }
    }
}

impl<const C: usize, const A: usize, const E: bool> TempStorage<C, A, false, E> {
    /// Reserve `N` numeric elements with their natural alignment.
    ///
    /// `N` must be positive. The view is uninitialized and exposes only a raw
    /// pointer; see [`StorageReservation`] for access and lifetime obligations.
    /// Reservations follow this descriptor's shared/exclusive allocation policy.
    #[inline(always)]
    pub fn reserve<T: Numeric, const N: usize>(&mut self) -> StorageReservation<T, N> {
        const { assert!(N > 0) };
        // SAFETY: the compiler checks participation, origin, live overlap and
        // layout. Returning a raw token performs no access to an element.
        StorageReservation {
            raw: unsafe { library::storage_reserve_v1::<T, _, N, 0>(&mut self.raw) },
        }
    }

    /// Reserve `N` numeric elements with a power-of-two minimum alignment.
    ///
    /// `N` and `ALIGN` must be positive. The requested alignment applies to this
    /// reservation, including an exclusive slice at a nonzero offset.
    #[inline(always)]
    pub fn reserve_aligned<T: Numeric, const N: usize, const ALIGN: usize>(
        &mut self,
    ) -> StorageReservation<T, N> {
        const {
            assert!(N > 0);
            assert!(ALIGN.is_power_of_two() && ALIGN <= u32::MAX as usize);
        }
        // SAFETY: the compiler validates the genuine descriptor and typed
        // layout. Actual raw memory accesses retain their unsafe obligations.
        StorageReservation {
            raw: unsafe { library::storage_reserve_v1::<T, _, N, ALIGN>(&mut self.raw) },
        }
    }
}

/// A numeric shared-memory view with explicitly unsafe element access.
///
/// This non-Copy token grants no initialized value, Rust reference or slice.
/// It does not retain an exclusive Rust borrow of its descriptor, so separate
/// exclusive reservations may remain live together. The supported compiler
/// checks view origins and rejects overlapping planned reuse while a view is live.
/// It does not prove arbitrary unsafe pointer operations.
///
/// The allocation belongs to the executing block for this kernel execution.
/// Every raw access must prove bounds, initialization before reads, aliasing,
/// non-conflicting accesses and phase synchronization. Complete asynchronous
/// accesses before ending their view. Saved pointers must not be dereferenced
/// after [`release`](Self::release), during overlapping reuse, or after kernel
/// exit. Copies of a pointer do not extend the view's lifetime.
///
/// Dropping or forgetting the token does not release it or synchronize.
#[must_use = "release the reservation before overlapping storage reuse"]
pub struct StorageReservation<T: Numeric, const N: usize> {
    raw: library::StorageReservation<T, N>,
}

impl<T: Numeric, const N: usize> StorageReservation<T, N> {
    /// Obtain the raw pointer without creating a reference or reading an element.
    ///
    /// Dereferencing it requires the obligations in [`StorageReservation`].
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.raw.as_mut_ptr()
    }

    /// End the view without completing shared or global-memory accesses.
    ///
    /// Synchronize the full block **after** release and before overlapping
    /// reuse. A barrier before release is insufficient. No saved raw pointer
    /// may be accessed afterward; only unsafe code could perform such an access.
    #[inline(always)]
    pub fn release(self) {
        // SAFETY: the compiler validates uniform release of a live view. The
        // safe API grants no reference or access; saved raw pointers remain
        // subject to the caller's unsafe lifetime and completion obligations.
        unsafe { library::storage_release_v1(self.raw) }
    }
}

/// A sealed storage argument carried by Load/Store options.
///
/// `()` selects compiler-managed scratch; a mutable [`TempStorage`] borrow
/// selects explicit storage. Operations consume options without requiring Copy.
pub trait TempStorageLike: sealed::Storage {}

impl TempStorageLike for () {}
impl sealed::Storage for () {
    type Raw = ();
    fn into_raw(self) {}
}

impl<const C: usize, const A: usize, const S: bool, const E: bool> TempStorageLike
    for &mut TempStorage<C, A, S, E>
{
}
impl<'a, const C: usize, const A: usize, const S: bool, const E: bool> sealed::Storage
    for &'a mut TempStorage<C, A, S, E>
{
    type Raw = &'a mut library::StorageDescriptor<C, A, S, E>;

    #[inline(always)]
    fn into_raw(self) -> Self::Raw {
        &mut self.raw
    }
}

pub(crate) mod sealed {
    pub trait Storage {
        type Raw;
        fn into_raw(self) -> Self::Raw;
    }
}
