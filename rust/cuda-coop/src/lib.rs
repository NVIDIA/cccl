// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Per-thread payloads and CUDA hierarchy queries for Rust `cuda.coop`.
//!
//! This unpublished, feature-free `no_std` crate provides [`ThreadData`] and
//! its sealed [`ThreadDataLike`] interface. Each slot starts unwritten.
//! [`ThreadData::write`] initializes one slot; checked access exposes only
//! initialized values. Copies and moves preserve each slot's validity.
//!
//! ```
//! use cuda_coop as coop;
//!
//! fn payload<const ITEMS_PER_THREAD: usize>(value: i16)
//!     -> coop::ThreadData<i16, ITEMS_PER_THREAD>
//! {
//!     let mut items = coop::ThreadData::<_, ITEMS_PER_THREAD>::new();
//!     items.write(0, value); // Infers i16 without changing its width.
//!     items
//! }
//!
//! let mut items = payload::<4>(7);
//! assert_eq!(items.get(0), Some(7));
//! assert_eq!(items.get(1), None);
//! items[0] += 2; // Indexing can mutate an already initialized slot.
//! assert_eq!(items[0], 9);
//! ```
//!
//! Device queries use Oxide and the pinned `1.99.1` toolchain.
//! Plain Cargo commands also require `RUSTC_BOOTSTRAP=1`, set by this workspace's
//! `.cargo/config.toml`; standalone consumers must supply that setting themselves.
//! [`this_block`] and the other dimension-free group constructors describe
//! the executing thread's current launch. Queries grant no collective
//! participation or synchronization guarantee. [`load`] and [`store`] support
//! Direct, Striped, Vectorize and three transpose algorithms on complete blocks,
//! and Direct on physical Warp groups, with compiler-checked participation and
//! collective memory ownership. The transpose algorithms use compiler-planned
//! shared scratch. [`TempStorage`] selects explicit Block reuse and completion
//! policies; [`StorageReservation`] exposes typed scratch through raw pointers.
//! Host payload tests need no GPU;
//! executing hardware queries requires an Oxide-compiled CUDA kernel.

#![no_std]

mod alignment;
mod group;
mod load_store;
mod temp_storage;
mod thread_data;

pub use alignment::{Alignment, SupportedAlignment};
pub use cuda_device::{CollectiveTiles, DisjointSlice};
pub use load_store::{
    Algorithm, Direct, LoadOptions, LoadSource, LoadStoreGroup, StoreOptions, StoreValue, Striped,
    Transpose, Vectorize, WarpTranspose, WarpTransposeTimesliced, load, store,
};
pub use temp_storage::{StorageReservation, TempStorage, TempStorageLike};
pub use thread_data::{Numeric, ThreadData, ThreadDataLike};

pub use group::{
    Block, Cluster, Grid, GroupScope, Hierarchy, Level, QueryAt, QueryInteger, Thread, ThreadGroup,
    ThreadHierarchy, ThreadsWithinWarp, Warp, WarpsWithinBlock, this_block, this_cluster,
    this_grid, this_thread, this_warp,
};
