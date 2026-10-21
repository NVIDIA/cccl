// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Per-thread payloads and CUDA hierarchy queries for Rust `cuda.coop`.
//!
//! This unpublished `no_std` crate provides [`ThreadData`] and
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
//! Device queries use Oxide and the pinned `nightly-2026-10-11` toolchain.
//! [`this_block`] and the other dimension-free group constructors describe
//! the executing thread's current launch. Queries grant no collective
//! participation or synchronization guarantee. [`load`] and [`store`] support
//! Direct, Striped, Vectorize and three transpose algorithms on complete blocks.
//! Physical Warp groups and logical widths 1, 2, 4, 8, 16 and 32 support Direct,
//! Striped, Vectorize and Transpose. These operations have compiler-checked
//! participation and collective memory ownership. The transpose algorithms use
//! compiler-planned shared scratch. [`TempStorage`] selects explicit Block reuse
//! and completion policies; [`StorageReservation`] exposes typed scratch through
//! raw pointers.
//! [`scan`] and its four named forms return fresh numeric prefixes. Blocks
//! accept scalars or borrowed payloads; physical/logical Warp accepts scalars.
//! Mode, built-in operator and Block algorithm are selected statically through
//! [`ScanOptions`], with exclusive sum as the default. Scan preserves its input
//! and uses the same compiler-checked scratch and participation rules.
//! [`reduce`] and [`sum`] return `Some(total)` only at group rank zero, with
//! `None` at other participants. They accept scalars or borrowed payloads on
//! Blocks and Warp groups, including guarded non-exhaustive widths 17–31.
//! [`ReduceOptions`] selects the operator, Block algorithm, scalar count and
//! storage without changing the input.
//! [`reduce_batched`] independently reduces each payload slot across physical or
//! power-of-two logical Warp groups. Its fresh partial payload has a calculated
//! extent and explicit striped or blocked result ownership. The defining crate
//! and callers instantiating these calculated signatures enable nightly
//! `generic_const_exprs`; ordinary non-calculated consumers remain feature-free.
//! [`exchange`] converts between blocked and striped layouts on complete Blocks
//! and physical or power-of-two logical Warp groups. It preserves input and
//! returns a fresh payload with the same type, extent and alignment.
//! Host payload tests need no GPU;
//! executing hardware queries requires an Oxide-compiled CUDA kernel.

#![no_std]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod alignment;
mod exchange;
mod group;
mod load_store;
mod reduce;
mod reduce_batched;
mod scan;
mod temp_storage;
mod thread_data;

pub use alignment::{Alignment, SupportedAlignment};
pub use cuda_device::{CollectiveTiles, DisjointSlice};
pub use exchange::{
    BlockedToStriped, ExchangeGroup, ExchangeMode, ExchangeOptions, StripedToBlocked, exchange,
};
pub use load_store::{
    Algorithm, Direct, LoadOptions, LoadSource, LoadStoreGroup, StoreOptions, StoreValue, Striped,
    Transpose, Vectorize, WarpTranspose, WarpTransposeTimesliced, load, store,
};
pub use reduce::{
    RakingCommutativeOnly, ReduceAlgorithm, ReduceGroup, ReduceOperator, ReduceOptions,
    ReduceValue, SumOptions, WarpReductions, reduce, sum,
};
pub use reduce_batched::{
    BatchedOutputLayout, Blocked, ReduceBatchedGroup, ReduceBatchedOptions, batched_extent,
    reduce_batched,
};
pub use scan::{
    BitAnd, BitOr, BitXor, Exclusive, ExclusiveScanOptions, Inclusive, InclusiveScanOptions, Max,
    Min, Multiplies, Raking, RakingMemoize, ScanAlgorithm, ScanGroup, ScanMode, ScanOperator,
    ScanOptions, ScanValue, Sum, SumScanOptions, WarpScans, exclusive_scan, exclusive_sum,
    inclusive_scan, inclusive_sum, scan,
};
pub use temp_storage::{StorageReservation, TempStorage, TempStorageLike};
pub use thread_data::{Numeric, ThreadData, ThreadDataLike};

pub use group::{
    Block, Cluster, Grid, GroupScope, Hierarchy, Level, QueryAt, QueryInteger, Thread, ThreadGroup,
    ThreadHierarchy, ThreadsWithinWarp, Warp, WarpsWithinBlock, this_block, this_cluster,
    this_grid, this_thread, this_warp,
};
