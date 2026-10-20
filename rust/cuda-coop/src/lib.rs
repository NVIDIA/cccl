// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Checked per-thread payloads for Rust `cuda.coop`.
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
//! The crate requires Rust 1.85 or newer and has no host compiler or CUDA
//! dependencies. Oxide is the intended device compiler integration. This
//! payload API supplies no collective operations, group descriptors, scratch
//! storage, or synchronization guarantees.

#![no_std]

mod alignment;
mod thread_data;

pub use alignment::{Alignment, SupportedAlignment};
pub use thread_data::{Numeric, ThreadData, ThreadDataLike};
