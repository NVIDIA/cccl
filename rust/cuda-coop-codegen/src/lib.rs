// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Native, host-only CUB specialization for Rust cooperative algorithms.
//!
//! This private development crate compiles Load/Store, Scan, Reduce, batched
//! Reduce, Exchange, Shuffle and Adjacent Difference with NVRTC. It returns
//! linkable LTO-IR and the exact C++ scratch layout from the
//! **same compilation**, without a CUDA context, GPU or Python dependency.
//! It is a compiler building block, not a device API or a package release.
//!
//! # Supported specializations
//!
//! [`Request`] describes an exact complete block shape, positive fixed
//! items per thread, and [`Algorithm::Direct`], [`Algorithm::Striped`],
//! [`Algorithm::Vectorize`], [`Algorithm::Transpose`], [`Algorithm::WarpTranspose`]
//! or [`Algorithm::WarpTransposeTimesliced`]. The two warp-transpose algorithms
//! require a total block thread count divisible by 32.
//! The raw i32 recipes support all six algorithms. Their Store payloads must
//! be fully initialized. The validity-aware recipes support all ten
//! [`Dtype`] variants with separate value storage and packed per-slot validity bits.
//! Masked Load resets validity outside its prefix; Direct/Striped/Vectorize Store preserves both
//! arrays and checks initialization only in its write domain. Transposing Store
//! replaces each lane's blocked payload and validity with striped items across
//! the block or selected Warp, including when its write count is zero. Full Vectorize operations
//! use numeric local arrays after full initialization or complete Store preflight;
//! counted Vectorize overloads retain the opaque Direct path.
//! Physical and logical Warp requests support Direct, Striped, Vectorize and
//! Transpose at power-of-two widths from 1 through 32, within complete physical
//! warps. Their scratch layout describes one group, not the complete block.
//!
//! [`ScanRequest`] selects a separate numeric Scan family: all ten dtypes, seven
//! built-in operators (bitwise for integers only), inclusive or exclusive output,
//! and Raking/RakingMemoize/WarpScans for Blocks. Warp Scan accepts one scalar
//! per lane with implicit algorithm selection. Scan checks every input validity
//! bit, preserves the input, and initializes every fresh output slot. Seeded
//! exclusive forms carry one runtime value of exactly the input dtype.
//! [`ReduceRequest`] preserves scalar or fixed-payload inputs and produces one
//! initialized scalar only at group rank zero. Scalar forms optionally take a
//! nonempty valid prefix; payload forms include every item. Blocks select
//! RakingCommutativeOnly, Raking or WarpReductions. Warp reductions also support
//! guarded low-lane prefixes of widths 17..31 within each physical warp.
//! [`ReduceBatchedRequest`] reduces each payload position independently across
//! a power-of-two Warp. Its fresh per-lane output capacity is ceil(batches/width),
//! with striped or blocked distribution and validity only for real batches.
//! [`ExchangeRequest`] preserves complete numeric payloads and produces fresh
//! same-extent outputs redistributed between blocked and striped layouts. It
//! supports exact Blocks and power-of-two Warps using shared memory.
//! [`ShuffleRequest`] shifts a complete Block tile by one item, preserving input
//! and returning fresh output with exactly one invalid boundary slot.
//! [`AdjacentDifferenceRequest`] subtracts left or right neighbors in a Block
//! tile or prefix, preserving input and copying unchanged boundary/suffix items
//! into a fully initialized fresh output.
//! [`Provider::compile_native`] batches these families in one native compilation.
//!
//! [`Provider::from_checkout`] snapshots the selected CCCL and CUDA headers;
//! [`Provider::compile`] uses those exact bytes and loaded NVRTC to emit an
//! [`Artifact`]. Each returned device ABI is unsafe to call: its documented
//! participation, pointer, capacity, validity and synchronization obligations
//! belong to the compiler integration. This crate allocates no device memory,
//! links no final kernel and supplies no launch policy.
//!
//! # Host example
//!
//! ```no_run
//! use cuda_coop_codegen::{Algorithm, Operation, Provider, Request};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = Provider::from_checkout("/path/to/cccl", "/usr/local/cuda")?;
//! let request = Request::new(Operation::LoadFull, Algorithm::Transpose, 128, 4)?;
//! let artifact = provider.compile(&[request], "compute_120")?;
//! std::fs::write("block_load.ltoir", &artifact.ltoir)?;
//! println!("{}: {:?}", artifact.specializations[0].symbol, artifact.specializations[0].scratch);
//! # Ok(()) }
//! ```
//!
//! The example compiles on the host; executing it requires a compatible Linux
//! CUDA toolkit. The `specialize` example writes the artifact and prints its
//! identity/layout. `cargo doc --no-deps` generates this API reference.

#![deny(unsafe_code)]

mod adjacent_difference;
mod exchange;
mod headers;
mod reduce;
mod reduce_batched;
mod scan;
mod shuffle;
pub use adjacent_difference::{AdjacentDifferenceRecipe, AdjacentDifferenceRequest};
pub use exchange::{ExchangeMode, ExchangeRecipe, ExchangeRequest};
pub use reduce::{ReduceAlgorithm, ReduceForm, ReduceOperator, ReduceRecipe, ReduceRequest};
pub use reduce_batched::{ReduceBatchedLayout, ReduceBatchedRecipe, ReduceBatchedRequest};
pub use scan::{ScanAlgorithm, ScanOperation, ScanOperator, ScanRecipe, ScanRequest};
pub use shuffle::{ShuffleRecipe, ShuffleRequest};
#[allow(unsafe_code)]
mod nvrtc;
pub mod oxide;

use headers::{Headers, hash_field};
use sha2::{Digest, Sha256};
use std::{fmt, path::Path};

/// A provider input, loader or compiler error, including NVRTC's compile log.
#[derive(Debug)]
pub struct Error(String);
impl Error {
    pub(crate) fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
    pub(crate) fn input(path: &Path, error: std::io::Error) -> Self {
        Self(format!("cannot read {}: {error}", path.display()))
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}
impl From<std::ffi::NulError> for Error {
    fn from(error: std::ffi::NulError) -> Self {
        Self(error.to_string())
    }
}
/// Result returned by native provider operations.
pub type Result<T> = std::result::Result<T, Error>;

/// CUB block layout algorithm and per-thread item arrangement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Algorithm {
    /// Consecutive items per thread, without shared scratch or a reuse barrier.
    Direct,
    /// Thread `t` accesses item `i` at `t + group_threads * i`, without scratch.
    Striped,
    /// Blocked access with aligned vector operations for full tiles.
    /// Misaligned pointers and counted overloads use Direct accesses.
    Vectorize,
    /// Coalesced access with shared-memory transposition to/from blocked layout.
    Transpose,
    /// Coalesced access with a separate shared exchange for each physical warp.
    /// Requires a block thread count divisible by 32.
    WarpTranspose,
    /// Warp transposition that reuses one physical warp's CUB exchange storage.
    /// Requires a block thread count divisible by 32.
    WarpTransposeTimesliced,
}
impl Algorithm {
    fn scratch_free(self) -> bool {
        matches!(self, Self::Direct | Self::Striped | Self::Vectorize)
    }

    fn token(self) -> &'static str {
        match self {
            Self::Direct => "DIRECT",
            Self::Striped => "STRIPED",
            Self::Vectorize => "VECTORIZE",
            Self::Transpose => "TRANSPOSE",
            Self::WarpTranspose => "WARP_TRANSPOSE",
            Self::WarpTransposeTimesliced => "WARP_TRANSPOSE_TIMESLICED",
        }
    }
}

/// Exact scalar type used in a native specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    /// Rust `i8`, C++ `signed char`.
    I8,
    /// Rust `u8`, C++ `unsigned char`.
    U8,
    /// Rust `i16`, C++ `short`.
    I16,
    /// Rust `u16`, C++ `unsigned short`.
    U16,
    /// Rust `i32`, C++ `int`.
    I32,
    /// Rust `u32`, C++ `unsigned int`.
    U32,
    /// Rust `i64`, C++ `long long`.
    I64,
    /// Rust `u64`, C++ `unsigned long long`.
    U64,
    /// Rust `f32`, C++ `float`.
    F32,
    /// Rust `f64`, C++ `double`.
    F64,
}
impl Dtype {
    fn native(self) -> (&'static str, usize) {
        match self {
            Self::I8 => ("signed char", 1),
            Self::U8 => ("unsigned char", 1),
            Self::I16 => ("short", 2),
            Self::U16 => ("unsigned short", 2),
            Self::I32 => ("int", 4),
            Self::U32 => ("unsigned int", 4),
            Self::I64 => ("long long", 8),
            Self::U64 => ("unsigned long long", 8),
            Self::F32 => ("float", 4),
            Self::F64 => ("double", 8),
        }
    }
}

/// Device operation and its validity controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    /// Load every slot. ABI: `(const int* memory, int* payload, unsigned char* scratch)`.
    LoadFull,
    /// Load a valid prefix and fill its suffix. ABI adds `(int valid, int default_value)`.
    LoadFilled,
    /// Store every slot. ABI: `(int* memory, int* payload, unsigned char* scratch)`.
    StoreFull,
    /// Store a prefix only. ABI adds `(int valid)`; other destination elements remain unchanged.
    StorePartial,
    /// Full Load with separate value storage and packed validity bits; marks every slot initialized.
    LoadFullValidity,
    /// Counted Load without a default; clears validity bits outside the loaded prefix.
    LoadMaskedValidity,
    /// Counted Load with an exact typed default; marks every slot initialized.
    LoadFilledValidity,
    /// Full Store after checking every validity bit is set.
    /// Transposing algorithms redistribute the payload; all resulting slots remain initialized.
    StoreFullValidity,
    /// Counted Store after checking its write domain.
    /// Transposing algorithms redistribute all values and validity bits, even outside that domain.
    StorePartialValidity,
}
impl Operation {
    fn is_load(self) -> bool {
        matches!(
            self,
            Self::LoadFull
                | Self::LoadFilled
                | Self::LoadFullValidity
                | Self::LoadMaskedValidity
                | Self::LoadFilledValidity
        )
    }
    fn validity(self) -> bool {
        matches!(
            self,
            Self::LoadFullValidity
                | Self::LoadMaskedValidity
                | Self::LoadFilledValidity
                | Self::StoreFullValidity
                | Self::StorePartialValidity
        )
    }
    fn counted(self) -> bool {
        matches!(
            self,
            Self::LoadFilled
                | Self::StorePartial
                | Self::LoadMaskedValidity
                | Self::LoadFilledValidity
                | Self::StorePartialValidity
        )
    }
    fn filled(self) -> bool {
        matches!(self, Self::LoadFilled | Self::LoadFilledValidity)
    }
    fn token(self) -> &'static str {
        match self {
            Self::LoadFull => "load_full",
            Self::LoadFilled => "load_filled",
            Self::StoreFull => "store_full",
            Self::StorePartial => "store_partial",
            Self::LoadFullValidity => "load_full_validity_bits_v1",
            Self::LoadMaskedValidity => "load_masked_validity_bits_v1",
            Self::LoadFilledValidity => "load_filled_validity_bits_v1",
            Self::StoreFullValidity => "store_full_validity_bits_v1",
            Self::StorePartialValidity => "store_partial_validity_bits_v1",
        }
    }
}

/// A validated specialization for an exact one-, two- or three-dimensional block.
///
/// This is an in-process implementation interface, not a serialized compiler
/// protocol. All fields are static specialization inputs. The generated ABI
/// uses runtime `valid` and `default_value` only when selected by [`Operation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Request {
    operation: Operation,
    algorithm: Algorithm,
    dtype: Dtype,
    block: [u32; 3],
    items_per_thread: u32,
    warp_width: Option<u32>,
}
impl Request {
    /// Validate a one-dimensional block shape and payload extent.
    ///
    /// `block_threads` must be 1..=1024 and `items_per_thread` positive. Their
    /// product must fit i32, matching CUB's valid-prefix count. Target-specific
    /// compiler restrictions are diagnosed by NVRTC during [`Provider::compile`].
    ///
    /// # Errors
    ///
    /// Returns an error for zero/oversized blocks, zero item counts or a tile
    /// extent that cannot be represented by CUB's signed valid-prefix count.
    pub fn new(
        operation: Operation,
        algorithm: Algorithm,
        block_threads: u32,
        items_per_thread: u32,
    ) -> Result<Self> {
        Self::new_typed(
            Dtype::I32,
            operation,
            algorithm,
            block_threads,
            items_per_thread,
        )
    }

    /// Validate an exact typed block specialization.
    ///
    /// The five validity-aware recipes support every [`Dtype`] with every algorithm.
    /// Legacy raw recipes retain their i32 ABI and algorithm support.
    ///
    /// # Errors
    ///
    /// Rejects the invalid shapes documented by [`Self::new`], or a non-i32 raw
    /// recipe.
    pub fn new_typed(
        dtype: Dtype,
        operation: Operation,
        algorithm: Algorithm,
        block_threads: u32,
        items_per_thread: u32,
    ) -> Result<Self> {
        Self::new_typed_with_shape(
            dtype,
            operation,
            algorithm,
            [block_threads, 1, 1],
            items_per_thread,
        )
    }

    /// Validate an exact `[X, Y, Z]` block shape with i32 elements.
    ///
    /// Each axis must be positive, their product at most 1024, and the positive
    /// per-thread item count times that product must fit i32. Threads are ranked
    /// x-fastest: `x + X * (y + Y * z)`. Device-specific axis and launch limits
    /// must also be checked by the caller when preparing a launch.
    ///
    /// # Errors
    ///
    /// Rejects zero axes, an oversized or overflowing block product, zero items,
    /// or a tile extent that cannot be represented by CUB's signed prefix count.
    pub fn new_with_shape(
        operation: Operation,
        algorithm: Algorithm,
        block: [u32; 3],
        items_per_thread: u32,
    ) -> Result<Self> {
        Self::new_typed_with_shape(Dtype::I32, operation, algorithm, block, items_per_thread)
    }

    /// Validate an exact typed `[X, Y, Z]` block specialization.
    ///
    /// Shape and launch requirements match [`Self::new_with_shape`]. The five
    /// validity-aware recipes support every [`Dtype`]; raw recipes require i32.
    ///
    /// # Errors
    ///
    /// Rejects the shapes documented by [`Self::new_with_shape`] and a non-i32
    /// raw recipe. Warp-transpose algorithms require a total thread count
    /// divisible by 32; the X extent need not be divisible by 32.
    pub fn new_typed_with_shape(
        dtype: Dtype,
        operation: Operation,
        algorithm: Algorithm,
        block: [u32; 3],
        items_per_thread: u32,
    ) -> Result<Self> {
        if !operation.validity() && dtype != Dtype::I32 {
            return Err(Error::message(
                "legacy raw recipes require signed i32 elements",
            ));
        }
        let block_threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|threads| (1..=1024).contains(threads));
        if block_threads.is_none()
            || items_per_thread == 0
            || u64::from(block_threads.unwrap_or(0)) * u64::from(items_per_thread) > i32::MAX as u64
        {
            return Err(Error::message(
                "require 1..=1024 block threads, positive items per thread and tile extent <= i32::MAX",
            ));
        }
        if matches!(
            algorithm,
            Algorithm::WarpTranspose | Algorithm::WarpTransposeTimesliced
        ) && block_threads.unwrap() % 32 != 0
        {
            return Err(Error::message(
                "block warp transpose requires a block thread count divisible by 32",
            ));
        }
        Ok(Self {
            operation,
            algorithm,
            dtype,
            block,
            items_per_thread,
            warp_width: None,
        })
    }
    /// Select a physical or logical Warp specialization within this block.
    ///
    /// # Errors
    /// Rejects non-power-of-two widths and widths outside 1..=32, partial physical warps,
    /// algorithms other than Direct, Striped, Vectorize and Transpose,
    /// and legacy raw recipes. The exact root block shape is retained.
    pub fn with_warp_width(mut self, width: u32) -> Result<Self> {
        if width == 0
            || width > 32
            || !width.is_power_of_two()
            || self.block_threads() % 32 != 0
            || matches!(
                self.algorithm,
                Algorithm::WarpTranspose | Algorithm::WarpTransposeTimesliced
            )
            || !self.operation.validity()
        {
            return Err(Error::message(
                "Warp recipes require power-of-two width 1..=32, complete physical warps, Direct/Striped/Vectorize/Transpose and a validity-aware operation",
            ));
        }
        self.warp_width = Some(width);
        Ok(self)
    }
    /// The operation selected by this request.
    pub fn operation(self) -> Operation {
        self.operation
    }
    /// The layout algorithm selected by this request.
    pub fn algorithm(self) -> Algorithm {
        self.algorithm
    }
    /// Exact numeric element type.
    pub fn dtype(self) -> Dtype {
        self.dtype
    }
    /// Exact block shape `[X, Y, Z]`, with x-fastest thread rank.
    pub fn block_shape(self) -> [u32; 3] {
        self.block
    }
    /// Total number of threads in the validated block.
    pub fn block_threads(self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }
    /// Number of payload elements per thread.
    pub fn items_per_thread(self) -> u32 {
        self.items_per_thread
    }
}

/// One member of an ordered native batch, with family-specific validated inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeRequest {
    /// Existing Load/Store operation and layout algorithm.
    LoadStore(Request),
    /// Numeric Scan with preserved input and fresh initialized output.
    Scan(ScanRequest),
    /// Numeric reduction with preserved input and a fresh root-only scalar.
    Reduce(ReduceRequest),
    /// Batched Warp reduction with preserved input and partial fresh outputs.
    ReduceBatched(ReduceBatchedRequest),
    /// Out-of-place Block or Warp Exchange with preserved input.
    Exchange(ExchangeRequest),
    /// One-item Block Shuffle with preserved input and a partial fresh output.
    Shuffle(ShuffleRequest),
    /// Block adjacent subtraction with preserved input and a complete fresh output.
    AdjacentDifference(AdjacentDifferenceRequest),
}

impl NativeRequest {
    fn block_shape(self) -> [u32; 3] {
        match self {
            Self::LoadStore(request) => request.block_shape(),
            Self::Scan(request) => request.block_shape(),
            Self::Reduce(request) => request.block_shape(),
            Self::ReduceBatched(request) => request.block_shape(),
            Self::Exchange(request) => request.block_shape(),
            Self::Shuffle(request) => request.block_shape(),
            Self::AdjacentDifference(request) => request.block_shape(),
        }
    }

    fn source(self, symbol: &str) -> String {
        match self {
            Self::LoadStore(request) => source(request, symbol),
            Self::Scan(request) => request.source(symbol),
            Self::Reduce(request) => request.source(symbol),
            Self::ReduceBatched(request) => request.source(symbol),
            Self::Exchange(request) => request.source(symbol),
            Self::Shuffle(request) => request.source(symbol),
            Self::AdjacentDifference(request) => request.source(symbol),
        }
    }

    fn token(self) -> String {
        match self {
            Self::LoadStore(request) => request.operation.token().into(),
            Self::Scan(request) => request.recipe.operation().token(),
            Self::Reduce(request) => request.recipe.token(),
            Self::ReduceBatched(request) => request.recipe.token(),
            Self::Exchange(request) => request.recipe.token().into(),
            Self::Shuffle(request) => request.recipe.token().into(),
            Self::AdjacentDifference(request) => request.recipe.token().into(),
        }
    }

    fn scratch_free(self) -> bool {
        match self {
            Self::LoadStore(request) => request.algorithm.scratch_free(),
            Self::Scan(_)
            | Self::Reduce(_)
            | Self::ReduceBatched(_)
            | Self::Exchange(_)
            | Self::Shuffle(_)
            | Self::AdjacentDifference(_) => false,
        }
    }

    fn mutates_payload(self) -> bool {
        match self {
            Self::LoadStore(request) => {
                !request.operation.is_load() && !request.algorithm.scratch_free()
            }
            // These families write fresh outputs, never rearranging inputs.
            Self::Scan(_)
            | Self::Reduce(_)
            | Self::ReduceBatched(_)
            | Self::Exchange(_)
            | Self::Shuffle(_)
            | Self::AdjacentDifference(_) => false,
        }
    }
}

/// Byte size and minimum byte alignment of a storage region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    /// Required byte capacity; zero means no shared scratch is accessed.
    pub size: usize,
    /// Positive power-of-two alignment in bytes.
    pub alignment: usize,
}

/// One CUB specialization and its exact storage requirements.
///
/// # Device ABI obligations
///
/// The memory-tile and payload-layout requirements below describe Load/Store.
/// Scan, Reduce, batched Reduce, Exchange, Shuffle and Adjacent Difference use
/// lane-private input and fresh output. Their ABI and validity contracts are
/// documented by [`ScanRequest`], [`ReduceRequest`], [`ReduceBatchedRequest`],
/// [`ExchangeRequest`], [`ShuffleRequest`] and [`AdjacentDifferenceRequest`].
///
/// Every thread in the selected group executes the provider in converged control
/// flow. The group is the exact block, or one aligned Warp of the selected width.
/// Memory is device-accessible and properly aligned; input/output spans
/// cover the full tile or selected valid prefix. Payloads contain the requested
/// number of slots of the requested type. Raw Store payloads are fully
/// initialized and may be mutated.
/// `memory` is the same group tile base pointer in every participating lane; the
/// provider applies no block-origin or subgroup-origin offset for the caller.
/// Load output and Store input must not overlap the memory tile or scratch.
/// The caller supplies disjoint collective ownership of the memory tile.
/// A lane is the x-fastest linear thread rank, `cub::RowMajorTid(X, Y, Z)`.
///
/// Partial controls are group-uniform, with `0 <= valid <= tile_size`; generated
/// code traps on invalid counts. A filled Load's default is also group-uniform.
/// Scratch for transposing algorithms satisfies [`Self::scratch`] and cannot alias another live
/// region. Concurrent groups require disjoint scratch regions. The caller
/// synchronizes participating members before reuse, including loop iterations;
/// the provider does not add a trailing reuse barrier. Direct, Striped and Vectorize ignore scratch,
/// so a null pointer is permitted for that argument only. All providers return
/// unit; loads fill payloads in place and stores write memory.
///
/// Validity-aware ABIs add a byte `flags` pointer immediately after
/// scratch. `payload` addresses aligned storage for the requested number of
/// elements; it need not contain initialized values. Flags occupy a separate,
/// initialized array of `items_per_thread.div_ceil(8)` bytes. Bit `i % 8` of
/// byte `i / 8` denotes whether slot `i` is initialized. Both arrays are private
/// to the lane, disjoint from memory/scratch and one another. Load exclusively
/// borrows both arrays; Direct/Striped/Vectorize Store receives const pointers and borrows them
/// read-only. Transposing Stores exclusively borrow both arrays. Destination
/// `(lane, item)` receives source `(q / items_per_thread, q % items_per_thread)`:
/// Transpose uses `q = group_threads * item + lane`; both Block warp variants use
/// `q = (lane / 32) * (32 * items_per_thread) + 32 * item + lane % 32`.
/// Lane rank is x-fastest within the block, or modulo the selected Warp width.
/// This applies to every slot, independently of the Store count. Repeated
/// Stores permute the current payload again. Native code accesses payload bytes
/// without evaluating invalid values. Masked Load clears bits outside its prefix; filled/full Load write
/// every value before setting its bit. Store checks its entire lane-local
/// write domain before numeric conversion. The counted transposing forms transport
/// arbitrary validity in a separate shared bitmap included in [`Self::scratch`].
/// A full Store checks the complete tile and therefore needs no shared bitmap.
/// Failure traps; it does not promise rollback of stores by other lanes.
#[derive(Debug)]
pub struct Specialization {
    /// SHA-256 of request/source, target/options, captured headers and native toolchain inputs.
    pub identity: [u8; 32],
    /// Name of the `extern "C"` device function in the LTO artifact.
    pub symbol: String,

    /// Actual CUB TempStorage sizeof/alignof from the same NVRTC program.
    pub cub_layout: Layout,
    /// Complete shared region, including any validity plane and padding.
    /// Direct, Striped and Vectorize use zero bytes despite their empty C++ types' sizeof=1.
    pub scratch: Layout,
    /// Whether this specialization accesses no shared scratch or scratch-reuse barrier.
    pub scratch_free: bool,
    /// Whether Store redistributes payloads across lanes into striped order.
    /// True for transposing stores, including counted stores with zero writes.
    /// This describes Store rearrangement, not whether Load writes its payload.
    pub mutates_payload: bool,
}

/// A batch emitted by one NVRTC compilation with one target and header snapshot.
///
/// [`Self::specializations`] preserves request order. Its symbols include each
/// specialization's full input digest, independently of other batch members.
/// Request planning must assign each specialization to exactly one batch per
/// final link. A batch is indivisible; a symbol name cannot deduplicate its
/// strong definitions against a partially overlapping batch.
#[derive(Debug)]
pub struct Artifact {
    /// SHA-256 input identity including ordered specialization identities.
    pub identity: [u8; 32],
    /// Per-request symbols and exact layouts, in request order.
    pub specializations: Vec<Specialization>,
    /// One LTO-IR buffer containing every requested specialization.
    pub ltoir: Vec<u8>,
    /// Generated C++ provider source, including each precise device ABI.
    pub source: String,
    /// Exact NVRTC options, in order.
    pub options: Vec<String>,
    /// NVRTC diagnostics, including successful-compilation warnings.
    pub compiler_log: String,
}

/// Host compiler bound to one immutable header snapshot and CUDA toolkit.
///
/// Construct once for a compilation and reuse for independent specializations.
/// No cache, environment-based header search, Driver API or GPU is involved.
/// Keep the selected toolkit installation unchanged while the provider is alive;
/// its native libraries are executable compiler inputs, not inert data.
pub struct Provider {
    headers: Headers,
    compiler: nvrtc::Compiler,
}
impl Provider {
    /// Load Linux NVRTC from `toolkit/lib64` and snapshot header inputs.
    ///
    /// `checkout` must be a complete CCCL source checkout containing `cub/cub`,
    /// `libcudacxx/include` and `thrust/thrust`. Paths are resolved explicitly;
    /// neither the current directory nor an installed Python package is searched.
    /// CCCL's selected headers take priority over the toolkit's bundled copies.
    /// NVRTC and builtins are loaded by their explicit absolute paths, without
    /// a link-time CUDA dependency. Header bytes are passed in memory to the
    /// compiler, so later filesystem changes do not silently alter this
    /// provider's input identity.
    ///
    /// # Errors
    ///
    /// Errors identify unreadable inputs, unsupported directory symlinks, missing
    /// compiler libraries and incompatible NVRTC entry points. A new instance
    /// observes changed headers/toolchain bytes. Linux is the initial supported host.
    pub fn from_checkout(checkout: impl AsRef<Path>, toolkit: impl AsRef<Path>) -> Result<Self> {
        let checkout = checkout
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::input(checkout.as_ref(), e))?;
        let toolkit = toolkit
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::input(toolkit.as_ref(), e))?;
        Ok(Self {
            headers: Headers::read(&checkout, &toolkit)?,
            compiler: nvrtc::Compiler::load(&toolkit)?,
        })
    }

    /// Compile a nonempty batch for an explicit architecture such as `compute_120`.
    ///
    /// Every request uses the same captured headers, target, C++17 options and
    /// LTO program. Duplicate requests are rejected, since each input identity
    /// owns one definition. The order of requests is preserved in the output.
    /// Separate CUB and complete scratch layout name expressions are registered
    /// before compilation; literal template arguments carry sizeof/alignof
    /// without another compilation or query kernel.
    ///
    /// # Errors
    ///
    /// Empty/duplicate requests and malformed targets are rejected before native
    /// compilation. Unsupported targets/instantiations return NVRTC's log. An
    /// unrecognized layout name fails closed instead of guessing storage sizes.
    pub fn compile(&self, requests: &[Request], target: &str) -> Result<Artifact> {
        let requests: Vec<_> = requests
            .iter()
            .copied()
            .map(NativeRequest::LoadStore)
            .collect();
        self.compile_native(&requests, target)
    }

    /// Compile an ordered mixed batch of the supported families in one NVRTC program.
    ///
    /// Per-specialization identity is independent of unrelated batch members.
    /// Scan/Reduce/batched Reduce/Exchange/Shuffle/Adjacent Difference scratch
    /// describes one group and includes the actual CUB TempStorage,
    /// even when its C++ empty type occupies one byte. The caller plans replication.
    ///
    /// # Errors
    /// Rejects empty/duplicate requests and malformed targets as [`Self::compile`]
    /// does; native compilation and layout decoding errors retain their diagnostics.
    pub fn compile_native(&self, requests: &[NativeRequest], target: &str) -> Result<Artifact> {
        if !target.starts_with("compute_")
            || !target[8..].chars().all(|c| c.is_ascii_alphanumeric())
            || target.len() <= 8
        {
            return Err(Error::message(
                "target must be an explicit NVRTC architecture such as compute_120",
            ));
        }
        if requests.is_empty() {
            return Err(Error::message("at least one specialization is required"));
        }
        for (index, request) in requests.iter().enumerate() {
            if requests[..index].contains(request) {
                return Err(Error::message("duplicate specialization in one batch"));
            }
        }
        let options = vec![
            "--std=c++17".to_owned(),
            format!("--gpu-architecture={target}"),
            "--device-c".to_owned(),
            "-dlto".to_owned(),
            "--no-source-include".to_owned(),
        ];
        let mut inputs = Sha256::new();
        for bytes in [
            b"cuda-coop-codegen-v1".as_slice(),
            &self.headers.digest,
            &self.compiler.identity,
            include_bytes!("lib.rs"),
            include_bytes!("headers.rs"),
            include_bytes!("nvrtc.rs"),
            include_bytes!("block_io.cuh"),
            include_bytes!("scan.rs"),
            include_bytes!("reduce.rs"),
            include_bytes!("reduce_batched.rs"),
            include_bytes!("exchange.rs"),
            include_bytes!("shuffle.rs"),
            include_bytes!("adjacent_difference.rs"),
        ] {
            hash_field(&mut inputs, bytes);
        }
        for option in &options {
            hash_field(&mut inputs, option.as_bytes());
        }
        let mut batch_identity = inputs.clone();
        let mut source = String::from(include_str!("block_io.cuh"));
        source.push_str("\ntemplate <unsigned long long Size, unsigned long long Alignment>\nextern __device__ unsigned char layout_probe;\n");
        let mut expressions = Vec::with_capacity(requests.len() * 2);
        let mut specializations = Vec::with_capacity(requests.len());
        for &request in requests {
            let mut hash = inputs.clone();
            for axis in request.block_shape() {
                hash_field(&mut hash, &axis.to_le_bytes());
            }
            hash_field(
                &mut hash,
                request.source("cuda_coop_specialization").as_bytes(),
            );
            let identity: [u8; 32] = hash.finalize().into();
            hash_field(&mut batch_identity, &identity);
            let suffix: String = identity.iter().map(|byte| format!("{byte:02x}")).collect();
            let symbol = format!("cuda_coop_{}_{}", request.token(), suffix);
            source.push_str(&request.source(&symbol));
            expressions.push(format!("&layout_probe<sizeof({symbol}_detail::primitive::TempStorage), alignof({symbol}_detail::primitive::TempStorage)>"));
            expressions.push(format!("&layout_probe<sizeof({symbol}_detail::scratch_storage), alignof({symbol}_detail::scratch_storage)>"));
            specializations.push(Specialization {
                identity,
                symbol,
                cub_layout: Layout {
                    size: 0,
                    alignment: 1,
                },
                scratch: Layout {
                    size: 0,
                    alignment: 1,
                },
                scratch_free: request.scratch_free(),
                mutates_payload: request.mutates_payload(),
            });
        }
        let output = self
            .compiler
            .compile(&source, &options, &expressions, &self.headers)?;
        for (specialization, layouts) in specializations
            .iter_mut()
            .zip(output.layouts.chunks_exact(2))
        {
            specialization.cub_layout = layouts[0];
            if !specialization.scratch_free {
                specialization.scratch = layouts[1];
            }
        }
        Ok(Artifact {
            identity: batch_identity.finalize().into(),
            specializations,
            ltoir: output.ltoir,
            source,
            options,
            compiler_log: output.log,
        })
    }
}

fn source(request: Request, symbol: &str) -> String {
    let op = request.operation;
    let transpose = !request.algorithm.scratch_free();
    let warp_transpose = matches!(
        request.algorithm,
        Algorithm::WarpTranspose | Algorithm::WarpTransposeTimesliced
    );
    // CUB's full vectorized overload requires the same numeric T for the
    // global pointer and local array. Counted overloads keep opaque cells.
    let numeric_full = request.algorithm == Algorithm::Vectorize
        && matches!(
            op,
            Operation::LoadFullValidity | Operation::StoreFullValidity
        );
    let (header, primitive, algorithm, qualifier, method) = if op.is_load() {
        ("load", "BlockLoad", "BLOCK_LOAD", "const ", "Load")
    } else {
        ("store", "BlockStore", "BLOCK_STORE", "", "Store")
    };
    let (dtype, size) = request.dtype.native();
    let items = request.items_per_thread;
    let [block_x, block_y, block_z] = request.block;
    let threads = request
        .warp_width
        .unwrap_or_else(|| request.block_threads());
    let (group, primitive_type, rank) = if let Some(width) = request.warp_width {
        let (primitive, algorithm) = if op.is_load() {
            ("WarpLoad", "WARP_LOAD")
        } else {
            ("WarpStore", "WARP_STORE")
        };
        (
            "warp",
            format!(
                "cub::{primitive}<cell, {items}, cub::{algorithm}_{}, {width}>",
                request.algorithm.token()
            ),
            format!("cub::RowMajorTid({block_x}, {block_y}, {block_z}) % {width}"),
        )
    } else {
        (
            "block",
            format!(
                "cub::{primitive}<cell, {block_x}, {items}, cub::{algorithm}_{}, {block_y}, {block_z}>",
                request.algorithm.token()
            ),
            format!("cub::RowMajorTid({block_x}, {block_y}, {block_z})"),
        )
    };
    let bitmap_bytes = items.div_ceil(8);
    let tile = threads * items;
    let source_index = if warp_transpose {
        format!("(tid / 32) * (32 * {items}) + 32 * i + (tid % 32)")
    } else {
        format!("{threads} * i + tid")
    };
    let mask_plane = op == Operation::StorePartialValidity && transpose && items > 1;
    let storage = if mask_plane {
        let plane_bytes = u64::from(threads.div_ceil(32)) * u64::from(items) * 4;
        let layout_assertions = if request.warp_width.is_some() {
            format!(
                "static_assert(offsetof(scratch_storage, masks) >= sizeof(primitive::TempStorage));\n\
                 static_assert(offsetof(scratch_storage, masks) % alignof(unsigned int) == 0);\n\
                 static_assert(offsetof(scratch_storage, masks) + {plane_bytes} <= sizeof(scratch_storage));\n\
                 static_assert(alignof(scratch_storage) >= alignof(primitive::TempStorage));"
            )
        } else {
            "static_assert(offsetof(scratch_storage, masks) == sizeof(primitive::TempStorage));\n\
             static_assert(alignof(scratch_storage) == alignof(primitive::TempStorage));"
                .to_owned()
        };
        format!(
            r#"struct scratch_storage {{
  alignas(alignof(primitive::TempStorage)) unsigned char cub[sizeof(primitive::TempStorage)];
  alignas(unsigned int) unsigned char masks[{plane_bytes}];
}};
static_assert(sizeof(unsigned int) == 4);
{layout_assertions}"#
        )
    } else {
        "using scratch_storage = primitive::TempStorage;".to_owned()
    };
    let mut controls = String::new();
    let mut operands = String::new();
    let check = if op.counted() {
        controls.push_str(", int valid");
        operands.push_str(", valid");
        format!("if (valid < 0 || valid > {tile}) __trap();")
    } else {
        String::new()
    };
    if op.filled() {
        controls.push_str(&format!(", {dtype} default_value"));
        operands.push_str(", default_value");
    }
    let constructor = if request.algorithm.scratch_free() {
        "primitive operation;"
    } else {
        "primitive operation{*reinterpret_cast<primitive::TempStorage*>(scratch)};"
    };
    let memory = if matches!(
        op,
        Operation::StorePartial | Operation::StorePartialValidity
    ) {
        "cuda_coop_detail::guarded_output<value_type>{memory, 0}"
    } else {
        "memory"
    };
    let payload_const = if op.validity() && !op.is_load() && !transpose {
        "const "
    } else {
        ""
    };
    let (cell, flags, body) = if op.validity() {
        let cell_bytes = if numeric_full {
            "&values[i]"
        } else {
            "values[i].bytes"
        };
        let domain = if matches!(
            op,
            Operation::LoadMaskedValidity | Operation::StorePartialValidity
        ) {
            if request.algorithm == Algorithm::Striped {
                format!("tid + {threads} * i < valid")
            } else {
                format!("tid * {items} + i < valid")
            }
        } else {
            "true".to_owned()
        };
        let body = if op.is_load() {
            format!(
                r#"cell values[{items}];
  operation.Load(memory, values{operands});
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ({domain}) cuda::std::memcpy(bytes + sizeof(value_type) * i, {cell_bytes}, sizeof(value_type));
  }}
#pragma unroll
  for (int byte = 0; byte < {bitmap_bytes}; ++byte) {{
    unsigned int bits = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {{
      const int i = byte * 8 + bit;
      if (i < {items} && ({domain})) bits |= 1u << bit;
    }}
    flags[byte] = static_cast<unsigned char>(bits);
  }}"#
            )
        } else {
            // The general Store permutes all source slots, including initialized
            // slots outside its write domain. Nonmutating algorithms only
            // need their write domain, checked before any numeric copy below.
            let copy_domain = if transpose { "true" } else { &domain };
            let prepare = if let Some(width) = request.warp_width.filter(|_| mask_plane) {
                let mask = u32::MAX >> (32 - width);
                format!(
                    r#"unsigned char* const mask_plane = scratch + offsetof(scratch_storage, masks);
  const unsigned int first_lane = cuda::ptx::get_sreg_laneid() & (32u - {width});
  const unsigned int member = 0x{mask:08x}u << first_lane;
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    const unsigned int bits = __ballot_sync(member, (flags[i / 8] & (1u << (i % 8))) != 0) >> first_lane;
    if (tid == 0) {{
      void* const address = mask_plane + sizeof(unsigned int) * i;
      ::new (address) unsigned int(bits);
    }}
  }}"#
                )
            } else if mask_plane {
                format!(
                    r#"unsigned char* const mask_plane = scratch + offsetof(scratch_storage, masks);
  const int warp = tid / 32;
  const int remaining = {threads} - warp * 32;
  const unsigned int active = remaining >= 32 ? 0xffffffffu : (1u << remaining) - 1;
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    const unsigned int bits = __ballot_sync(active, (flags[i / 8] & (1u << (i % 8))) != 0);
    if ((tid & 31) == 0) {{
      void* const address = mask_plane + sizeof(unsigned int) * (warp * {items} + i);
      ::new (address) unsigned int(bits);
    }}
  }}"#
                )
            } else {
                String::new()
            };
            let restore = if transpose {
                let validity = if op == Operation::StoreFullValidity {
                    // The full-domain preflight proves every permuted slot
                    // initialized; no collective mask exchange is necessary.
                    format!(
                        r#"#pragma unroll
  for (int byte = 0; byte < {bitmap_bytes}; ++byte) {{
    unsigned int bits = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {{
      if (byte * 8 + bit < {items}) bits |= 1u << bit;
    }}
    flags[byte] = static_cast<unsigned char>(bits);
  }}"#
                    )
                } else if mask_plane {
                    let completion = if request.warp_width.is_some() {
                        "// Complete this group's plane writes before any plane read.\n  __syncwarp(member);"
                    } else {
                        "// CUB's exchange barrier orders the initialized plane writes above."
                    };
                    format!(
                        r#"{completion}
#pragma unroll
  for (int byte = 0; byte < {bitmap_bytes}; ++byte) {{
    unsigned int bits = 0;
#pragma unroll
    for (int bit = 0; bit < 8; ++bit) {{
      const int i = byte * 8 + bit;
      if (i < {items}) {{
        const int q = {source_index};
        const int source_thread = q / {items};
        const int source_item = q % {items};
        const auto* const address = reinterpret_cast<const unsigned int*>(
          mask_plane + sizeof(unsigned int) * ((source_thread / 32) * {items} + source_item));
        if (((*address >> (source_thread % 32)) & 1u) != 0) bits |= 1u << bit;
      }}
    }}
    flags[byte] = static_cast<unsigned char>(bits);
  }}"#
                    )
                } else {
                    // With one item per lane, BlockedToStriped is identity.
                    String::new()
                };
                format!(
                    r#"
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    cuda::std::memcpy(bytes + sizeof(value_type) * i, values[i].bytes, sizeof(value_type));
  }}
  {validity}"#
                )
            } else {
                String::new()
            };
            format!(
                r#"// Complete this lane's preflight before any numeric conversion or store.
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if (({domain}) && (flags[i / 8] & (1u << (i % 8))) == 0) __trap();
  }}
  {prepare}
  cell values[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ({copy_domain}) cuda::std::memcpy({cell_bytes}, bytes + sizeof(value_type) * i, sizeof(value_type));
  }}
  operation.Store({memory}, values{operands});{restore}"#
            )
        };
        (
            if numeric_full {
                "value_type"
            } else {
                "cuda_coop_detail::opaque_cell<value_type>"
            },
            if op.is_load() || transpose {
                ", unsigned char* flags"
            } else {
                ", const unsigned char* flags"
            },
            format!(
                "auto* const bytes = reinterpret_cast<{payload_const}unsigned char*>(payload);\n  {body}"
            ),
        )
    } else {
        (
            "value_type",
            "",
            format!(
                "operation.{method}({memory}, *reinterpret_cast<value_type (*)[{items}]>(payload){operands});"
            ),
        )
    };
    format!(
        r#"// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <cub/{group}/{group}_{header}.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/cstddef>
namespace {symbol}_detail {{
using value_type = {dtype};
using cell = {cell};
using primitive = {primitive_type};
{storage}
static_assert(sizeof(value_type) == {size} && alignof(value_type) == {size});
static_assert(sizeof(cell) == sizeof(value_type) && alignof(cell) == alignof(value_type));
extern "C" __device__ void {symbol}({qualifier}value_type* memory, {payload_const}value_type* payload, unsigned char* scratch{flags}{controls}) {{
  const int tid = {rank};
  {check}
  {constructor}
  {body}
}}
}}
"#
    )
}
