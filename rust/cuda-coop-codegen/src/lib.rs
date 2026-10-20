// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Native, host-only CUB specialization for Rust cooperative algorithms.
//!
//! This private development crate compiles block Load/Store providers with
//! NVRTC. It returns linkable LTO-IR and the exact C++ scratch layout from the
//! **same compilation**, without a CUDA context, GPU or Python dependency.
//! It is a compiler building block, not a device API or a package release.
//!
//! # Supported specializations
//!
//! [`Request`] describes a one-dimensional complete block, positive fixed
//! items per thread, and [`Algorithm::Direct`] or [`Algorithm::Transpose`].
//! The raw i32 recipes support Direct and Transpose. Their Store payloads must
//! be fully initialized. The validity-aware Direct recipes support all ten
//! [`Dtype`] variants with separate value storage and packed per-slot validity bits.
//! Masked Load resets validity outside its prefix; Direct Store preserves both
//! arrays and checks initialization only in its write domain. Transpose Store
//! replaces each lane's blocked payload with striped items across the block;
//! validity-aware Transpose recipes are not yet supported.
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

mod headers;
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

/// CUB block layout algorithm; both variants produce/consume blocked payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Algorithm {
    /// Consecutive items per thread, without shared scratch or a reuse barrier.
    Direct,
    /// Coalesced access with shared-memory transposition to/from blocked layout.
    Transpose,
}
impl Algorithm {
    fn token(self) -> &'static str {
        match self {
            Self::Direct => "DIRECT",
            Self::Transpose => "TRANSPOSE",
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
    /// Full Store after checking every validity bit is set; preserves values and bits.
    StoreFullValidity,
    /// Counted Store after checking its write domain; preserves all values and flags.
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

/// A validated, one-dimensional block specialization.
///
/// This is an in-process implementation interface, not a serialized compiler
/// protocol. All fields are static specialization inputs. The generated ABI
/// uses runtime `valid` and `default_value` only when selected by [`Operation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Request {
    operation: Operation,
    algorithm: Algorithm,
    dtype: Dtype,
    block_threads: u32,
    items_per_thread: u32,
}
impl Request {
    /// Validate an exact block shape and payload extent.
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
    /// The five validity-aware recipes support every [`Dtype`] with Direct.
    /// Legacy raw recipes retain their i32 ABI and algorithm support.
    ///
    /// # Errors
    ///
    /// Rejects the invalid shapes documented by [`Self::new`], a non-i32 raw
    /// recipe, or a validity-aware Transpose recipe.
    pub fn new_typed(
        dtype: Dtype,
        operation: Operation,
        algorithm: Algorithm,
        block_threads: u32,
        items_per_thread: u32,
    ) -> Result<Self> {
        if !operation.validity() && dtype != Dtype::I32 {
            return Err(Error::message(
                "legacy raw recipes require signed i32 elements",
            ));
        }
        if operation.validity() && algorithm != Algorithm::Direct {
            return Err(Error::message(
                "validity-aware recipes currently require Direct",
            ));
        }
        if !(1..=1024).contains(&block_threads)
            || items_per_thread == 0
            || u64::from(block_threads) * u64::from(items_per_thread) > i32::MAX as u64
        {
            return Err(Error::message(
                "require 1..=1024 block threads, positive items per thread and tile extent <= i32::MAX",
            ));
        }
        Ok(Self {
            operation,
            algorithm,
            dtype,
            block_threads,
            items_per_thread,
        })
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
    /// Exact number of threads in the one-dimensional block.
    pub fn block_threads(self) -> u32 {
        self.block_threads
    }
    /// Number of payload elements per thread.
    pub fn items_per_thread(self) -> u32 {
        self.items_per_thread
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
/// Every thread in the exact block executes the provider in converged control
/// flow. Memory is device-accessible and properly aligned; input/output spans
/// cover the full tile or selected valid prefix. Payloads contain the requested
/// number of slots of the requested type. Raw Store payloads are fully
/// initialized and may be mutated.
/// `memory` is the same block tile base pointer in every lane; the provider
/// applies no block-origin offset for the caller.
/// Load output and Store input must not overlap the memory tile or scratch.
/// The caller supplies disjoint collective ownership of the memory tile.
///
/// Partial controls are block-uniform, with `0 <= valid <= tile_size`; generated
/// code traps on invalid counts. A filled Load's default is also block-uniform.
/// Transpose scratch satisfies [`Self::scratch`] and cannot alias another live
/// region. The caller synchronizes before reuse, including loop iterations;
/// the provider does not add a trailing reuse barrier. Direct ignores scratch,
/// so a null pointer is permitted for that argument only. All providers return
/// unit; loads fill payloads in place and stores write memory.
///
/// Validity-aware Direct ABIs add `unsigned char* flags` immediately after
/// scratch. `payload` addresses aligned storage for the requested number of
/// elements; it need not contain initialized values. Flags occupy a separate,
/// initialized array of `items_per_thread.div_ceil(8)` bytes. Bit `i % 8` of
/// byte `i / 8` denotes whether slot `i` is initialized. Both arrays are private
/// to the lane, disjoint from memory/scratch and one another, and remain
/// exclusively borrowed for the call. Native code accesses payload bytes
/// without evaluating invalid values. Masked Load clears bits outside its
/// prefix; filled/full Load write every value before setting its bit.
/// Store checks its entire lane-local write domain before numeric conversion
/// and preserves both arrays. Failure traps; it does not promise rollback of
/// stores by other lanes.
#[derive(Debug)]
pub struct Specialization {
    /// SHA-256 of request/source, target/options, captured headers and native toolchain inputs.
    pub identity: [u8; 32],
    /// Name of the `extern "C"` device function in the LTO artifact.
    pub symbol: String,

    /// Actual CUB TempStorage sizeof/alignof from the same NVRTC program.
    pub cub_layout: Layout,
    /// Required shared region; Direct uses zero bytes despite its empty C++ type's sizeof=1.
    pub scratch: Layout,
    /// Whether this specialization accesses no shared scratch or scratch-reuse barrier.
    pub scratch_free: bool,
    /// Whether Store redistributes payloads across lanes into striped order.
    /// True only for raw Transpose stores; every slot remains initialized.
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
    /// CUB layout name expressions are registered before compilation; literal
    /// template arguments carry sizeof/alignof without another compilation or
    /// query kernel.
    ///
    /// # Errors
    ///
    /// Empty/duplicate requests and malformed targets are rejected before native
    /// compilation. Unsupported targets/instantiations return NVRTC's log. An
    /// unrecognized layout name fails closed instead of guessing storage sizes.
    pub fn compile(&self, requests: &[Request], target: &str) -> Result<Artifact> {
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
        ] {
            hash_field(&mut inputs, bytes);
        }
        for option in &options {
            hash_field(&mut inputs, option.as_bytes());
        }
        let mut batch_identity = inputs.clone();
        let mut source = String::from(include_str!("block_io.cuh"));
        source.push_str("\ntemplate <unsigned long long Size, unsigned long long Alignment>\nextern __device__ unsigned char layout_probe;\n");
        let mut expressions = Vec::with_capacity(requests.len());
        let mut specializations = Vec::with_capacity(requests.len());
        for &request in requests {
            let mut hash = inputs.clone();
            hash_field(
                &mut hash,
                self::source(request, "cuda_coop_specialization").as_bytes(),
            );
            let identity: [u8; 32] = hash.finalize().into();
            hash_field(&mut batch_identity, &identity);
            let suffix: String = identity.iter().map(|byte| format!("{byte:02x}")).collect();
            let symbol = format!("cuda_coop_{}_{}", request.operation.token(), suffix);
            source.push_str(&self::source(request, &symbol));
            expressions.push(format!("&layout_probe<sizeof({symbol}_detail::primitive::TempStorage), alignof({symbol}_detail::primitive::TempStorage)>"));
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
                scratch_free: request.algorithm == Algorithm::Direct,
                mutates_payload: !request.operation.is_load()
                    && request.algorithm == Algorithm::Transpose,
            });
        }
        let output = self
            .compiler
            .compile(&source, &options, &expressions, &self.headers)?;
        for (specialization, layout) in specializations.iter_mut().zip(output.layouts) {
            specialization.cub_layout = layout;
            if !specialization.scratch_free {
                specialization.scratch = layout;
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
    let (header, primitive, algorithm, qualifier, method) = if op.is_load() {
        ("load", "BlockLoad", "BLOCK_LOAD", "const ", "Load")
    } else {
        ("store", "BlockStore", "BLOCK_STORE", "", "Store")
    };
    let (dtype, size) = request.dtype.native();
    let items = request.items_per_thread;
    let bitmap_bytes = items.div_ceil(8);
    let tile = request.block_threads * items;
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
    let constructor = if request.algorithm == Algorithm::Direct {
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
    let (cell, flags, body) = if op.validity() {
        let domain = if matches!(
            op,
            Operation::LoadMaskedValidity | Operation::StorePartialValidity
        ) {
            format!("int(threadIdx.x) * {items} + i < valid")
        } else {
            "true".to_owned()
        };
        let body = if op.is_load() {
            format!(
                r#"cell values[{items}];
  operation.Load(memory, values{operands});
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ({domain}) cuda::std::memcpy(bytes + sizeof(value_type) * i, values[i].bytes, sizeof(value_type));
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
            format!(
                r#"// Complete this lane's preflight before any numeric conversion or store.
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if (({domain}) && (flags[i / 8] & (1u << (i % 8))) == 0) __trap();
  }}
  cell values[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ({domain}) cuda::std::memcpy(values[i].bytes, bytes + sizeof(value_type) * i, sizeof(value_type));
  }}
  operation.Store({memory}, values{operands});"#
            )
        };
        (
            "cuda_coop_detail::opaque_cell<value_type>",
            ", unsigned char* flags",
            format!("auto* const bytes = reinterpret_cast<unsigned char*>(payload);\n  {body}"),
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
#include <cub/block/block_{header}.cuh>
namespace {symbol}_detail {{
using value_type = {dtype};
using cell = {cell};
using primitive = cub::{primitive}<cell, {threads}, {items}, cub::{algorithm}_{choice}>;
static_assert(sizeof(value_type) == {size} && alignof(value_type) == {size});
static_assert(sizeof(cell) == sizeof(value_type) && alignof(cell) == alignof(value_type));
extern "C" __device__ void {symbol}({qualifier}value_type* memory, value_type* payload, unsigned char* scratch{flags}{controls}) {{
  {check}
  {constructor}
  {body}
}}
}}
"#,
        threads = request.block_threads,
        choice = request.algorithm.token()
    )
}
