// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Native, host-only CUB specialization for Rust cooperative algorithms.
//!
//! This private development crate compiles i32 block Load/Store providers with
//! NVRTC. It returns linkable LTO-IR and the exact C++ scratch layout from the
//! **same compilation**, without a CUDA context, GPU or Python dependency.
//! It is a compiler building block, not a device API or a package release.
//!
//! # Supported specializations
//!
//! [`Request`] describes a one-dimensional complete block, positive fixed
//! items per thread, and [`Algorithm::Direct`] or [`Algorithm::Transpose`].
//! Loads either fill the full tile or take an explicit default for the masked
//! suffix. Stores require all payload slots initialized, even for partial
//! stores. Transpose Store replaces each lane's blocked payload with striped
//! items from across the block. No-default masked loads are
//! intentionally absent from this provider surface.
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

/// Device operation and its validity controls, all with i32 elements.
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
}
impl Operation {
    fn is_load(self) -> bool {
        matches!(self, Self::LoadFull | Self::LoadFilled)
    }
    fn token(self) -> &'static str {
        match self {
            Self::LoadFull => "load_full",
            Self::LoadFilled => "load_filled",
            Self::StoreFull => "store_full",
            Self::StorePartial => "store_partial",
        }
    }
}

/// A validated i32, one-dimensional block specialization.
///
/// This is an in-process implementation interface, not a serialized compiler
/// protocol. All fields are static specialization inputs. The generated ABI
/// uses runtime `valid` and `default_value` only when selected by [`Operation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Request {
    operation: Operation,
    algorithm: Algorithm,
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
    /// Exact number of threads in the one-dimensional block.
    pub fn block_threads(self) -> u32 {
        self.block_threads
    }
    /// Number of i32 payload elements per thread.
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
/// number of i32 slots. Store payloads are fully initialized and may be mutated.
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
    /// True only for Transpose stores; every slot remains initialized.
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
        ] {
            hash_field(&mut inputs, bytes);
        }
        for option in &options {
            hash_field(&mut inputs, option.as_bytes());
        }
        let mut batch_identity = inputs.clone();
        let mut source = String::from(
            "template <unsigned long long Size, unsigned long long Alignment>\nextern __device__ unsigned char layout_probe;\n",
        );
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
    let (header, primitive, algorithm, pointer) = if request.operation.is_load() {
        ("load", "BlockLoad", "BLOCK_LOAD", "const int*")
    } else {
        ("store", "BlockStore", "BLOCK_STORE", "int*")
    };
    let (controls, operands, check) = match request.operation {
        Operation::LoadFull | Operation::StoreFull => ("", "", String::new()),
        Operation::LoadFilled => (
            ", int valid, int default_value",
            ", valid, default_value",
            format!(
                "if (valid < 0 || valid > {}) __trap();",
                request.block_threads * request.items_per_thread
            ),
        ),
        Operation::StorePartial => (
            ", int valid",
            ", valid",
            format!(
                "if (valid < 0 || valid > {}) __trap();",
                request.block_threads * request.items_per_thread
            ),
        ),
    };
    let constructor = if request.algorithm == Algorithm::Direct {
        "primitive operation;"
    } else {
        "primitive operation{*reinterpret_cast<primitive::TempStorage*>(scratch)};"
    };
    let method = if request.operation.is_load() {
        "Load"
    } else {
        "Store"
    };
    format!(
        r#"// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <cub/block/block_{header}.cuh>
namespace {symbol}_detail {{
using primitive = cub::{primitive}<int, {threads}, {items}, cub::{algorithm}_{choice}>;
static_assert(sizeof(int) == 4 && alignof(int) == 4);
extern "C" __device__ void {symbol}({pointer} memory, int* payload, unsigned char* scratch{controls}) {{
  {check}
  {constructor}
  operation.{method}(memory, *reinterpret_cast<int (*)[{items}]>(payload){operands});
}}
}}
"#,
        threads = request.block_threads,
        items = request.items_per_thread,
        choice = request.algorithm.token()
    )
}
