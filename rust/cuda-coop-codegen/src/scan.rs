// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Scan recipes, separate from Load/Store algorithms.

use crate::{Dtype, Error, Result};

/// Static associative operator used by a native Scan specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOperator {
    /// Addition in the input dtype.
    Sum,
    /// Multiplication in the input dtype.
    Multiplies,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Integer bitwise AND.
    BitAnd,
    /// Integer bitwise OR.
    BitOr,
    /// Integer bitwise XOR.
    BitXor,
}

impl ScanOperator {
    fn token(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Multiplies => "multiplies",
            Self::Min => "min",
            Self::Max => "max",
            Self::BitAnd => "bit_and",
            Self::BitOr => "bit_or",
            Self::BitXor => "bit_xor",
        }
    }

    fn parse(token: &str) -> Result<Self> {
        match token {
            "sum" => Ok(Self::Sum),
            "multiplies" => Ok(Self::Multiplies),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "bit_and" => Ok(Self::BitAnd),
            "bit_or" => Ok(Self::BitOr),
            "bit_xor" => Ok(Self::BitXor),
            _ => Err(Error::message("unsupported Scan operator")),
        }
    }

    fn functor(self) -> &'static str {
        match self {
            Self::Sum => "cuda::std::plus<value_type>",
            Self::Multiplies => "cuda::std::multiplies<value_type>",
            Self::Min => "cuda::minimum<value_type>",
            Self::Max => "cuda::maximum<value_type>",
            Self::BitAnd => "cuda::std::bit_and<value_type>",
            Self::BitOr => "cuda::std::bit_or<value_type>",
            Self::BitXor => "cuda::std::bit_xor<value_type>",
        }
    }
}

/// Scan mode and runtime seed presence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOperation {
    /// Exclusive sum with typed zero, without a runtime seed argument.
    ExclusiveSum,
    /// Exclusive scan with one runtime seed of exactly the input dtype.
    Exclusive(ScanOperator),
    /// Inclusive scan without a seed argument.
    Inclusive(ScanOperator),
}

impl ScanOperation {
    pub(crate) fn seeded(self) -> bool {
        matches!(self, Self::Exclusive(_))
    }

    pub(crate) fn token(self) -> String {
        match self {
            Self::ExclusiveSum => "scan_exclusive_sum_validity_bits_v1".into(),
            Self::Exclusive(op) => {
                format!("scan_exclusive_{}_seeded_validity_bits_v1", op.token())
            }
            Self::Inclusive(op) => format!("scan_inclusive_{}_validity_bits_v1", op.token()),
        }
    }

    fn parse(token: &str) -> Result<Self> {
        if token == "scan_exclusive_sum_validity_bits_v1" {
            return Ok(Self::ExclusiveSum);
        }
        if let Some(op) = token
            .strip_prefix("scan_exclusive_")
            .and_then(|s| s.strip_suffix("_seeded_validity_bits_v1"))
        {
            return Ok(Self::Exclusive(ScanOperator::parse(op)?));
        }
        if let Some(op) = token
            .strip_prefix("scan_inclusive_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
        {
            return Ok(Self::Inclusive(ScanOperator::parse(op)?));
        }
        Err(Error::message("unsupported Scan operation or seed form"))
    }
}

/// CUB algorithm selected for a complete Block Scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanAlgorithm {
    /// Shared-memory raking, the default Block algorithm.
    Raking,
    /// Raking with register-cached partial reductions.
    RakingMemoize,
    /// Combine physical-warp scans; requires a block size divisible by 32.
    WarpScans,
}

/// Registration identity independent of dtype, geometry and payload extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanRecipe {
    /// Complete Block operation with its selected Block algorithm.
    Block(ScanOperation, ScanAlgorithm),
    /// Scalar physical/logical Warp operation with implicit algorithm selection.
    Warp(ScanOperation),
}

impl ScanRecipe {
    /// Operation and seed presence selected by this recipe.
    pub fn operation(self) -> ScanOperation {
        match self {
            Self::Block(operation, _) | Self::Warp(operation) => operation,
        }
    }

    pub(crate) fn algorithm_token(self) -> &'static str {
        match self {
            Self::Block(_, ScanAlgorithm::Raking) => "raking",
            Self::Block(_, ScanAlgorithm::RakingMemoize) => "raking_memoize",
            Self::Block(_, ScanAlgorithm::WarpScans) => "warp_scans",
            Self::Warp(_) => "warp",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        let operation = ScanOperation::parse(operation)?;
        match algorithm {
            "raking" => Ok(Self::Block(operation, ScanAlgorithm::Raking)),
            "raking_memoize" => Ok(Self::Block(operation, ScanAlgorithm::RakingMemoize)),
            "warp_scans" => Ok(Self::Block(operation, ScanAlgorithm::WarpScans)),
            "warp" => Ok(Self::Warp(operation)),
            _ => Err(Error::message("unsupported Scan algorithm")),
        }
    }
}

/// Validated numeric Scan specialization with exact root geometry.
///
/// Native calls preserve all input values and validity bits and initialize every
/// fresh output slot. Their ABI is `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits[, T seed])`. Input and output must be
/// live, disjoint lane-local storage of the declared extent; validity is packed
/// low-bit-first. Every input bit is checked before numeric input is read.
/// All group members must participate, seeds must be group-uniform, and scratch
/// ownership/completion belongs to the checked compiler integration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScanRequest {
    pub(crate) recipe: ScanRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
    width: Option<u32>,
}

impl ScanRequest {
    /// Validate a complete Block with positive fixed payload extent.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, more than 1024 threads, zero items or a
    /// total extent exceeding i32. WarpScans requires complete physical warps;
    /// bitwise operators require an integer dtype.
    pub fn block(
        dtype: Dtype,
        operation: ScanOperation,
        algorithm: ScanAlgorithm,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        Self::validate(
            dtype,
            ScanRecipe::Block(operation, algorithm),
            block,
            items,
            None,
        )
    }

    /// Validate one scalar per lane in a complete physical/logical Warp.
    ///
    /// # Errors
    /// Rejects invalid Block geometry, incomplete physical warps, widths other
    /// than 1/2/4/8/16/32, or a bitwise operator with a floating dtype.
    pub fn warp(
        dtype: Dtype,
        operation: ScanOperation,
        block: [u32; 3],
        width: u32,
    ) -> Result<Self> {
        Self::validate(dtype, ScanRecipe::Warp(operation), block, 1, Some(width))
    }

    fn validate(
        dtype: Dtype,
        recipe: ScanRecipe,
        block: [u32; 3],
        items: u32,
        width: Option<u32>,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n));
        let Some(threads) = threads else {
            return Err(Error::message(
                "Scan requires 1..=1024 block threads with positive axes",
            ));
        };
        if items == 0 || u64::from(threads) * u64::from(items) > i32::MAX as u64 {
            return Err(Error::message(
                "Scan requires positive items and tile extent <= i32::MAX",
            ));
        }
        if matches!(
            recipe,
            ScanRecipe::Block(_, ScanAlgorithm::WarpScans) | ScanRecipe::Warp(_)
        ) && threads % 32 != 0
        {
            return Err(Error::message(
                "Warp Scan and Block WarpScans require complete physical warps",
            ));
        }
        if width.is_some_and(|w| w == 0 || w > 32 || !w.is_power_of_two()) {
            return Err(Error::message(
                "Warp Scan width must be a power of two in 1..=32",
            ));
        }
        if matches!(dtype, Dtype::F32 | Dtype::F64)
            && matches!(
                recipe.operation(),
                ScanOperation::Exclusive(
                    ScanOperator::BitAnd | ScanOperator::BitOr | ScanOperator::BitXor
                ) | ScanOperation::Inclusive(
                    ScanOperator::BitAnd | ScanOperator::BitOr | ScanOperator::BitXor
                )
            )
        {
            return Err(Error::message(
                "bitwise Scan operators require integer elements",
            ));
        }
        Ok(Self {
            recipe,
            dtype,
            block,
            items,
            width,
        })
    }

    pub(crate) fn source(self, symbol: &str) -> String {
        let (dtype, _) = self.dtype.native();
        let [x, y, z] = self.block;
        let items = self.items;
        let primitive = match self.recipe {
            ScanRecipe::Block(_, algorithm) => {
                let algorithm = match algorithm {
                    ScanAlgorithm::Raking => "BLOCK_SCAN_RAKING",
                    ScanAlgorithm::RakingMemoize => "BLOCK_SCAN_RAKING_MEMOIZE",
                    ScanAlgorithm::WarpScans => "BLOCK_SCAN_WARP_SCANS",
                };
                format!("cub::BlockScan<value_type, {x}, cub::{algorithm}, {y}, {z}>")
            }
            ScanRecipe::Warp(_) => format!("cub::WarpScan<value_type, {}>", self.width.unwrap()),
        };
        let controls = if self.recipe.operation().seeded() {
            format!(", {dtype} seed")
        } else {
            String::new()
        };
        let operands = if items == 1 {
            "local_input[0], local_output[0]"
        } else {
            "local_input, local_output"
        };
        let call = match self.recipe.operation() {
            ScanOperation::ExclusiveSum => format!("operation.ExclusiveSum({operands});"),
            ScanOperation::Exclusive(op) => format!(
                "operation.ExclusiveScan({operands}, seed, {}{{}});",
                op.functor()
            ),
            ScanOperation::Inclusive(op) => {
                format!("operation.InclusiveScan({operands}, {}{{}});", op.functor())
            }
        };
        let tail = if items % 8 == 0 {
            String::new()
        } else {
            format!(
                "output_bits[{}] = {};",
                items / 8,
                (1_u32 << (items % 8)) - 1
            )
        };
        format!(
            r#"
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits{controls}) {{
  using namespace {symbol}_detail;
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{items}];
  value_type local_output[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_input[i] = input[i];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  {call}
#pragma unroll
  for (int i = 0; i < {items}; ++i) output[i] = local_output[i];
#pragma unroll
  for (int i = 0; i < {full_bytes}; ++i) output_bits[i] = 255;
  {tail}
}}
"#,
            full_bytes = items / 8
        )
    }
}
