// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated root-result Reduce recipes, separate from Scan and Load/Store.

use crate::{Dtype, Error, Result};

/// Static associative operator used by a native Reduce specialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOperator {
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

impl ReduceOperator {
    pub(crate) fn token(self) -> &'static str {
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

    pub(crate) fn parse(token: &str) -> Result<Self> {
        match token {
            "sum" => Ok(Self::Sum),
            "multiplies" => Ok(Self::Multiplies),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "bit_and" => Ok(Self::BitAnd),
            "bit_or" => Ok(Self::BitOr),
            "bit_xor" => Ok(Self::BitXor),
            _ => Err(Error::message("unsupported Reduce operator")),
        }
    }

    pub(crate) fn functor(self) -> &'static str {
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

/// Input form; a one-item payload remains distinct from a scalar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceForm {
    /// One initialized scalar from every member.
    Scalar,
    /// One initialized scalar per member, with a group-uniform valid count.
    ScalarPrefix,
    /// Every initialized item in a fixed per-thread payload.
    Payload,
}

/// CUB algorithm selected for a complete Block reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceAlgorithm {
    /// Raking specialized for the supported commutative builtins.
    RakingCommutativeOnly,
    /// Raking reduction.
    Raking,
    /// Combine physical-warp reductions, the default Block algorithm.
    WarpReductions,
}

/// Registration identity independent of dtype, geometry and input extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceRecipe {
    /// Complete Block operation with its selected algorithm and input form.
    Block(ReduceOperator, ReduceAlgorithm, ReduceForm),
    /// Physical/logical Warp operation with implicit algorithm selection.
    Warp(ReduceOperator, ReduceForm),
}

impl ReduceRecipe {
    /// Builtin selected by this recipe.
    pub fn operator(self) -> ReduceOperator {
        match self {
            Self::Block(op, _, _) | Self::Warp(op, _) => op,
        }
    }

    /// Scalar or fixed-payload input form.
    pub fn form(self) -> ReduceForm {
        match self {
            Self::Block(_, _, form) | Self::Warp(_, form) => form,
        }
    }

    pub(crate) fn token(self) -> String {
        let form = match self.form() {
            ReduceForm::Scalar => "scalar",
            ReduceForm::ScalarPrefix => "prefix",
            ReduceForm::Payload => "payload",
        };
        format!("reduce_{}_{form}_validity_bits_v1", self.operator().token())
    }

    pub(crate) fn algorithm_token(self) -> &'static str {
        match self {
            Self::Block(_, ReduceAlgorithm::RakingCommutativeOnly, _) => "raking_commutative_only",
            Self::Block(_, ReduceAlgorithm::Raking, _) => "raking",
            Self::Block(_, ReduceAlgorithm::WarpReductions, _) => "warp_reductions",
            Self::Warp(_, _) => "warp",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        let operation = operation
            .strip_prefix("reduce_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
            .ok_or_else(|| Error::message("unsupported Reduce operation or input form"))?;
        let (operator, form) = operation
            .rsplit_once('_')
            .ok_or_else(|| Error::message("missing Reduce input form"))?;
        let op = ReduceOperator::parse(operator)?;
        let form = match form {
            "scalar" => ReduceForm::Scalar,
            "prefix" => ReduceForm::ScalarPrefix,
            "payload" => ReduceForm::Payload,
            _ => return Err(Error::message("unsupported Reduce input form")),
        };
        match algorithm {
            "raking_commutative_only" => Ok(Self::Block(
                op,
                ReduceAlgorithm::RakingCommutativeOnly,
                form,
            )),
            "raking" => Ok(Self::Block(op, ReduceAlgorithm::Raking, form)),
            "warp_reductions" => Ok(Self::Block(op, ReduceAlgorithm::WarpReductions, form)),
            "warp" => Ok(Self::Warp(op, form)),
            _ => Err(Error::message("unsupported Reduce algorithm")),
        }
    }
}

/// Validated reduction with an initialized scalar result only at group rank zero.
///
/// Native calls preserve all input values and packed validity bits. The ABI is
/// `(const T* input, T* output, u8* scratch, const u8* input_bits,
/// u8* output_bits[, i32 valid])`. Input has the declared item extent and
/// `ceil(items/8)` low-bit-first validity bytes; output has exactly one T slot
/// and one validity byte. All pointers are live, aligned lane-local storage;
/// output is fresh and disjoint from input. Every input bit is checked before
/// numeric input is read. ScalarPrefix requires a group-uniform count in
/// `1..=group_size`, and every scalar is initialized even outside that prefix.
/// Only rank zero writes the result and sets its byte to one. Other members'
/// result bytes are zero and their numeric output slots remain uninitialized.
/// All members participate; non-power-of-two Warps mean exactly the low-width
/// prefix of each physical warp. The caller owns scratch replication and reuse
/// completion. The provider adds no completion barrier or result broadcast.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReduceRequest {
    pub(crate) recipe: ReduceRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
    width: Option<u32>,
}

impl ReduceRequest {
    /// Validate a complete Block reduction with exact XYZ geometry.
    ///
    /// # Errors
    /// Rejects invalid axes, more than 1024 threads, invalid item extents,
    /// scalar forms with more than one item, and floating bitwise operators.
    pub fn block(
        dtype: Dtype,
        operator: ReduceOperator,
        algorithm: ReduceAlgorithm,
        form: ReduceForm,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        Self::validate(
            dtype,
            ReduceRecipe::Block(operator, algorithm, form),
            block,
            items,
            None,
        )
    }

    /// Validate a Warp reduction within complete physical warps.
    ///
    /// Widths 1/2/4/8/16/32 select aligned partitions; widths 17..31 select one
    /// participating low-lane prefix per physical warp. This native validation
    /// does not establish the caller's required membership or participation.
    ///
    /// # Errors
    /// Rejects invalid Block geometry, incomplete physical warps, unsupported
    /// widths, invalid input extents and floating bitwise operators.
    pub fn warp(
        dtype: Dtype,
        operator: ReduceOperator,
        form: ReduceForm,
        block: [u32; 3],
        width: u32,
        items: u32,
    ) -> Result<Self> {
        Self::validate(
            dtype,
            ReduceRecipe::Warp(operator, form),
            block,
            items,
            Some(width),
        )
    }

    fn validate(
        dtype: Dtype,
        recipe: ReduceRecipe,
        block: [u32; 3],
        items: u32,
        width: Option<u32>,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("Reduce requires 1..=1024 block threads with positive axes")
            })?;
        if items == 0 || u64::from(threads) * u64::from(items) > i32::MAX as u64 {
            return Err(Error::message(
                "Reduce requires positive items and tile extent <= i32::MAX",
            ));
        }
        if recipe.form() != ReduceForm::Payload && items != 1 {
            return Err(Error::message(
                "scalar Reduce forms require exactly one input item",
            ));
        }
        if let Some(width) = width {
            if threads % 32 != 0 {
                return Err(Error::message(
                    "Warp Reduce requires complete physical warps",
                ));
            }
            if width == 0 || width > 32 || (!width.is_power_of_two() && width < 17) {
                return Err(Error::message(
                    "Warp Reduce requires power-of-two width 1..=32 or guarded prefix 17..=31",
                ));
            }
        }
        if matches!(dtype, Dtype::F32 | Dtype::F64)
            && matches!(
                recipe.operator(),
                ReduceOperator::BitAnd | ReduceOperator::BitOr | ReduceOperator::BitXor
            )
        {
            return Err(Error::message(
                "bitwise Reduce operators require integer elements",
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

    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.block
    }

    pub(crate) fn source(self, symbol: &str) -> String {
        self.source_with_callback(symbol, None)
    }

    pub(crate) fn source_with_callback(self, symbol: &str, callback: Option<&str>) -> String {
        let (dtype, _) = self.dtype.native();
        let [x, y, z] = self.block;
        let items = self.items;
        let primitive = match self.recipe {
            ReduceRecipe::Block(_, algorithm, _) => {
                let algorithm = match algorithm {
                    ReduceAlgorithm::RakingCommutativeOnly => {
                        "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY"
                    }
                    ReduceAlgorithm::Raking => "BLOCK_REDUCE_RAKING",
                    ReduceAlgorithm::WarpReductions => "BLOCK_REDUCE_WARP_REDUCTIONS",
                };
                format!("cub::BlockReduce<value_type, {x}, cub::{algorithm}, {y}, {z}>")
            }
            ReduceRecipe::Warp(_, _) => {
                format!("cub::WarpReduce<value_type, {}>", self.width.unwrap())
            }
        };
        let counted = self.recipe.form() == ReduceForm::ScalarPrefix;
        let controls = if callback.is_some() {
            ", const void* environment"
        } else if counted {
            ", int valid"
        } else {
            ""
        };
        let group_size = self.width.unwrap_or(x * y * z);
        let validate_count = if counted {
            format!("if (valid < 1 || valid > {group_size}) __trap();")
        } else {
            String::new()
        };
        let operand = if self.recipe.form() == ReduceForm::Payload {
            "local_input"
        } else {
            "local_input[0]"
        };
        let count = if counted { ", valid" } else { "" };
        let callback_declaration = callback.map_or_else(String::new, |name| {
            format!("extern \"C\" __device__ int {name}(const void*, int, int);\n")
        });
        let callback_functor = callback.map_or_else(String::new, |name| {
            format!(
                "struct binary_op {{ const void* environment;\n  __device__ int operator()(int a, int b) const {{ return {name}(environment, a, b); }}\n}};"
            )
        });
        let call = if callback.is_some() {
            format!("operation.Reduce({operand}, binary_op{{environment}})")
        } else if self.recipe.operator() == ReduceOperator::Sum {
            format!("operation.Sum({operand}{count})")
        } else {
            format!(
                "operation.Reduce({operand}, {}{{}}{count})",
                self.recipe.operator().functor()
            )
        };
        let root = match self.width {
            None => "thread_rank == 0".into(),
            Some(w) => format!(
                "thread_rank % {} == 0",
                if w.is_power_of_two() { w } else { 32 }
            ),
        };
        format!(
            r#"
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
{callback_declaration}
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
{callback_functor}
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits{controls}) {{
  using namespace {symbol}_detail;
  const unsigned int thread_rank = cub::RowMajorTid({x}, {y}, {z});
  output_bits[0] = 0;
  {validate_count}
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_input[i] = input[i];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  const value_type aggregate = {call};
  if ({root}) {{
    output[0] = aggregate;
    output_bits[0] = 1;
  }}
}}
"#
        )
    }
}
