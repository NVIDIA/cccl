// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Warp batched reductions with partial, derived-extent outputs.

use crate::{Dtype, Error, ReduceOperator, Result};

/// Distribution of batch results among the members of a Warp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceBatchedLayout {
    /// Lane r receives batches r, r + width, and so on.
    Striped,
    /// Lane r receives consecutive batches starting at r * ceil(batches/width).
    Blocked,
}

/// Static batched-reduction operator and output layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReduceBatchedRecipe {
    operator: ReduceOperator,
    layout: ReduceBatchedLayout,
}

impl ReduceBatchedRecipe {
    /// Select a builtin operator and the distribution of fresh output slots.
    pub fn new(operator: ReduceOperator, layout: ReduceBatchedLayout) -> Self {
        Self { operator, layout }
    }

    pub(crate) fn token(self) -> String {
        let layout = match self.layout {
            ReduceBatchedLayout::Striped => "striped",
            ReduceBatchedLayout::Blocked => "blocked",
        };
        format!(
            "reduce_batched_{}_{layout}_validity_bits_v1",
            self.operator.token()
        )
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "warp" {
            return Err(Error::message("batched Reduce requires the Warp algorithm"));
        }
        let operation = operation
            .strip_prefix("reduce_batched_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
            .ok_or_else(|| Error::message("unsupported batched Reduce operation"))?;
        let (operator, layout) = operation
            .rsplit_once('_')
            .ok_or_else(|| Error::message("missing batched Reduce output layout"))?;
        let layout = match layout {
            "striped" => ReduceBatchedLayout::Striped,
            "blocked" => ReduceBatchedLayout::Blocked,
            _ => return Err(Error::message("unsupported batched Reduce output layout")),
        };
        Ok(Self::new(ReduceOperator::parse(operator)?, layout))
    }
}

/// Validated numeric batched reduction within a physical or logical Warp.
///
/// Native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)` with no scalar controls. Each lane
/// supplies `batches` initialized input slots and `ceil(batches/8)` packed,
/// low-bit-first validity bytes. Every input bit is checked before any numeric
/// read. Inputs and their bits are preserved.
///
/// Output has exactly `ceil(batches/width)` slots and enough validity bytes for
/// that extent. Output values and bits are fresh, disjoint from input, live and
/// aligned. Only slots corresponding to real batches are written and marked
/// initialized; the other slots remain uninitialized, with zero validity bits.
/// All output validity bytes, including unused high bits, are reset.
///
/// Every member of the aligned Warp participates. Blocks contain complete
/// physical warps, and supported widths are 1/2/4/8/16/32. Scratch describes
/// the actual CUB TempStorage for one group, including its nonzero empty-type
/// size. The caller owns its replication, alignment, lifetime and reuse
/// completion. This provider adds no completion barrier or result broadcast.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReduceBatchedRequest {
    pub(crate) recipe: ReduceBatchedRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    width: u32,
    batches: u32,
}

impl ReduceBatchedRequest {
    /// Validate one batched reduction in an aligned Warp with exact XYZ geometry.
    ///
    /// The input extent is `batches`; the output extent is derived from `width`.
    /// Bitwise operators require integer elements.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, more than 1024 threads, incomplete physical
    /// warps, widths other than 1/2/4/8/16/32, zero batches, or a padded output
    /// extent exceeding CUB's signed-int indexing range.
    pub fn warp(
        dtype: Dtype,
        recipe: ReduceBatchedRecipe,
        block: [u32; 3],
        width: u32,
        batches: u32,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("batched Reduce requires 1..=1024 block threads with positive axes")
            })?;
        if threads % 32 != 0 {
            return Err(Error::message(
                "batched Reduce requires complete physical warps",
            ));
        }
        if !width.is_power_of_two() || width > 32 {
            return Err(Error::message(
                "batched Reduce requires power-of-two width 1..=32",
            ));
        }
        let outputs = batches / width + u32::from(batches % width != 0);
        if batches == 0 || u64::from(outputs) * u64::from(width) > i32::MAX as u64 {
            return Err(Error::message(
                "batched Reduce requires positive batches and padded extent <= i32::MAX",
            ));
        }
        if matches!(dtype, Dtype::F32 | Dtype::F64)
            && matches!(
                recipe.operator,
                ReduceOperator::BitAnd | ReduceOperator::BitOr | ReduceOperator::BitXor
            )
        {
            return Err(Error::message(
                "bitwise batched Reduce operators require integer elements",
            ));
        }
        Ok(Self {
            recipe,
            dtype,
            block,
            width,
            batches,
        })
    }

    /// Derived per-lane output capacity; individual slots may be invalid.
    pub fn outputs_per_thread(self) -> u32 {
        self.batches / self.width + u32::from(self.batches % self.width != 0)
    }

    pub(crate) fn source(self, symbol: &str) -> String {
        let (dtype, _) = self.dtype.native();
        let [x, y, z] = self.block;
        let width = self.width;
        let batches = self.batches;
        let outputs = self.outputs_per_thread();
        let output_bytes = outputs / 8 + u32::from(outputs % 8 != 0);
        let (method, batch_index) = match self.recipe.layout {
            ReduceBatchedLayout::Striped => ("ReduceToStriped", format!("lane_rank + i * {width}")),
            ReduceBatchedLayout::Blocked => {
                ("ReduceToBlocked", format!("lane_rank * {outputs} + i"))
            }
        };
        let functor = self.recipe.operator.functor();
        format!(
            r#"
#include <cub/warp/warp_reduce_batched.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = cub::WarpReduceBatched<value_type, {batches}, {width}, false>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits) {{
  using namespace {symbol}_detail;
  const unsigned int lane_rank = cub::RowMajorTid({x}, {y}, {z}) % {width};
#pragma unroll
  for (int i = 0; i < {output_bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {batches}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{batches}];
#pragma unroll
  for (int i = 0; i < {batches}; ++i) local_input[i] = input[i];
  value_type local_output[{outputs}];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.{method}(local_input, local_output, {functor}{{}});
#pragma unroll
  for (int i = 0; i < {outputs}; ++i) {{
    if ({batch_index} < {batches}) {{
      output[i] = local_output[i];
      output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }}
  }}
}}
"#
        )
    }
}
