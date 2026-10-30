// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated numeric Block adjacent differences with preserved input.

use crate::{Dtype, Error, Result};

/// Direction, tile extent and optional boundary value for adjacent subtraction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdjacentDifferenceRecipe {
    /// Subtract each predecessor, copying the first tile item unchanged.
    LeftFull,
    /// Subtract each predecessor, including a supplied tile predecessor.
    LeftFullPredecessor,
    /// Subtract within a valid prefix, copying its first item and the suffix.
    LeftPartial,
    /// Subtract within a valid prefix using a supplied tile predecessor.
    LeftPartialPredecessor,
    /// Subtract each successor, copying the last tile item unchanged.
    RightFull,
    /// Subtract each successor, including a supplied tile successor.
    RightFullSuccessor,
    /// Subtract within a valid prefix, copying its last item and the suffix.
    RightPartial,
}

impl AdjacentDifferenceRecipe {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::LeftFull => "adjacent_difference_left_full_validity_bits_v1",
            Self::LeftFullPredecessor => {
                "adjacent_difference_left_full_predecessor_validity_bits_v1"
            }
            Self::LeftPartial => "adjacent_difference_left_partial_validity_bits_v1",
            Self::LeftPartialPredecessor => {
                "adjacent_difference_left_partial_predecessor_validity_bits_v1"
            }
            Self::RightFull => "adjacent_difference_right_full_validity_bits_v1",
            Self::RightFullSuccessor => "adjacent_difference_right_full_successor_validity_bits_v1",
            Self::RightPartial => "adjacent_difference_right_partial_validity_bits_v1",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "block" {
            return Err(Error::message(
                "Adjacent Difference requires a Block recipe",
            ));
        }
        match operation {
            "adjacent_difference_left_full_validity_bits_v1" => Ok(Self::LeftFull),
            "adjacent_difference_left_full_predecessor_validity_bits_v1" => {
                Ok(Self::LeftFullPredecessor)
            }
            "adjacent_difference_left_partial_validity_bits_v1" => Ok(Self::LeftPartial),
            "adjacent_difference_left_partial_predecessor_validity_bits_v1" => {
                Ok(Self::LeftPartialPredecessor)
            }
            "adjacent_difference_right_full_validity_bits_v1" => Ok(Self::RightFull),
            "adjacent_difference_right_full_successor_validity_bits_v1" => {
                Ok(Self::RightFullSuccessor)
            }
            "adjacent_difference_right_partial_validity_bits_v1" => Ok(Self::RightPartial),
            _ => Err(Error::message("unsupported Adjacent Difference operation")),
        }
    }

    pub(crate) fn counted(self) -> bool {
        matches!(
            self,
            Self::LeftPartial | Self::LeftPartialPredecessor | Self::RightPartial
        )
    }

    pub(crate) fn has_neighbor(self) -> bool {
        matches!(
            self,
            Self::LeftFullPredecessor | Self::LeftPartialPredecessor | Self::RightFullSuccessor
        )
    }
}

/// Validated Block adjacent subtraction with fresh, fully initialized output.
///
/// Native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)`, followed by an `i32` valid-item
/// count for partial recipes and then a value of exactly `T` for recipes with
/// a tile predecessor or successor. Controls are uniform across the Block.
/// Counts lie in `0..=block_threads*items`; invalid counts trap before the call.
/// A partial right recipe has no successor parameter.
///
/// Each lane supplies `items` initialized values and `ceil(items/8)` packed,
/// low-bit-first validity bytes. Every input bit is checked before any numeric
/// read or collective call, even for a zero count. Input values and validity
/// are preserved. Output has the same capacities, is disjoint from input and
/// scratch, and receives every value and validity bit; unused high bits are zero.
///
/// For x-fastest Block rank `r` and item `i`, position `r*items+i` uses native
/// `cuda::std::minus<T>` with its left or right neighbor. Without an explicit
/// neighbor, the corresponding tile boundary is copied unchanged. Partial
/// recipes copy every slot beyond the valid prefix and its unsupplied boundary.
/// A zero count copies the complete input; it still executes the CUB collective.
///
/// Every member of the exact Block participates in converged control flow.
/// Pointers remain live with their natural alignment and stated capacities.
/// Scratch uses the reported actual CUB TempStorage, including both endpoint
/// planes. The caller owns its alignment and lifetime and synchronizes Block
/// members before reuse. The provider adds no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdjacentDifferenceRequest {
    pub(crate) recipe: AdjacentDifferenceRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl AdjacentDifferenceRequest {
    /// Validate adjacent subtraction for an exact complete Block.
    ///
    /// `dtype` selects one of the ten numeric types; `recipe` selects one of
    /// the seven boundary/count combinations. `block` gives positive XYZ axes
    /// and `items` is the fixed per-lane input and output extent.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, more than 1024 threads, zero items, or
    /// a complete tile exceeding CUB's signed-int indexing range.
    pub fn block(
        dtype: Dtype,
        recipe: AdjacentDifferenceRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message(
                    "Adjacent Difference requires 1..=1024 block threads with positive axes",
                )
            })?;
        if items == 0 || items > i32::MAX as u32 / threads {
            return Err(Error::message(
                "Adjacent Difference requires a positive tile no larger than i32::MAX",
            ));
        }
        Ok(Self {
            recipe,
            dtype,
            block,
            items,
        })
    }

    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.block
    }

    pub(crate) fn source(self, symbol: &str) -> String {
        let (dtype, _) = self.dtype.native();
        let [x, y, z] = self.block;
        let items = self.items;
        let tile = x * y * z * items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let mut parameters = String::new();
        let mut controls = String::new();
        let mut guard = String::new();
        if self.recipe.counted() {
            parameters.push_str(", int valid_items");
            controls.push_str(", valid_items");
            guard = format!("if (valid_items < 0 || valid_items > {tile}) __trap();");
        }
        if self.recipe.has_neighbor() {
            parameters.push_str(&format!(", {dtype} neighbor"));
            controls.push_str(", neighbor");
        }
        let method =
            match self.recipe {
                AdjacentDifferenceRecipe::LeftFull
                | AdjacentDifferenceRecipe::LeftFullPredecessor => "SubtractLeft",
                AdjacentDifferenceRecipe::LeftPartial
                | AdjacentDifferenceRecipe::LeftPartialPredecessor => "SubtractLeftPartialTile",
                AdjacentDifferenceRecipe::RightFull
                | AdjacentDifferenceRecipe::RightFullSuccessor => "SubtractRight",
                AdjacentDifferenceRecipe::RightPartial => "SubtractRightPartialTile",
            };
        format!(
            r#"
#include <cub/block/block_adjacent_difference.cuh>
#include <cuda/std/functional>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = cub::BlockAdjacentDifference<value_type, {x}, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits{parameters}) {{
  using namespace {symbol}_detail;
  {guard}
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_input[i] = input[i];
  value_type local_output[{items}];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.{method}(local_input, local_output, ::cuda::std::minus<value_type>{{}}{controls});
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    output[i] = local_output[i];
    output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
  }}
}}
"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_and_complete_tile_index_bounds() {
        for (shape, items) in [
            ([1, 1, 1], i32::MAX as u32),
            ([33, 1, 1], 9),
            ([4, 3, 2], 4),
            ([1024, 1, 1], i32::MAX as u32 / 1024),
        ] {
            assert!(
                AdjacentDifferenceRequest::block(
                    Dtype::I8,
                    AdjacentDifferenceRecipe::LeftFull,
                    shape,
                    items
                )
                .is_ok()
            );
        }
        for (shape, items) in [
            ([0, 1, 1], 1),
            ([1, 0, 1], 1),
            ([1, 1, 0], 1),
            ([1025, 1, 1], 1),
            ([u32::MAX, 2, 2], 1),
            ([1, 1, 1], 0),
            ([1, 1, 1], i32::MAX as u32 + 1),
            ([1, 1, 1], u32::MAX),
            ([2, 1, 1], i32::MAX as u32),
            ([1024, 1, 1], i32::MAX as u32 / 1024 + 1),
        ] {
            assert!(
                AdjacentDifferenceRequest::block(
                    Dtype::F64,
                    AdjacentDifferenceRecipe::RightPartial,
                    shape,
                    items
                )
                .is_err()
            );
        }
    }
}
