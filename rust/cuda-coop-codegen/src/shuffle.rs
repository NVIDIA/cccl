// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated one-item Block shifts with preserved input and partial fresh output.

use crate::{Dtype, Error, Result};

/// Direction of a one-item shift across a Block's blocked-order tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShuffleRecipe {
    /// Read each slot's successor; the last tile slot is invalid.
    Down,
    /// Read each slot's predecessor; the first tile slot is invalid.
    Up,
}

impl ShuffleRecipe {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Down => "shuffle_down_validity_bits_v1",
            Self::Up => "shuffle_up_validity_bits_v1",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "block" {
            return Err(Error::message("Shuffle requires a Block recipe"));
        }
        match operation {
            "shuffle_down_validity_bits_v1" => Ok(Self::Down),
            "shuffle_up_validity_bits_v1" => Ok(Self::Up),
            _ => Err(Error::message("unsupported Shuffle operation")),
        }
    }
}

/// Validated numeric Block Shuffle with a fresh output of the same lane extent.
///
/// Native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)` with no scalar controls. Every lane
/// supplies `items` initialized input slots and `ceil(items/8)` packed,
/// low-bit-first validity bytes. Every input bit is checked before any numeric
/// read or collective call, including input slots discarded by the shift.
/// Input values and validity bytes are preserved.
///
/// Output has `items` slots and `ceil(items/8)` validity bytes, disjoint from
/// input and scratch. For x-fastest Block rank `r`, item `i` and extent `N`,
/// Down reads the successor of blocked position `r*N+i`; Up reads its
/// predecessor. Down leaves the last Block rank's item `N-1` unwritten and
/// invalid; Up leaves rank zero's item zero unwritten and invalid. All other
/// slots are initialized. Unused high validity bits are zero. Block1/N1 has
/// no initialized output. There is no wraparound or boundary fill.
///
/// Every member of the exact Block participates in converged control flow.
/// All pointers remain live with their natural alignment and stated capacities
/// throughout the call. Scratch has the reported actual CUB TempStorage layout;
/// the caller owns its alignment and lifetime and synchronizes Block members
/// before reuse. The provider adds no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShuffleRequest {
    pub(crate) recipe: ShuffleRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl ShuffleRequest {
    /// Validate a complete Block Shuffle with exact XYZ geometry and distance1.
    ///
    /// `dtype` selects one of the ten numeric types. `recipe` selects Down or
    /// Up. `block` gives the three positive Block axes; `items` is each lane's
    /// fixed input and output extent. Warp and scalar forms are not supported.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, more than 1024 threads, zero items, or
    /// an item extent above CUB's signed-int array indexing range.
    pub fn block(dtype: Dtype, recipe: ShuffleRecipe, block: [u32; 3], items: u32) -> Result<Self> {
        block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("Shuffle requires 1..=1024 block threads with positive axes")
            })?;
        if items == 0 || items > i32::MAX as u32 {
            return Err(Error::message("Shuffle requires items in 1..=i32::MAX"));
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
        let threads = x * y * z;
        let items = self.items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let (method, valid) = match self.recipe {
            ShuffleRecipe::Down => (
                "Down",
                format!("thread_rank != {} || i != {}", threads - 1, items - 1),
            ),
            ShuffleRecipe::Up => ("Up", "thread_rank != 0 || i != 0".into()),
        };
        format!(
            r#"
#include <cub/block/block_shuffle.cuh>
#include <cub/util_ptx.cuh>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = cub::BlockShuffle<value_type, {x}, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits) {{
  using namespace {symbol}_detail;
  const unsigned int thread_rank = cub::RowMajorTid({x}, {y}, {z});
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
  operation.{method}(local_input, local_output);
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ({valid}) {{
      output[i] = local_output[i];
      output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }}
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
    fn geometry_and_local_index_bounds() {
        for recipe in [ShuffleRecipe::Down, ShuffleRecipe::Up] {
            for (shape, items) in [
                ([1, 1, 1], 1),
                ([33, 1, 1], 9),
                ([4, 3, 2], 4),
                ([1024, 1, 1], i32::MAX as u32),
            ] {
                assert!(ShuffleRequest::block(Dtype::I8, recipe, shape, items).is_ok());
            }
            for shape in [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1025, 1, 1],
                [u32::MAX, 2, 2],
            ] {
                assert!(ShuffleRequest::block(Dtype::I32, recipe, shape, 1).is_err());
            }
            for items in [0, i32::MAX as u32 + 1, u32::MAX] {
                assert!(ShuffleRequest::block(Dtype::F64, recipe, [1, 1, 1], items).is_err());
            }
        }
    }
}
