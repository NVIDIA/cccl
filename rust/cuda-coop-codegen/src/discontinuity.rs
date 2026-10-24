// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block discontinuity flags with preserved numeric input.

use crate::{Dtype, Error, Result};

/// Head/tail results and optional neighbors outside a complete Block tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscontinuityRecipe {
    /// Head flags, with the first tile flag set to one.
    Heads,
    /// Head flags, comparing the first item with a supplied predecessor.
    HeadsPredecessor,
    /// Tail flags, with the last tile flag set to one.
    Tails,
    /// Tail flags, comparing the last item with a supplied successor.
    TailsSuccessor,
    /// Separate head and tail flags, with both tile boundaries set to one.
    HeadsAndTails,
    /// Both results with a supplied predecessor and an implicit final boundary.
    HeadsAndTailsPredecessor,
    /// Both results with an implicit first boundary and a supplied successor.
    HeadsAndTailsSuccessor,
    /// Both results with supplied predecessor and successor values.
    HeadsAndTailsBoth,
}

impl DiscontinuityRecipe {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Heads => "discontinuity_heads_validity_bits_v1",
            Self::HeadsPredecessor => "discontinuity_heads_predecessor_validity_bits_v1",
            Self::Tails => "discontinuity_tails_validity_bits_v1",
            Self::TailsSuccessor => "discontinuity_tails_successor_validity_bits_v1",
            Self::HeadsAndTails => "discontinuity_heads_and_tails_validity_bits_v1",
            Self::HeadsAndTailsPredecessor => {
                "discontinuity_heads_and_tails_predecessor_validity_bits_v1"
            }
            Self::HeadsAndTailsSuccessor => {
                "discontinuity_heads_and_tails_successor_validity_bits_v1"
            }
            Self::HeadsAndTailsBoth => "discontinuity_heads_and_tails_both_validity_bits_v1",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "block" {
            return Err(Error::message("Discontinuity requires a Block recipe"));
        }
        match operation {
            "discontinuity_heads_validity_bits_v1" => Ok(Self::Heads),
            "discontinuity_heads_predecessor_validity_bits_v1" => Ok(Self::HeadsPredecessor),
            "discontinuity_tails_validity_bits_v1" => Ok(Self::Tails),
            "discontinuity_tails_successor_validity_bits_v1" => Ok(Self::TailsSuccessor),
            "discontinuity_heads_and_tails_validity_bits_v1" => Ok(Self::HeadsAndTails),
            "discontinuity_heads_and_tails_predecessor_validity_bits_v1" => {
                Ok(Self::HeadsAndTailsPredecessor)
            }
            "discontinuity_heads_and_tails_successor_validity_bits_v1" => {
                Ok(Self::HeadsAndTailsSuccessor)
            }
            "discontinuity_heads_and_tails_both_validity_bits_v1" => Ok(Self::HeadsAndTailsBoth),
            _ => Err(Error::message("unsupported Discontinuity operation")),
        }
    }

    pub(crate) fn paired(self) -> bool {
        matches!(
            self,
            Self::HeadsAndTails
                | Self::HeadsAndTailsPredecessor
                | Self::HeadsAndTailsSuccessor
                | Self::HeadsAndTailsBoth
        )
    }

    pub(crate) fn predecessor(self) -> bool {
        matches!(
            self,
            Self::HeadsPredecessor | Self::HeadsAndTailsPredecessor | Self::HeadsAndTailsBoth
        )
    }

    pub(crate) fn successor(self) -> bool {
        matches!(
            self,
            Self::TailsSuccessor | Self::HeadsAndTailsSuccessor | Self::HeadsAndTailsBoth
        )
    }
}

/// Validated Block discontinuity with one or two fresh, initialized i32 outputs.
///
/// Single-result calls use `(const T* input, i32* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)`. Paired calls use
/// `(const T* input, i32* heads, u8* scratch, i32* tails,
/// const u8* input_bits, u8* heads_bits, u8* tails_bits)`.
/// A supplied predecessor and/or successor follows these pointers, in that
/// order, as values of exactly `T`, uniform across the Block. No count exists.
///
/// Every lane supplies `items` initialized numeric values and `ceil(items/8)`
/// low-bit-first validity bytes. All input bits are checked before any numeric
/// read or collective call. Each output has `items` i32 slots and its own
/// equally sized bitmap. The provider initializes every flag, marks every slot
/// valid and clears unused high bits. Input values and validity are preserved.
/// All input, output, bitmap and scratch buffers must be disjoint, live and
/// naturally aligned, with the stated capacities.
///
/// Flags follow blocked order with x-fastest Block rank. A head compares an
/// item with its predecessor; a tail compares it with its successor. Native
/// `cuda::std::not_equal_to<T>` produces flags zero or one, including its usual
/// floating-point NaN and signed-zero behavior. A tile boundary without a
/// supplied neighbor is flagged one. Paired results are ordered heads, tails.
///
/// Every member of the exact Block participates in converged control flow.
/// Scratch is the reported actual CUB TempStorage, including both endpoint
/// planes. The caller provides its alignment/lifetime and synchronizes Block
/// members before reuse. The provider adds no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiscontinuityRequest {
    pub(crate) recipe: DiscontinuityRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl DiscontinuityRequest {
    /// Validate a complete Block's single or paired discontinuity results.
    ///
    /// `dtype` selects one of the ten numeric input types; every output is i32.
    /// `recipe` selects one of eight mode/neighbor combinations. `block` gives
    /// positive XYZ axes; `items` is the per-lane extent of input and each output.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, more than 1024 threads, zero items, or
    /// a complete tile exceeding CUB's signed-int indexing range.
    pub fn block(
        dtype: Dtype,
        recipe: DiscontinuityRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("Discontinuity requires 1..=1024 block threads with positive axes")
            })?;
        if items == 0 || items > i32::MAX as u32 / threads {
            return Err(Error::message(
                "Discontinuity requires a positive tile no larger than i32::MAX",
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
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let mut controls = String::new();
        if self.recipe.predecessor() {
            controls.push_str(&format!(", {dtype} predecessor"));
        }
        if self.recipe.successor() {
            controls.push_str(&format!(", {dtype} successor"));
        }
        let (tail_parameter, tail_bits_parameter, clear_tails, declare_tails, copy_tails) = if self
            .recipe
            .paired()
        {
            (
                ", int* tails",
                ", unsigned char* tails_bits",
                "tails_bits[i] = 0;",
                format!("int local_tails[{items}];"),
                "tails[i] = local_tails[i]; tails_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));",
            )
        } else {
            ("", "", "", String::new(), "")
        };
        let invocation = match self.recipe {
            DiscontinuityRecipe::Heads => "FlagHeads(local_output, local_input, flag_op)",
            DiscontinuityRecipe::HeadsPredecessor => {
                "FlagHeads(local_output, local_input, flag_op, predecessor)"
            }
            DiscontinuityRecipe::Tails => "FlagTails(local_output, local_input, flag_op)",
            DiscontinuityRecipe::TailsSuccessor => {
                "FlagTails(local_output, local_input, flag_op, successor)"
            }
            DiscontinuityRecipe::HeadsAndTails => {
                "FlagHeadsAndTails(local_output, local_tails, local_input, flag_op)"
            }
            DiscontinuityRecipe::HeadsAndTailsPredecessor => {
                "FlagHeadsAndTails(local_output, predecessor, local_tails, local_input, flag_op)"
            }
            DiscontinuityRecipe::HeadsAndTailsSuccessor => {
                "FlagHeadsAndTails(local_output, local_tails, successor, local_input, flag_op)"
            }
            DiscontinuityRecipe::HeadsAndTailsBoth => {
                "FlagHeadsAndTails(local_output, predecessor, local_tails, successor, local_input, flag_op)"
            }
        };
        format!(
            r#"
#include <cub/block/block_discontinuity.cuh>
#include <cuda/std/functional>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = cub::BlockDiscontinuity<value_type, {x}, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, int* output,
    unsigned char* scratch{tail_parameter}, const unsigned char* input_bits,
    unsigned char* output_bits{tail_bits_parameter}{controls}) {{
  using namespace {symbol}_detail;
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) {{ output_bits[i] = 0; {clear_tails} }}
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_input[i] = input[i];
  int local_output[{items}];
  {declare_tails}
  const auto flag_op = ::cuda::std::not_equal_to<value_type>{{}};
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.{invocation};
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    output[i] = local_output[i];
    output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    {copy_tails}
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
                DiscontinuityRequest::block(Dtype::I8, DiscontinuityRecipe::Heads, shape, items)
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
            ([2, 1, 1], i32::MAX as u32),
            ([1024, 1, 1], i32::MAX as u32 / 1024 + 1),
        ] {
            assert!(
                DiscontinuityRequest::block(
                    Dtype::F64,
                    DiscontinuityRecipe::HeadsAndTailsBoth,
                    shape,
                    items,
                )
                .is_err()
            );
        }
    }
}
