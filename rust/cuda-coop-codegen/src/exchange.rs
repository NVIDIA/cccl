// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block and Warp exchanges with preserved input and fresh output.

use crate::{Dtype, Error, Result};

/// Direction of the redistribution of a group's complete tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExchangeMode {
    /// Convert striped input positions to blocked output positions.
    StripedToBlocked,
    /// Convert blocked input positions to striped output positions.
    BlockedToStriped,
}

/// Static group scope and direction for an out-of-place Exchange.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExchangeRecipe {
    /// Exchange across a complete Block, without warp time slicing.
    Block(ExchangeMode),
    /// Exchange within an aligned Warp using shared memory.
    Warp(ExchangeMode),
}

impl ExchangeRecipe {
    fn mode(self) -> ExchangeMode {
        match self {
            Self::Block(mode) | Self::Warp(mode) => mode,
        }
    }

    pub(crate) fn token(self) -> &'static str {
        match self.mode() {
            ExchangeMode::StripedToBlocked => "exchange_striped_to_blocked_validity_bits_v1",
            ExchangeMode::BlockedToStriped => "exchange_blocked_to_striped_validity_bits_v1",
        }
    }

    pub(crate) fn algorithm_token(self) -> &'static str {
        match self {
            Self::Block(_) => "block",
            Self::Warp(_) => "warp",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        let mode = match operation {
            "exchange_striped_to_blocked_validity_bits_v1" => ExchangeMode::StripedToBlocked,
            "exchange_blocked_to_striped_validity_bits_v1" => ExchangeMode::BlockedToStriped,
            _ => return Err(Error::message("unsupported Exchange operation")),
        };
        match algorithm {
            "block" => Ok(Self::Block(mode)),
            "warp" => Ok(Self::Warp(mode)),
            _ => Err(Error::message("Exchange requires a Block or Warp recipe")),
        }
    }
}

/// Validated numeric Exchange with a fresh output of the same per-lane extent.
///
/// Native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)` with no scalar controls. Each lane
/// supplies `items` initialized input slots and `ceil(items/8)` packed,
/// low-bit-first validity bytes. All input bits are checked before any numeric
/// read. Input values and validity bytes are preserved.
///
/// Output contains `items` fresh slots and `ceil(items/8)` validity bytes,
/// disjoint from input and scratch. Every slot is written and marked initialized;
/// unused high validity bits are zero. All pointers must remain live and have
/// their natural alignment and the stated capacities throughout the call.
///
/// Every member of the exact Block or aligned Warp participates in converged
/// control flow. Warp widths are 1/2/4/8/16/32 within complete physical warps.
/// The reported scratch is the actual CUB TempStorage for one group. The caller
/// owns its alignment, replication and lifetime, and synchronizes group members
/// before reuse. The provider adds no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExchangeRequest {
    pub(crate) recipe: ExchangeRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
    width: Option<u32>,
}

impl ExchangeRequest {
    /// Validate a complete Block Exchange with exact XYZ geometry.
    ///
    /// # Errors
    /// Rejects a Warp recipe, zero/overflowing axes, more than 1024 threads,
    /// zero items, or a padded tile exceeding CUB's signed-int indexing range.
    pub fn block(
        dtype: Dtype,
        recipe: ExchangeRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        if !matches!(recipe, ExchangeRecipe::Block(_)) {
            return Err(Error::message("Block Exchange requires a Block recipe"));
        }
        Self::new(dtype, recipe, block, items, None)
    }

    /// Validate an aligned Warp Exchange within exact XYZ Block geometry.
    ///
    /// # Errors
    /// Rejects a Block recipe, invalid Block geometry, incomplete physical warps,
    /// widths other than 1/2/4/8/16/32, zero items, or a padded tile exceeding
    /// CUB's signed-int indexing range.
    pub fn warp(
        dtype: Dtype,
        recipe: ExchangeRecipe,
        block: [u32; 3],
        width: u32,
        items: u32,
    ) -> Result<Self> {
        if !matches!(recipe, ExchangeRecipe::Warp(_)) {
            return Err(Error::message("Warp Exchange requires a Warp recipe"));
        }
        Self::new(dtype, recipe, block, items, Some(width))
    }

    fn new(
        dtype: Dtype,
        recipe: ExchangeRecipe,
        block: [u32; 3],
        items: u32,
        width: Option<u32>,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("Exchange requires 1..=1024 block threads with positive axes")
            })?;
        if let Some(width) = width {
            if !width.is_power_of_two() || width > 32 {
                return Err(Error::message(
                    "Warp Exchange requires power-of-two width 1..=32",
                ));
            }
            if threads % 32 != 0 {
                return Err(Error::message(
                    "Warp Exchange requires complete physical warps",
                ));
            }
        }
        // CUB adds one Warp sentinel and, for power-of-two extents > 4,
        // bank-conflict padding. Its indexing and array bounds use signed int.
        let tile =
            u64::from(width.unwrap_or(threads)) * u64::from(items) + u64::from(width.is_some());
        let padding = if items > 4 && items.is_power_of_two() {
            tile >> 5
        } else {
            0
        };
        if items == 0 || tile + padding > i32::MAX as u64 {
            return Err(Error::message(
                "Exchange requires positive items and padded tile <= i32::MAX",
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
        let (dtype, _) = self.dtype.native();
        let [x, y, z] = self.block;
        let items = self.items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let (header, primitive) = match self.width {
            None => (
                "block/block_exchange.cuh",
                format!("cub::BlockExchange<value_type, {x}, {items}, false, {y}, {z}>"),
            ),
            Some(width) => (
                "warp/warp_exchange.cuh",
                format!("cub::WarpExchange<value_type, {items}, {width}, cub::WARP_EXCHANGE_SMEM>"),
            ),
        };
        let method = match self.recipe.mode() {
            ExchangeMode::StripedToBlocked => "StripedToBlocked",
            ExchangeMode::BlockedToStriped => "BlockedToStriped",
        };
        format!(
            r#"
#include <cub/{header}>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits) {{
  using namespace {symbol}_detail;
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
    fn scope_geometry_and_index_bounds() {
        let block = ExchangeRecipe::Block(ExchangeMode::StripedToBlocked);
        let warp = ExchangeRecipe::Warp(ExchangeMode::BlockedToStriped);
        assert!(ExchangeRequest::block(Dtype::I8, block, [33, 1, 1], 8).is_ok());
        assert!(ExchangeRequest::block(Dtype::F64, block, [4, 3, 2], 4).is_ok());
        assert!(ExchangeRequest::warp(Dtype::U8, warp, [8, 4, 2], 1, 9).is_ok());
        assert!(ExchangeRequest::block(Dtype::I32, warp, [32, 1, 1], 1).is_err());
        assert!(ExchangeRequest::warp(Dtype::I32, block, [32, 1, 1], 32, 1).is_err());
        for shape in [[0, 1, 1], [1025, 1, 1], [u32::MAX, 2, 2]] {
            assert!(ExchangeRequest::block(Dtype::I32, block, shape, 1).is_err());
        }
        for width in [0, 3, 17, 31, 64] {
            assert!(ExchangeRequest::warp(Dtype::I32, warp, [64, 1, 1], width, 1).is_err());
        }
        assert!(ExchangeRequest::warp(Dtype::I32, warp, [33, 1, 1], 32, 1).is_err());
        assert!(ExchangeRequest::block(Dtype::I32, block, [1, 1, 1], 0).is_err());
        assert!(ExchangeRequest::block(Dtype::I32, block, [2, 1, 1], 1 << 30).is_err());
        // Warp adds a sentinel even when no bank-conflict padding is needed.
        assert!(ExchangeRequest::warp(Dtype::I32, warp, [32, 1, 1], 1, i32::MAX as u32).is_err());
    }
}
