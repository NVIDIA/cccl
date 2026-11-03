// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block radix sorting and ranking with preserved input keys.

use crate::{Dtype, Error, Result};

/// Stable ordering of the selected radix bits in a complete Block tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadixSortKeysRecipe {
    /// Sort selected digits in ascending order.
    Ascending,
    /// Sort selected digits in descending order.
    Descending,
}

impl RadixSortKeysRecipe {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Ascending => "radix_sort_keys_ascending_validity_bits_v1",
            Self::Descending => "radix_sort_keys_descending_validity_bits_v1",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "block" {
            return Err(Error::message("Radix Sort Keys requires a Block recipe"));
        }
        match operation {
            "radix_sort_keys_ascending_validity_bits_v1" => Ok(Self::Ascending),
            "radix_sort_keys_descending_validity_bits_v1" => Ok(Self::Descending),
            _ => Err(Error::message("unsupported Radix Sort Keys operation")),
        }
    }
}

/// Validated integer Block radix sort with fresh, fully initialized output.
///
/// Native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits, i64 begin_bit, i64 end_bit)`.
/// The two scalar controls are uniform across the Block and must satisfy
/// `0 <= begin_bit < end_bit <= sizeof(T)*8`. Invalid bounds trap while still
/// signed 64-bit values, before narrowing or calling CUB. Bounds do not enter
/// specialization identity; the caller resolves defaults to zero and key width.
///
/// Each lane supplies `items` initialized keys and `ceil(items/8)` packed,
/// low-bit-first validity bytes. Every input bit is checked before any numeric
/// read. Input values and validity bytes are preserved. Output has the same
/// capacities, is disjoint from input and scratch, and receives every key and
/// validity bit; unused high bits are zero.
///
/// Input and output are blocked: item `i` of x-fastest rank `r` has tile
/// position `r*items+i`. Equal selected digits retain their input order in
/// either direction. CUB transforms signed keys before selecting the bit range
/// and restores their original representation after sorting.
///
/// Every member of the exact Block participates in converged control flow.
/// All pointers remain live with their natural alignment and stated capacities.
/// Scratch uses the complete queried CUB TempStorage. The caller owns its
/// alignment and lifetime and synchronizes Block members before reuse; the
/// provider adds no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadixSortKeysRequest {
    pub(crate) recipe: RadixSortKeysRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl RadixSortKeysRequest {
    /// Validate radix sorting for an exact complete XYZ Block.
    ///
    /// `dtype` is `i32`, `u32`, `i64` or `u64`; `recipe` selects ordering.
    /// `block` contains the three positive axes and `items` is the fixed
    /// per-lane input and output extent. Warp and partial tiles are unsupported.
    ///
    /// # Errors
    /// Rejects unsupported key types, zero/overflowing axes, more than 1024
    /// threads, zero items, or a tile exceeding CUB's 65,535-key counter bound.
    pub fn block(
        dtype: Dtype,
        recipe: RadixSortKeysRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        if !matches!(dtype, Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64) {
            return Err(Error::message(
                "Radix Sort Keys requires i32/u32/i64/u64 keys",
            ));
        }
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message("Radix Sort Keys requires 1..=1024 block threads with positive axes")
            })?;
        if items == 0 || items > u16::MAX as u32 / threads {
            return Err(Error::message(
                "Radix Sort Keys requires a positive tile of at most 65535 keys",
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
        let (dtype, width) = self.dtype.native();
        let bits = width * 8;
        let [x, y, z] = self.block;
        let items = self.items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let method = match self.recipe {
            RadixSortKeysRecipe::Ascending => "Sort",
            RadixSortKeysRecipe::Descending => "SortDescending",
        };
        format!(
            r#"
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_type.cuh>
#include <driver_types.h>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = cub::BlockRadixSort<value_type, {x}, {items}, cub::NullType,
    4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits,
    long long begin_bit, long long end_bit) {{
  using namespace {symbol}_detail;
  if (begin_bit < 0 || begin_bit >= end_bit || end_bit > {bits}) __trap();
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_keys[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_keys[i] = input[i];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.{method}(local_keys, static_cast<int>(begin_bit), static_cast<int>(end_bit));
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    output[i] = local_keys[i];
    output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
  }}
}}
"#
        )
    }
}

/// Stable ordering of paired keys and values over the selected key bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadixSortPairsRecipe {
    /// Sort selected key digits in ascending order, carrying their values.
    Ascending,
    /// Sort selected key digits in descending order, carrying their values.
    Descending,
}

impl RadixSortPairsRecipe {
    pub(crate) fn token(self, value: Dtype) -> String {
        let order = match self {
            Self::Ascending => "ascending",
            Self::Descending => "descending",
        };
        let value = match value {
            Dtype::I8 => "i8",
            Dtype::U8 => "u8",
            Dtype::I16 => "i16",
            Dtype::U16 => "u16",
            Dtype::I32 => "i32",
            Dtype::U32 => "u32",
            Dtype::I64 => "i64",
            Dtype::U64 => "u64",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
        };
        format!("radix_sort_pairs_{order}_{value}_validity_bits_v1")
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<(Self, Dtype)> {
        if algorithm != "block" {
            return Err(Error::message("Radix Sort Pairs requires a Block recipe"));
        }
        let unsupported = || Error::message("unsupported Radix Sort Pairs operation");
        let operation = operation
            .strip_prefix("radix_sort_pairs_")
            .and_then(|token| token.strip_suffix("_validity_bits_v1"))
            .ok_or_else(unsupported)?;
        let (recipe, value) = if let Some(value) = operation.strip_prefix("ascending_") {
            (Self::Ascending, value)
        } else if let Some(value) = operation.strip_prefix("descending_") {
            (Self::Descending, value)
        } else {
            return Err(unsupported());
        };
        let value = match value {
            "i8" => Dtype::I8,
            "u8" => Dtype::U8,
            "i16" => Dtype::I16,
            "u16" => Dtype::U16,
            "i32" => Dtype::I32,
            "u32" => Dtype::U32,
            "i64" => Dtype::I64,
            "u64" => Dtype::U64,
            "f32" => Dtype::F32,
            "f64" => Dtype::F64,
            _ => return Err(unsupported()),
        };
        Ok((recipe, value))
    }
}

/// Validated Block radix sort with preserved inputs and fresh paired outputs.
///
/// Native calls use `(const K* keys_in, K* keys_out, u8* scratch,
/// const V* values_in, V* values_out, const u8* keys_in_bits, u8* keys_out_bits,
/// const u8* values_in_bits, u8* values_out_bits, i64 begin_bit, i64 end_bit)`.
/// Keys are `i32/u32/i64/u64`; values independently use any [`Dtype`]. Both
/// arrays have the same positive per-lane extent. The two uniform signed64
/// bounds satisfy `0 <= begin_bit < end_bit <= sizeof(K)*8` before narrowing.
/// Defaults are resolved by the caller, and bounds do not enter identity.
///
/// Each lane supplies initialized, naturally aligned key and value arrays
/// with separate packed low-bit-first validity maps of `ceil(items/8)` bytes.
/// Every bit in both input maps is checked before reading either numeric
/// array. Sorting uses private copies and preserves both inputs and maps.
/// Output arrays, output validity maps and scratch are mutually disjoint and
/// disjoint from the input arrays and maps. Read-only inputs may alias one another.
/// Every output slot is written and marked valid; unused high bits are zero.
///
/// Input and output are blocked: item `i` at x-fastest rank `r` has tile
/// position `r*items+i`. CUB transforms signed keys before selecting digits
/// and restores them afterward. Equal selected digits retain input order in
/// both directions, and each value follows its associated key unchanged.
///
/// All exact Block members participate in converged control flow. Pointers
/// remain live with their declared capacities and natural alignment. Scratch
/// is the complete queried pair primitive TempStorage, including the value
/// exchange; it cannot be inferred from the keys-only layout. The caller
/// synchronizes members before scratch reuse; no trailing barrier is added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadixSortPairsRequest {
    pub(crate) recipe: RadixSortPairsRecipe,
    pub(crate) key_dtype: Dtype,
    pub(crate) value_dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl RadixSortPairsRequest {
    /// Validate independently typed key/value sorting for an exact XYZ Block.
    ///
    /// `key_dtype` is one of the four integer key types; `value_dtype` is any
    /// supported numeric type. `recipe` selects stable ordering. `items` is
    /// the common positive per-lane extent of both inputs and both outputs.
    ///
    /// # Errors
    /// Rejects key types, shapes and tile capacities rejected by
    /// [`RadixSortKeysRequest::block`], including `Block threads * items > 65535`.
    pub fn block(
        key_dtype: Dtype,
        value_dtype: Dtype,
        recipe: RadixSortPairsRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        let order = match recipe {
            RadixSortPairsRecipe::Ascending => RadixSortKeysRecipe::Ascending,
            RadixSortPairsRecipe::Descending => RadixSortKeysRecipe::Descending,
        };
        RadixSortKeysRequest::block(key_dtype, order, block, items)?;
        Ok(Self {
            recipe,
            key_dtype,
            value_dtype,
            block,
            items,
        })
    }

    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.block
    }

    pub(crate) fn source(self, symbol: &str) -> String {
        let (key, width) = self.key_dtype.native();
        let (value, _) = self.value_dtype.native();
        let bits = width * 8;
        let [x, y, z] = self.block;
        let items = self.items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let method = match self.recipe {
            RadixSortPairsRecipe::Ascending => "Sort",
            RadixSortPairsRecipe::Descending => "SortDescending",
        };
        format!(
            r#"
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <driver_types.h>
namespace {symbol}_detail {{
using key_type = {key};
using value_type = {value};
using primitive = cub::BlockRadixSort<key_type, {x}, {items}, value_type,
    4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {key}* keys_in, {key}* keys_out,
    unsigned char* scratch, const {value}* values_in, {value}* values_out,
    const unsigned char* keys_in_bits, unsigned char* keys_out_bits,
    const unsigned char* values_in_bits, unsigned char* values_out_bits,
    long long begin_bit, long long end_bit) {{
  using namespace {symbol}_detail;
  if (begin_bit < 0 || begin_bit >= end_bit || end_bit > {bits}) __trap();
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) {{
    keys_out_bits[i] = 0;
    values_out_bits[i] = 0;
  }}
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if ((keys_in_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
    if ((values_in_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  }}
  key_type local_keys[{items}];
  value_type local_values[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    local_keys[i] = keys_in[i];
    local_values[i] = values_in[i];
  }}
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.{method}(local_keys, local_values, static_cast<int>(begin_bit), static_cast<int>(end_bit));
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    keys_out[i] = local_keys[i];
    values_out[i] = local_values[i];
    keys_out_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    values_out_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
  }}
}}
"#
        )
    }
}

/// Stable ordering and static digit width for Block radix ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadixRankKeysRecipe {
    /// Rank selected digits in ascending order with a width of 1 through 8 bits.
    Ascending(u32),
    /// Rank selected digits in descending order with a width of 1 through 8 bits.
    Descending(u32),
}

impl RadixRankKeysRecipe {
    fn bits(self) -> u32 {
        match self {
            Self::Ascending(bits) | Self::Descending(bits) => bits,
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        if !(1..=8).contains(&self.bits()) {
            return Err(Error::message("Radix Rank Keys requires 1..=8 radix bits"));
        }
        Ok(())
    }

    pub(crate) fn token(self) -> String {
        let order = match self {
            Self::Ascending(_) => "ascending",
            Self::Descending(_) => "descending",
        };
        format!("radix_rank_keys_{order}_{}_validity_bits_v1", self.bits())
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if algorithm != "block" {
            return Err(Error::message("Radix Rank Keys requires a Block recipe"));
        }
        let unsupported = || Error::message("unsupported Radix Rank Keys operation");
        let operation = operation
            .strip_prefix("radix_rank_keys_")
            .and_then(|token| token.strip_suffix("_validity_bits_v1"))
            .ok_or_else(unsupported)?;
        let (descending, bits) = if let Some(bits) = operation.strip_prefix("ascending_") {
            (false, bits)
        } else if let Some(bits) = operation.strip_prefix("descending_") {
            (true, bits)
        } else {
            return Err(unsupported());
        };
        let bits = match bits {
            "1" => 1,
            "2" => 2,
            "3" => 3,
            "4" => 4,
            "5" => 5,
            "6" => 6,
            "7" => 7,
            "8" => 8,
            _ => return Err(unsupported()),
        };
        Ok(if descending {
            Self::Descending(bits)
        } else {
            Self::Ascending(bits)
        })
    }
}

/// Validated integer Block ranking with fresh i32 ranks in input-slot order.
///
/// Native calls use `(const T* keys, i32* ranks, u8* scratch,
/// const u8* key_bits, u8* rank_bits, i64 begin_bit, i64 end_constraint)`.
/// Both scalar controls are uniform across the Block. The static digit width
/// is 1 through 8; `begin_bit` must be in `0..=sizeof(T)*8-width`.
/// `end_constraint` is -1 for an omitted end, or an explicit static endpoint
/// equal to `begin_bit+width` and at most the key width. Checks use signed64
/// arithmetic before narrowing, reading keys or calling CUB.
///
/// Each lane supplies `items` initialized keys and `ceil(items/8)` packed,
/// low-bit-first validity bytes. Every input bit is checked before any numeric
/// read. Keys and their validity map remain unchanged. Output contains `items`
/// naturally aligned i32 ranks with a separate packed validity map, independent
/// of key width/alignment. All output slots become valid; padding bits are zero.
/// Output arrays, output maps and scratch are mutually disjoint and disjoint
/// from the input arrays and maps, which remain live throughout the call.
///
/// Ranks describe stable selected-digit order in the complete blocked tile:
/// input slot `i` at x-fastest rank `r` has position `r*items+i`. Signed keys
/// are transformed by flipping their sign bit before digit extraction. Equal
/// digits retain input order in either direction. The ranks form a permutation
/// of `0..Block_threads*items`, stored in original input-slot order.
///
/// Every member of the exact Block participates in converged control flow.
/// Scratch is the queried BlockRadixRank TempStorage, not sort/exchange storage.
/// The caller supplies its alignment and lifetime and synchronizes members
/// before reuse. No trailing completion barrier is added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RadixRankKeysRequest {
    pub(crate) recipe: RadixRankKeysRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl RadixRankKeysRequest {
    /// Validate stable ranking for an exact complete XYZ Block.
    ///
    /// `dtype` is i32/u32/i64/u64; `recipe` selects ordering and static width.
    /// `block` contains positive XYZ axes and `items` is the positive per-lane
    /// extent of keys and ranks. Runtime begin/end controls do not enter identity.
    ///
    /// # Errors
    /// Rejects widths outside 1..=8 and the key types, shapes and tile capacities
    /// rejected by [`RadixSortKeysRequest::block`], including tiles above 65535 keys.
    pub fn block(
        dtype: Dtype,
        recipe: RadixRankKeysRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        recipe.validate()?;
        RadixSortKeysRequest::block(dtype, RadixSortKeysRecipe::Ascending, block, items)?;
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
        let (dtype, width) = self.dtype.native();
        let bits = width * 8;
        let unsigned = if width == 4 {
            "unsigned int"
        } else {
            "unsigned long long"
        };
        let transform = if matches!(self.dtype, Dtype::I32 | Dtype::I64) {
            format!(" ^ (unsigned_type{{1}} << {})", bits - 1)
        } else {
            String::new()
        };
        let [x, y, z] = self.block;
        let items = self.items;
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let radix_bits = self.recipe.bits();
        let descending = matches!(self.recipe, RadixRankKeysRecipe::Descending(_));
        format!(
            r#"
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/block/block_scan.cuh>
#include <driver_types.h>
namespace {symbol}_detail {{
using unsigned_type = {unsigned};
using primitive = cub::BlockRadixRank<{x}, {radix_bits}, {descending}, true,
    cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, {y}, {z}>;
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {dtype}* keys, int* ranks,
    unsigned char* scratch, const unsigned char* key_bits, unsigned char* rank_bits,
    long long begin_bit, long long end_constraint) {{
  using namespace {symbol}_detail;
  if (begin_bit < 0 || begin_bit > {bits} - {radix_bits} ||
      (end_constraint != -1 &&
       (end_constraint < {radix_bits} || end_constraint > {bits} ||
        begin_bit != end_constraint - {radix_bits}))) __trap();
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((key_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  unsigned_type local_keys[{items}];
  int local_ranks[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    local_keys[i] = static_cast<unsigned_type>(keys[i]){transform};
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.RankKeys(local_keys, local_ranks,
      cub::BFEDigitExtractor<unsigned_type>(static_cast<int>(begin_bit), {radix_bits}));
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) rank_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    ranks[i] = local_ranks[i];
    rank_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
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
    fn rank_key_types_and_tile_capacity() {
        for dtype in [Dtype::I32, Dtype::U32, Dtype::I64, Dtype::U64] {
            assert!(
                RadixRankKeysRequest::block(
                    dtype,
                    RadixRankKeysRecipe::Ascending(1),
                    [3, 5, 17],
                    257
                )
                .is_ok()
            );
            assert!(
                RadixRankKeysRequest::block(
                    dtype,
                    RadixRankKeysRecipe::Descending(8),
                    [256, 1, 1],
                    256
                )
                .is_err()
            );
        }
        for dtype in [
            Dtype::I8,
            Dtype::U8,
            Dtype::I16,
            Dtype::U16,
            Dtype::F32,
            Dtype::F64,
        ] {
            assert!(
                RadixRankKeysRequest::block(
                    dtype,
                    RadixRankKeysRecipe::Ascending(4),
                    [64, 1, 1],
                    4
                )
                .is_err()
            );
        }
        for (block, items) in [
            ([0, 1, 1], 1),
            ([u32::MAX, 2, 2], 1),
            ([1025, 1, 1], 1),
            ([64, 1, 1], 0),
        ] {
            assert!(
                RadixRankKeysRequest::block(
                    Dtype::I64,
                    RadixRankKeysRecipe::Ascending(4),
                    block,
                    items
                )
                .is_err()
            );
        }
    }

    #[test]
    fn key_types_and_complete_tile_counter_bounds() {
        for dtype in [Dtype::I32, Dtype::U32, Dtype::I64, Dtype::U64] {
            for (block, items) in [
                ([1, 1, 1], 65535),
                ([3, 5, 17], 257),
                ([4, 4, 4], 4),
                ([33, 1, 1], 9),
            ] {
                assert!(
                    RadixSortKeysRequest::block(
                        dtype,
                        RadixSortKeysRecipe::Ascending,
                        block,
                        items
                    )
                    .is_ok()
                );
            }
        }
        for dtype in [
            Dtype::I8,
            Dtype::U8,
            Dtype::I16,
            Dtype::U16,
            Dtype::F32,
            Dtype::F64,
        ] {
            assert!(
                RadixSortKeysRequest::block(dtype, RadixSortKeysRecipe::Ascending, [64, 1, 1], 4)
                    .is_err()
            );
        }
        for (block, items) in [
            ([0, 1, 1], 1),
            ([1, 0, 1], 1),
            ([1, 1, 0], 1),
            ([u32::MAX, 2, 2], 1),
            ([1025, 1, 1], 1),
            ([1, 1, 1], 0),
            ([1, 1, 1], 65536),
            ([256, 1, 1], 256),
            ([64, 1, 1], u32::MAX),
        ] {
            assert!(
                RadixSortKeysRequest::block(
                    Dtype::I64,
                    RadixSortKeysRecipe::Descending,
                    block,
                    items
                )
                .is_err()
            );
        }
    }
}
