// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block/Warp merge sorting with preserved inputs and fresh outputs.

use crate::{Dtype, Error, Result};

/// Built-in ordering and full/partial output selection for Merge Sort.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeSortKeysRecipe {
    /// Sort the complete tile in ascending order.
    AscendingFull,
    /// Sort the complete tile in descending order.
    DescendingFull,
    /// Sort a uniform valid prefix in ascending order.
    AscendingPartial,
    /// Sort a uniform valid prefix in descending order.
    DescendingPartial,
}

impl MergeSortKeysRecipe {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::AscendingFull => "merge_sort_keys_ascending_full_validity_bits_v1",
            Self::DescendingFull => "merge_sort_keys_descending_full_validity_bits_v1",
            Self::AscendingPartial => "merge_sort_keys_ascending_partial_validity_bits_v1",
            Self::DescendingPartial => "merge_sort_keys_descending_partial_validity_bits_v1",
        }
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        if !matches!(algorithm, "block" | "warp") {
            return Err(Error::message(
                "Merge Sort Keys requires a Block or Warp recipe",
            ));
        }
        match operation {
            "merge_sort_keys_ascending_full_validity_bits_v1" => Ok(Self::AscendingFull),
            "merge_sort_keys_descending_full_validity_bits_v1" => Ok(Self::DescendingFull),
            "merge_sort_keys_ascending_partial_validity_bits_v1" => Ok(Self::AscendingPartial),
            "merge_sort_keys_descending_partial_validity_bits_v1" => Ok(Self::DescendingPartial),
            _ => Err(Error::message("unsupported Merge Sort Keys operation")),
        }
    }

    pub(crate) fn partial(self) -> bool {
        matches!(self, Self::AscendingPartial | Self::DescendingPartial)
    }
}

/// Validated numeric Block/Warp merge sort with fresh same-extent output.
///
/// Full native calls use `(const T* input, T* output, u8* scratch,
/// const u8* input_bits, u8* output_bits)`. Partial calls append
/// `(i64 valid_items, T sentinel)`. These controls are uniform within the group;
/// `0 <= valid_items <= group_threads*items` is checked before narrowing or
/// numeric input reads. The sentinel must be ordered after every valid key.
/// It has exactly the key type. Floating comparisons must satisfy strict weak
/// ordering; no NaN total order or stable ordering of equal keys is promised.
///
/// Every lane supplies `items` initialized keys and `ceil(items/8)` packed,
/// low-bit-first validity bytes. All input bits are checked even for a zero
/// count. Input keys and masks are preserved; sorting uses private copies.
/// Output keys and masks are disjoint from input and scratch. Full calls write
/// all outputs; partial calls write only the blocked valid prefix. Other output
/// bits, including padding bits, are cleared, and invalid keys are not written.
/// Block rank is `x + X*(y + Y*z)`; Warp rank is that value modulo its width.
/// Slot `i` has group index `rank*items+i`.
///
/// Every member of the power-of-two Block or aligned power-of-two Warp group
/// participates in converged control flow, including at count zero. Warp calls
/// require complete physical warps in the enclosing Block; sibling groups may
/// have different controls. All pointers remain live with their
/// stated capacities and natural alignment. Scratch uses the complete queried
/// CUB TempStorage, including its padding element. The caller synchronizes
/// members before scratch reuse; the provider adds no trailing barrier. Warp
/// callers supply an independent scratch replica for each actual group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergeSortKeysRequest {
    pub(crate) dtype: Dtype,
    pub(crate) recipe: MergeSortKeysRecipe,
    block: [u32; 3],
    items: u32,
    width: Option<u32>,
}

impl MergeSortKeysRequest {
    /// Validate built-in merge sorting for a complete XYZ Block.
    ///
    /// `dtype` is any supported numeric type, `recipe` selects ordering and
    /// full/partial validity, and `items` is the positive per-lane extent.
    ///
    /// # Errors
    /// Rejects zero/overflowing axes, a non-power-of-two thread count or more
    /// than 1024 threads, zero items, and a tile whose padded extent exceeds
    /// CUB's signed-int template capacity.
    pub fn block(
        dtype: Dtype,
        recipe: MergeSortKeysRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        Self::validate(dtype, recipe, block, items, None)
    }

    /// Validate an aligned Warp group within complete physical warps.
    ///
    /// Widths 1/2/4/8/16/32 are supported. The exact enclosing XYZ Block need
    /// not have a power-of-two total. The caller owns scratch replication and
    /// completion; this request does not establish participation.
    ///
    /// # Errors
    /// Rejects invalid axes, more than 1024 threads, incomplete physical warps,
    /// unsupported widths, zero items, and a padded group tile exceeding CUB's
    /// signed-int template capacity.
    pub fn warp(
        dtype: Dtype,
        recipe: MergeSortKeysRecipe,
        block: [u32; 3],
        width: u32,
        items: u32,
    ) -> Result<Self> {
        Self::validate(dtype, recipe, block, items, Some(width))
    }

    fn validate(
        dtype: Dtype,
        recipe: MergeSortKeysRecipe,
        block: [u32; 3],
        items: u32,
        width: Option<u32>,
    ) -> Result<Self> {
        let threads = block
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| (1..=1024).contains(n))
            .ok_or_else(|| {
                Error::message(
                    "Merge Sort Keys requires positive axes and at most 1024 block threads",
                )
            })?;
        let group_threads = if let Some(width) = width {
            if !matches!(width, 1 | 2 | 4 | 8 | 16 | 32) || threads % 32 != 0 {
                return Err(Error::message(
                    "Warp Merge Sort Keys requires a power-of-two width in 1..=32 and complete physical warps",
                ));
            }
            width
        } else {
            if !threads.is_power_of_two() {
                return Err(Error::message(
                    "Merge Sort Keys requires a power-of-two Block",
                ));
            }
            threads
        };
        if items == 0
            || group_threads
                .checked_mul(items)
                .and_then(|tile| tile.checked_add(1))
                .filter(|padded| *padded <= i32::MAX as u32)
                .is_none()
        {
            return Err(Error::message(
                "Merge Sort Keys requires positive items and a padded tile fitting CUB int capacity",
            ));
        }
        Ok(Self {
            dtype,
            recipe,
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
        let tile = self.width.unwrap_or(x * y * z) * items;
        let (header, primitive, rank) = if let Some(width) = self.width {
            (
                "warp/warp_merge_sort.cuh",
                format!("cub::WarpMergeSort<value_type, {items}, {width}, cub::NullType>"),
                format!("(threadIdx.x + {x} * (threadIdx.y + {y} * threadIdx.z)) % {width}"),
            )
        } else {
            (
                "block/block_merge_sort.cuh",
                format!("cub::BlockMergeSort<value_type, {x}, {items}, cub::NullType, {y}, {z}>"),
                format!("threadIdx.x + {x} * (threadIdx.y + {y} * threadIdx.z)"),
            )
        };
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let comparison = match self.recipe {
            MergeSortKeysRecipe::AscendingFull | MergeSortKeysRecipe::AscendingPartial => "<",
            MergeSortKeysRecipe::DescendingFull | MergeSortKeysRecipe::DescendingPartial => ">",
        };
        let (parameters, bounds, arguments, count) = if self.recipe.partial() {
            (
                format!(", long long valid_items, {dtype} sentinel"),
                format!("if (valid_items < 0 || valid_items > {tile}) __trap();"),
                ", static_cast<int>(valid_items), sentinel",
                "valid_items".into(),
            )
        } else {
            (String::new(), String::new(), "", tile.to_string())
        };
        format!(
            r#"
#include <cub/{header}>
#include <cub/util_type.cuh>
namespace {symbol}_detail {{
using value_type = {dtype};
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
struct compare {{
  __device__ __forceinline__ bool operator()(value_type lhs, value_type rhs) const {{
    return lhs {comparison} rhs;
  }}
}};
}}
extern "C" __device__ void {symbol}(const {dtype}* input, {dtype}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits{parameters}) {{
  using namespace {symbol}_detail;
  {bounds}
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_keys[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) local_keys[i] = input[i];
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.Sort(local_keys, compare{{}}{arguments});
  const int rank = static_cast<int>({rank});
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if (rank * {items} + i < {count}) {{
      output[i] = local_keys[i];
      output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
    }}
  }}
}}
"#
        )
    }
}

/// Built-in ordering and full/partial selection for associated key/value pairs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeSortPairsRecipe {
    /// Sort all pairs in ascending key order.
    AscendingFull,
    /// Sort all pairs in descending key order.
    DescendingFull,
    /// Sort a uniform prefix of pairs in ascending key order.
    AscendingPartial,
    /// Sort a uniform prefix of pairs in descending key order.
    DescendingPartial,
}

impl MergeSortPairsRecipe {
    fn keys(self) -> MergeSortKeysRecipe {
        match self {
            Self::AscendingFull => MergeSortKeysRecipe::AscendingFull,
            Self::DescendingFull => MergeSortKeysRecipe::DescendingFull,
            Self::AscendingPartial => MergeSortKeysRecipe::AscendingPartial,
            Self::DescendingPartial => MergeSortKeysRecipe::DescendingPartial,
        }
    }
    pub(crate) fn partial(self) -> bool {
        self.keys().partial()
    }
    pub(crate) fn token(self, value: Dtype) -> String {
        let form = match self {
            Self::AscendingFull => "ascending_full",
            Self::DescendingFull => "descending_full",
            Self::AscendingPartial => "ascending_partial",
            Self::DescendingPartial => "descending_partial",
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
        format!("merge_sort_pairs_{form}_{value}_validity_bits_v1")
    }
    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<(Self, Dtype)> {
        if !matches!(algorithm, "block" | "warp") {
            return Err(Error::message(
                "Merge Sort Pairs requires a Block or Warp recipe",
            ));
        }
        let unsupported = || Error::message("unsupported Merge Sort Pairs operation");
        let operation = operation
            .strip_prefix("merge_sort_pairs_")
            .and_then(|op| op.strip_suffix("_validity_bits_v1"))
            .ok_or_else(unsupported)?;
        let (recipe, value) = [
            ("ascending_full_", Self::AscendingFull),
            ("descending_full_", Self::DescendingFull),
            ("ascending_partial_", Self::AscendingPartial),
            ("descending_partial_", Self::DescendingPartial),
        ]
        .into_iter()
        .find_map(|(prefix, recipe)| operation.strip_prefix(prefix).map(|v| (recipe, v)))
        .ok_or_else(unsupported)?;
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

/// Validated Block/Warp merge sort with independently typed associated values.
///
/// Keys and values independently use any [`Dtype`], with the same positive
/// per-lane extent. Full native calls use `(const K* keys_in, K* keys_out,
/// u8* scratch, const V* values_in, V* values_out, const u8* keys_in_bits,
/// u8* keys_out_bits, const u8* values_in_bits, u8* values_out_bits)`.
/// Partial calls append `(i64 valid_items, K sentinel)`. The count and sentinel
/// are uniform within the group; `0 <= valid_items <= group_threads*items` is
/// checked before narrowing or numeric reads. The key-typed sentinel must
/// follow all valid keys in the chosen order. Values are carried unchanged,
/// not compared. Equal keys have no stable-tie guarantee; floating keys must
/// satisfy strict weak ordering, with no NaN total-order guarantee.
///
/// Each lane supplies fully initialized keys AND values, even for a zero
/// count, with separate packed low-bit-first `ceil(items/8)` validity maps.
/// Every input bit is checked before reading either array. Both inputs and
/// maps are preserved; sorting uses private copies. Outputs, output maps and
/// scratch are mutually disjoint and disjoint from both input arrays/maps.
/// Read-only inputs may alias. Pointers remain live and naturally aligned.
/// Full calls initialize both complete outputs; partial calls write only the
/// same blocked prefix in both outputs. All other bits, including padding,
/// are cleared; invalid output slots are not written.
///
/// Group geometry, rank, participation and completion follow
/// [`MergeSortKeysRequest`]. Scratch is the complete queried *pair* CUB
/// TempStorage including padding and value exchange, not the keys-only
/// layout. Each Warp group has an independent replica. No trailing barrier
/// is inserted; the caller completes accesses before scratch reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergeSortPairsRequest {
    pub(crate) recipe: MergeSortPairsRecipe,
    pub(crate) key_dtype: Dtype,
    pub(crate) value_dtype: Dtype,
    keys: MergeSortKeysRequest,
}

impl MergeSortPairsRequest {
    /// Validate independently typed pairs for a complete XYZ Block.
    ///
    /// `recipe` selects key ordering and full/partial output. `items` is the
    /// shared per-lane extent of both inputs and outputs.
    ///
    /// # Errors
    /// Rejects the same geometry, zero extent and padded signed-int capacity
    /// violations as [`MergeSortKeysRequest::block`].
    pub fn block(
        key_dtype: Dtype,
        value_dtype: Dtype,
        recipe: MergeSortPairsRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        let keys = MergeSortKeysRequest::block(key_dtype, recipe.keys(), block, items)?;
        Ok(Self {
            recipe,
            key_dtype,
            value_dtype,
            keys,
        })
    }
    /// Validate independently typed pairs for a physical/logical Warp group.
    ///
    /// Supports widths 1/2/4/8/16/32 within complete physical warps. `items`
    /// is the shared per-lane extent; scratch is queried for one group.
    ///
    /// # Errors
    /// Rejects the same geometry, width and padded capacity violations as
    /// [`MergeSortKeysRequest::warp`].
    pub fn warp(
        key_dtype: Dtype,
        value_dtype: Dtype,
        recipe: MergeSortPairsRecipe,
        block: [u32; 3],
        width: u32,
        items: u32,
    ) -> Result<Self> {
        let keys = MergeSortKeysRequest::warp(key_dtype, recipe.keys(), block, width, items)?;
        Ok(Self {
            recipe,
            key_dtype,
            value_dtype,
            keys,
        })
    }
    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.keys.block
    }
    pub(crate) fn source(self, symbol: &str) -> String {
        let (key, _) = self.key_dtype.native();
        let (value, _) = self.value_dtype.native();
        let [x, y, z] = self.keys.block;
        let items = self.keys.items;
        let tile = self.keys.width.unwrap_or(x * y * z) * items;
        let (header, primitive, rank) = if let Some(width) = self.keys.width {
            (
                "warp/warp_merge_sort.cuh",
                format!("cub::WarpMergeSort<key_type, {items}, {width}, value_type>"),
                format!("(threadIdx.x + {x} * (threadIdx.y + {y} * threadIdx.z)) % {width}"),
            )
        } else {
            (
                "block/block_merge_sort.cuh",
                format!("cub::BlockMergeSort<key_type, {x}, {items}, value_type, {y}, {z}>"),
                format!("threadIdx.x + {x} * (threadIdx.y + {y} * threadIdx.z)"),
            )
        };
        let bytes = items / 8 + u32::from(items % 8 != 0);
        let comparison = match self.recipe {
            MergeSortPairsRecipe::AscendingFull | MergeSortPairsRecipe::AscendingPartial => "<",
            MergeSortPairsRecipe::DescendingFull | MergeSortPairsRecipe::DescendingPartial => ">",
        };
        let (parameters, bounds, arguments, count) = if self.recipe.partial() {
            (
                format!(", long long valid_items, {key} sentinel"),
                format!("if (valid_items < 0 || valid_items > {tile}) __trap();"),
                ", static_cast<int>(valid_items), sentinel",
                "valid_items".into(),
            )
        } else {
            (String::new(), String::new(), "", tile.to_string())
        };
        format!(
            r#"
#include <cub/{header}>
namespace {symbol}_detail {{
using key_type = {key};
using value_type = {value};
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
struct compare {{
  __device__ __forceinline__ bool operator()(key_type lhs, key_type rhs) const {{
    return lhs {comparison} rhs;
  }}
}};
}}
extern "C" __device__ void {symbol}(const {key}* keys_in, {key}* keys_out,
    unsigned char* scratch, const {value}* values_in, {value}* values_out,
    const unsigned char* keys_in_bits, unsigned char* keys_out_bits,
    const unsigned char* values_in_bits, unsigned char* values_out_bits{parameters}) {{
  using namespace {symbol}_detail;
  {bounds}
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) {{ keys_out_bits[i] = 0; values_out_bits[i] = 0; }}
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    const unsigned int bit = 1u << (i % 8);
    if ((keys_in_bits[i / 8] & bit) == 0 || (values_in_bits[i / 8] & bit) == 0) __trap();
  }}
  key_type local_keys[{items}];
  value_type local_values[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{ local_keys[i] = keys_in[i]; local_values[i] = values_in[i]; }}
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.Sort(local_keys, local_values, compare{{}}{arguments});
  const int rank = static_cast<int>({rank});
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    if (rank * {items} + i < {count}) {{
      keys_out[i] = local_keys[i]; values_out[i] = local_values[i];
      keys_out_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
      values_out_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
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
    fn complete_block_and_padded_int_capacity() {
        for dtype in [
            Dtype::I8,
            Dtype::U8,
            Dtype::I16,
            Dtype::U16,
            Dtype::I32,
            Dtype::U32,
            Dtype::I64,
            Dtype::U64,
            Dtype::F32,
            Dtype::F64,
        ] {
            for (block, items) in [
                ([1, 1, 1], i32::MAX as u32 - 1),
                ([4, 4, 4], 4),
                ([2, 2, 2], 9),
            ] {
                assert!(
                    MergeSortKeysRequest::block(
                        dtype,
                        MergeSortKeysRecipe::AscendingFull,
                        block,
                        items
                    )
                    .is_ok()
                );
            }
        }
        for (block, items) in [
            ([0, 1, 1], 1),
            ([1, 0, 1], 1),
            ([1, 1, 0], 1),
            ([3, 1, 1], 1),
            ([3, 5, 2], 1),
            ([2048, 1, 1], 1),
            ([u32::MAX, 2, 2], 1),
            ([64, 1, 1], 0),
            ([1, 1, 1], i32::MAX as u32),
            ([1024, 1, 1], 1 << 21),
            ([64, 1, 1], u32::MAX),
        ] {
            assert!(
                MergeSortKeysRequest::block(
                    Dtype::I64,
                    MergeSortKeysRecipe::DescendingPartial,
                    block,
                    items
                )
                .is_err()
            );
        }
    }

    #[test]
    fn warp_groups_use_group_capacity_with_complete_physical_warps() {
        for width in [1, 2, 4, 8, 16, 32] {
            for block in [[64, 1, 1], [4, 4, 6]] {
                assert!(
                    MergeSortKeysRequest::warp(
                        Dtype::I64,
                        MergeSortKeysRecipe::DescendingPartial,
                        block,
                        width,
                        9
                    )
                    .is_ok()
                );
            }
        }
        // Capacity is per logical group, not the enclosing Block.
        assert!(
            MergeSortKeysRequest::warp(
                Dtype::U8,
                MergeSortKeysRecipe::AscendingFull,
                [1024, 1, 1],
                1,
                i32::MAX as u32 - 1
            )
            .is_ok()
        );
        for (block, width, items) in [
            ([0, 1, 1], 8, 1),
            ([u32::MAX, 2, 2], 8, 1),
            ([2048, 1, 1], 8, 1),
            ([33, 1, 1], 8, 1),
            ([4, 4, 1], 8, 1),
            ([64, 1, 1], 0, 1),
            ([64, 1, 1], 3, 1),
            ([64, 1, 1], 17, 1),
            ([64, 1, 1], 64, 1),
            ([64, 1, 1], 8, 0),
            ([64, 1, 1], 1, i32::MAX as u32),
            ([64, 1, 1], 32, (i32::MAX as u32 / 32) + 1),
        ] {
            assert!(
                MergeSortKeysRequest::warp(
                    Dtype::I64,
                    MergeSortKeysRecipe::AscendingFull,
                    block,
                    width,
                    items
                )
                .is_err()
            );
        }
    }
}
