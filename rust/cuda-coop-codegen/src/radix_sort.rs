// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block radix sorting with preserved keys and fresh blocked output.

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

#[cfg(test)]
mod tests {
    use super::*;

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
