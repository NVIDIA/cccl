// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use crate::{Dtype, Error, Result};

/// A fresh decoded window or a complete write into a checked destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunLengthDecodeRecipe {
    /// Return a zero-filled window with an independent per-thread extent.
    Window,
    /// Prepare once and write the complete stream, returning its u32 total.
    Into,
}
impl RunLengthDecodeRecipe {
    pub(crate) fn token(self, length: Dtype, decoded: u32, offset: Dtype) -> String {
        let form = match self {
            Self::Window => "window",
            Self::Into => "into",
        };
        format!(
            "run_length_decode_{form}_l{}_d{decoded:010}_o{}_validity_bits_v1",
            tag(length),
            tag(offset)
        )
    }
    pub(crate) fn parse(token: &str) -> Result<(Self, Dtype, u32, Dtype)> {
        let rest = token
            .strip_prefix("run_length_decode_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
            .ok_or_else(|| Error::message("invalid Run Length Decode operation"))?;
        let (form, rest) = rest
            .split_once("_l")
            .ok_or_else(|| Error::message("missing Run Length Decode length type"))?;
        let recipe = match form {
            "window" => Self::Window,
            "into" => Self::Into,
            _ => return Err(Error::message("invalid Run Length Decode form")),
        };
        let (length, rest) = rest
            .split_once("_d")
            .ok_or_else(|| Error::message("missing Run Length Decode output extent"))?;
        let (decoded, offset) = rest
            .split_once("_o")
            .ok_or_else(|| Error::message("missing Run Length Decode offset type"))?;
        if decoded.len() != 10 || !decoded.bytes().all(|c| c.is_ascii_digit()) {
            return Err(Error::message(
                "Run Length Decode extent requires ten ASCII digits",
            ));
        }
        let decoded = decoded
            .parse::<u32>()
            .map_err(|_| Error::message("Run Length Decode extent overflow"))?;
        let offset = integer(offset)?;
        if !matches!(offset, Dtype::I64 | Dtype::U64) {
            return Err(Error::message(
                "Run Length Decode offset must be widened to i64 or u64",
            ));
        }
        Ok((recipe, integer(length)?, decoded, offset))
    }
}

fn tag(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::I8 => "i8_",
        Dtype::U8 => "u8_",
        Dtype::I16 => "i16",
        Dtype::U16 => "u16",
        Dtype::I32 => "i32",
        Dtype::U32 => "u32",
        Dtype::I64 => "i64",
        Dtype::U64 => "u64",
        Dtype::F32 => "f32",
        Dtype::F64 => "f64",
    }
}
fn integer(tag: &str) -> Result<Dtype> {
    match tag {
        "i8_" => Ok(Dtype::I8),
        "u8_" => Ok(Dtype::U8),
        "i16" => Ok(Dtype::I16),
        "u16" => Ok(Dtype::U16),
        "i32" => Ok(Dtype::I32),
        "u32" => Ok(Dtype::U32),
        "i64" => Ok(Dtype::I64),
        "u64" => Ok(Dtype::U64),
        _ => Err(Error::message(
            "Run Length Decode lengths require an integer type",
        )),
    }
}

/// Checked CUB Block Run Length Decode with independent run/window extents.
///
/// Values support all ten numeric types; lengths support eight integer types.
/// Every input item and validity bit must be readable, even for an empty stream.
/// Inputs are preserved. Positive lengths precede all zero padding and the sum
/// fits u32. Invalid lengths or controls trap before decoding or global writes.
/// Window output is fresh and fully valid, with explicit zero tails. Into writes
/// only its checked destination interval and a lane-local u32 total; the caller
/// supplies the genuine global allocation/ownership witness and capacity bounds.
/// Local outputs/maps and scratch are disjoint from both inputs and each other.
/// Scratch is the full union of validation scan and decoder, synchronized between
/// lifetimes and kept live through all Into windows. Pointers cannot escape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunLengthDecodeRequest {
    pub(crate) dtype: Dtype,
    pub(crate) length: Dtype,
    pub(crate) recipe: RunLengthDecodeRecipe,
    pub(crate) decoded: u32,
    pub(crate) offset: Dtype,
    block: [u32; 3],
    items: u32,
}
impl RunLengthDecodeRequest {
    /// Validate the types, complete 1D Block, and positive independent extents.
    ///
    /// # Errors
    /// Rejects noninteger lengths, controls other than i64/u64, a Block outside
    /// 1..=1024, multidimensional geometry, or either tile exceeding i32::MAX.
    pub fn block(
        dtype: Dtype,
        length: Dtype,
        recipe: RunLengthDecodeRecipe,
        decoded: u32,
        offset: Dtype,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        integer(tag(length))?;
        if !matches!(offset, Dtype::I64 | Dtype::U64) {
            return Err(Error::message(
                "Run Length Decode offset must be i64 or u64",
            ));
        }
        if !(1..=1024).contains(&block[0]) || block[1..] != [1, 1] {
            return Err(Error::message(
                "Run Length Decode requires a complete one-dimensional Block of 1..=1024 threads",
            ));
        }
        for extent in [items, decoded] {
            block[0].checked_mul(extent).filter(|n| extent > 0 && *n <= i32::MAX as u32)
                .ok_or_else(|| Error::message("Run Length Decode requires positive run and window extents with tiles fitting i32"))?;
        }
        Ok(Self {
            dtype,
            length,
            recipe,
            decoded,
            offset,
            block,
            items,
        })
    }
    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.block
    }
    pub(crate) fn source(self, symbol: &str) -> String {
        let (value, _) = self.dtype.native();
        let (length, _) = self.length.native();
        let (control, _) = self.offset.native();
        let n = self.items;
        let d = self.decoded;
        let b = self.block[0];
        let bytes = d / 8 + u32::from(d % 8 != 0);
        let (parameters, body) = match self.recipe {
            RunLengthDecodeRecipe::Window => (
                format!(
                    "const {value}* values, {value}* output, unsigned char* scratch, const {length}* lengths, const unsigned char* value_bits, const unsigned char* length_bits, unsigned char* output_bits, {control} offset"
                ),
                format!(
                    r#"
  {value} decoded[{d}];
#pragma unroll
  for (int i = 0; i < {d}; ++i) decoded[i] = static_cast<{value}>(0);
  if (start < total) {{
    primitive decoder(storage.decode, local_values, offsets);
    decoder.RunLengthDecode(decoded, start);
#pragma unroll
    for (int i = 0; i < {d}; ++i) {{
      if (wide_t{{threadIdx.x}} * {d} + i >= total - start) decoded[i] = static_cast<{value}>(0);
    }}
  }}
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {d}; ++i) {{
    output[i] = decoded[i];
    output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
  }}
"#
                ),
            ),
            RunLengthDecodeRecipe::Into => (
                format!(
                    "{value}* destination, const {value}* values, unsigned char* scratch, const {length}* lengths, const unsigned char* value_bits, const unsigned char* length_bits, unsigned int* result, {control} offset, unsigned long long capacity, unsigned long long period, unsigned long long region_begin, unsigned long long region_capacity"
                ),
                format!(
                    r#"
  if (start > capacity || total > capacity - start) __trap();
  if (total != 0) {{
    if (period == 0 || region_begin > period || region_capacity > period - region_begin) __trap();
    const wide_t within = start % period;
    if (within < region_begin || within - region_begin > region_capacity
        || total > region_capacity - (within - region_begin)) __trap();
  }}
  *result = static_cast<unsigned int>(total);
  if (total == 0) return;
  primitive decoder(storage.decode, local_values, offsets);
  for (wide_t base = 0; base < total; base += window_size) {{
    {value} decoded[{d}];
    decoder.RunLengthDecode(decoded, base);
#pragma unroll
    for (int i = 0; i < {d}; ++i) {{
      const wide_t index = wide_t{{threadIdx.x}} * {d} + i;
      if (index < total - base) destination[start + base + index] = decoded[i];
    }}
  }}
"#
                ),
            ),
        };
        format!(
            r#"
#include <cub/block/block_run_length_decode.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
namespace {symbol}_detail {{
using wide_t = unsigned long long;
inline constexpr wide_t window_size = wide_t{{{b}}} * {d};
inline constexpr wide_t max_total = ::cuda::std::numeric_limits<unsigned int>::max();
inline constexpr wide_t overflow = max_total + 1;
struct run_state {{ wide_t count; bool has_zero; bool invalid; }};
struct append_runs {{
  __device__ run_state operator()(const run_state& a, const run_state& b) const {{
    const wide_t count = b.count > overflow - a.count ? overflow : a.count + b.count;
    return {{count, a.has_zero || b.has_zero,
             a.invalid || b.invalid || (a.has_zero && b.count != 0)}};
  }}
}};
using scan_t = cub::BlockScan<run_state, {b}>;
using primitive = cub::BlockRunLengthDecode<{value}, {b}, {n}, {d}, wide_t>;
union scratch_storage {{ scan_t::TempStorage scan; primitive::TempStorage decode; }};
template <typename T> __device__ bool negative(T value) {{
  if constexpr (::cuda::std::is_signed_v<T>) return value < 0;
  return false;
}}
__device__ wide_t prepare(scratch_storage& storage, const {length} (&lengths)[{n}], wide_t (&offsets)[{n}]) {{
  run_state input[{n}];
  run_state prefix[{n}];
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{
    const wide_t count = static_cast<wide_t>(lengths[i]);
    input[i] = {{count > max_total ? overflow : count, count == 0,
                negative(lengths[i]) || count > max_total}};
  }}
  run_state total;
  scan_t(storage.scan).ExclusiveScan(input, prefix, run_state{{0, false, false}}, append_runs{{}}, total);
  if (total.invalid || total.count > max_total) __trap();
#pragma unroll
  for (int i = 0; i < {n}; ++i) offsets[i] = prefix[i].count;
  __syncthreads();
  return total.count;
}}
}}
extern "C" __device__ void {symbol}({parameters}) {{
  using namespace {symbol}_detail;
  if (negative(offset)) __trap();
  const wide_t start = static_cast<wide_t>(offset);
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{
    if ((value_bits[i / 8] & (1u << (i % 8))) == 0
        || (length_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  }}
  {value} local_values[{n}];
  {length} local_lengths[{n}];
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{ local_values[i] = values[i]; local_lengths[i] = lengths[i]; }}
  auto& storage = *reinterpret_cast<scratch_storage*>(scratch);
  wide_t offsets[{n}];
  const wide_t total = prepare(storage, local_lengths, offsets);
  {body}
}}
"#
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_extents_integer_lengths_and_closed_transport() {
        for recipe in [RunLengthDecodeRecipe::Window, RunLengthDecodeRecipe::Into] {
            let token = recipe.token(Dtype::U64, 3, Dtype::I64);
            assert_eq!(
                RunLengthDecodeRecipe::parse(&token).unwrap(),
                (recipe, Dtype::U64, 3, Dtype::I64)
            );
            assert!(
                RunLengthDecodeRequest::block(
                    Dtype::F64,
                    Dtype::I8,
                    recipe,
                    3,
                    Dtype::U64,
                    [33, 1, 1],
                    9
                )
                .is_ok()
            );
            for (length, d, offset, block, n) in [
                (Dtype::F32, 3, Dtype::U64, [32, 1, 1], 1),
                (Dtype::U32, 0, Dtype::U64, [32, 1, 1], 1),
                (Dtype::U32, 3, Dtype::I32, [32, 1, 1], 1),
                (Dtype::U32, 3, Dtype::U64, [4, 4, 2], 1),
                (Dtype::U32, 3, Dtype::U64, [32, 1, 1], 0),
                (Dtype::U32, u32::MAX, Dtype::U64, [32, 1, 1], 1),
            ] {
                assert!(
                    RunLengthDecodeRequest::block(Dtype::I64, length, recipe, d, offset, block, n)
                        .is_err()
                );
            }
            assert!(
                RunLengthDecodeRecipe::parse(&token.replace("0000000003", "00000000x3")).is_err()
            );
            assert!(RunLengthDecodeRecipe::parse(&token.replace("_lu64", "_lf64")).is_err());
        }
    }
}
