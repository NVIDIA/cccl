// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Validated Block histograms with independent counter types and output extents.

use crate::{Dtype, Error, Result};

/// The CUB implementation used to construct a fresh Block histogram.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistogramAlgorithm {
    /// Accumulate samples using atomics, the common API default.
    Atomic,
    /// Sort a private copy of the samples before counting runs.
    Sort,
}

impl HistogramAlgorithm {
    pub(crate) fn token(self) -> &'static str {
        match self {
            Self::Atomic => "atomic",
            Self::Sort => "sort",
        }
    }

    fn native(self) -> &'static str {
        match self {
            Self::Atomic => "cub::BLOCK_HISTO_ATOMIC",
            Self::Sort => "cub::BLOCK_HISTO_SORT",
        }
    }
}

pub(crate) const TEMPLATE_OPERATION: &str =
    "histogram_xxx_bins0000000000_k0000000000_validity_bits_v1";

/// Static histogram choices independent of the input type and extent.
///
/// `counter` is one of i32/u32/i64/u64; `bins` is the block-wide bin count;
/// `output_items` is each lane's output capacity. A request additionally checks
/// that the complete Block has capacity for every bin. Every fresh count fits
/// all four counter types because the input tile is bounded by i32::MAX.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramRecipe {
    algorithm: HistogramAlgorithm,
    pub(crate) counter: Dtype,
    bins: u32,
    output_items: u32,
}

impl HistogramRecipe {
    /// Select the algorithm, counter type, total bins and per-lane output extent.
    ///
    /// # Errors
    /// Rejects counters other than i32/u32/i64/u64 and zero or greater-than-i32::MAX
    /// bin counts or output extents. Block capacity is checked by
    /// [`HistogramRequest::block`].
    pub fn new(
        algorithm: HistogramAlgorithm,
        counter: Dtype,
        bins: u32,
        output_items: u32,
    ) -> Result<Self> {
        if !matches!(counter, Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64) {
            return Err(Error::message("Histogram counters must be i32/u32/i64/u64"));
        }
        if bins == 0
            || bins > i32::MAX as u32
            || output_items == 0
            || output_items > i32::MAX as u32
        {
            return Err(Error::message(
                "Histogram bins and output extent must be positive and no larger than i32::MAX",
            ));
        }
        Ok(Self {
            algorithm,
            counter,
            bins,
            output_items,
        })
    }

    pub(crate) fn token(self) -> String {
        let counter = match self.counter {
            Dtype::I32 => "i32",
            Dtype::U32 => "u32",
            Dtype::I64 => "i64",
            Dtype::U64 => "u64",
            _ => unreachable!("validated histogram counter"),
        };
        format!(
            "histogram_{counter}_bins{:010}_k{:010}_validity_bits_v1",
            self.bins, self.output_items
        )
    }

    pub(crate) fn parse(operation: &str, algorithm: &str) -> Result<Self> {
        let algorithm = match algorithm {
            "atomic" => HistogramAlgorithm::Atomic,
            "sort" => HistogramAlgorithm::Sort,
            _ => return Err(Error::message("Histogram algorithm must be atomic or sort")),
        };
        let invalid = || Error::message("unsupported Histogram operation");
        let body = operation
            .strip_prefix("histogram_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
            .ok_or_else(invalid)?;
        let (counter, extents) = body.split_once("_bins").ok_or_else(invalid)?;
        let counter = match counter {
            "i32" => Dtype::I32,
            "u32" => Dtype::U32,
            "i64" => Dtype::I64,
            "u64" => Dtype::U64,
            _ => return Err(invalid()),
        };
        let (bins, output_items) = extents.split_once("_k").ok_or_else(invalid)?;
        let decimal = |text: &str| {
            if text.len() != 10 || !text.bytes().all(|b| b.is_ascii_digit()) {
                return Err(invalid());
            }
            text.parse::<u32>().map_err(|_| invalid())
        };
        Self::new(algorithm, counter, decimal(bins)?, decimal(output_items)?)
    }
}

/// A complete one-dimensional Block histogram with preserved input.
///
/// Input samples are u8/i32/u32/i64/u64, with `items` slots per lane. The native
/// ABI is `(const T* input, C* output, u8* scratch, const u8* input_bits,
/// u8* output_bits)`, with no scalar controls. Input and output bitmaps contain
/// respectively ceil(items/8) and ceil(output_items/8) low-bit-first bytes.
/// Every input bit is checked before any numeric read. Samples outside
/// `[0, bins)` trap before CUB; signed negatives and wide integers are checked
/// without narrowing. All buffers must be disjoint, live and naturally aligned.
///
/// CUB operates on a private sample copy. Each rank t receives bins t+i*BlockX;
/// padding slots are zero. Every output slot is initialized and marked valid,
/// with unused high bitmap bits cleared. Input values and validity are preserved.
/// Scratch includes both CUB TempStorage and unsigned `counters[bins]`, using the
/// public counter width. The returned layout includes their padding/alignment.
///
/// Every Block member participates in converged control flow. The provider
/// synchronizes after CUB counting and before reading the counters. Callers
/// must still complete the Block collective before scratch reuse: output
/// distribution has no trailing completion barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistogramRequest {
    pub(crate) recipe: HistogramRecipe,
    pub(crate) dtype: Dtype,
    block: [u32; 3],
    items: u32,
}

impl HistogramRequest {
    /// Validate input samples, complete 1D Block geometry and output capacity.
    ///
    /// `dtype` selects u8/i32/u32/i64/u64. `items` is the positive input extent;
    /// the independent counter type and output extent come from `recipe`.
    ///
    /// # Errors
    /// Rejects unsupported samples, non-1D or nonpositive Blocks, more than 1024
    /// threads, a zero input extent, input/output tiles exceeding i32::MAX, or
    /// insufficient total output capacity for the selected bins.
    pub fn block(
        dtype: Dtype,
        recipe: HistogramRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        if !matches!(
            dtype,
            Dtype::U8 | Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64
        ) {
            return Err(Error::message(
                "Histogram samples must be u8/i32/u32/i64/u64",
            ));
        }
        let [threads, y, z] = block;
        if !(1..=1024).contains(&threads) || y != 1 || z != 1 {
            return Err(Error::message(
                "Histogram requires a 1D Block with 1..=1024 threads",
            ));
        }
        let bounded_tile = |extent: u32| {
            threads
                .checked_mul(extent)
                .filter(|n| (1..=i32::MAX as u32).contains(n))
        };
        if bounded_tile(items).is_none() {
            return Err(Error::message(
                "Histogram input tile must be positive and no larger than i32::MAX",
            ));
        }
        let capacity = bounded_tile(recipe.output_items).ok_or_else(|| {
            Error::message("Histogram output tile must be positive and no larger than i32::MAX")
        })?;
        if capacity < recipe.bins {
            return Err(Error::message(
                "Histogram output capacity is smaller than its bin count",
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
        let (sample, _) = self.dtype.native();
        let (counter, _) = self.recipe.counter.native();
        let unsigned_counter = if matches!(self.recipe.counter, Dtype::I64 | Dtype::U64) {
            "unsigned long long"
        } else {
            "unsigned int"
        };
        let negative = if matches!(self.dtype, Dtype::I32 | Dtype::I64) {
            "if (value < 0) __trap();"
        } else {
            ""
        };
        let threads = self.block[0];
        let items = self.items;
        let HistogramRecipe {
            algorithm,
            bins,
            output_items,
            ..
        } = self.recipe;
        let algorithm = algorithm.native();
        let output_bytes = output_items / 8 + u32::from(output_items % 8 != 0);
        format!(
            r#"
#include <cub/block/block_histogram.cuh>
namespace {symbol}_detail {{
using value_type = {sample};
using counter_type = {counter};
using internal_counter_type = {unsigned_counter};
using primitive = cub::BlockHistogram<value_type, {threads}, {items}, {bins}, {algorithm}>;
struct scratch_storage {{
  primitive::TempStorage cub_storage;
  internal_counter_type counters[{bins}];
}};
}}
extern "C" __device__ void {symbol}(const {sample}* input, {counter}* output,
    unsigned char* scratch, const unsigned char* input_bits, unsigned char* output_bits) {{
  using namespace {symbol}_detail;
#pragma unroll
  for (int i = 0; i < {output_bytes}; ++i) output_bits[i] = 0;
#pragma unroll
  for (int i = 0; i < {items}; ++i)
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
  value_type local_input[{items}];
#pragma unroll
  for (int i = 0; i < {items}; ++i) {{
    const value_type value = input[i];
    {negative}
    if (static_cast<unsigned long long>(value) >= {bins}ull) __trap();
    local_input[i] = value;
  }}
  auto& storage = *reinterpret_cast<scratch_storage*>(scratch);
  primitive{{storage.cub_storage}}.Histogram(local_input, storage.counters);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < {output_items}; ++i) {{
    const int bin = static_cast<int>(threadIdx.x) + i * {threads};
    output[i] = bin < {bins} ? static_cast<counter_type>(storage.counters[bin]) : counter_type{{0}};
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
    fn histogram_geometry_types_and_independent_capacity() {
        let recipe = HistogramRecipe::new(HistogramAlgorithm::Sort, Dtype::U64, 65, 2).unwrap();
        assert!(HistogramRequest::block(Dtype::U8, recipe, [64, 1, 1], 4).is_ok());
        assert!(
            HistogramRequest::block(Dtype::U8, recipe, [32, 1, 1], 4)
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
        for shape in [[0, 1, 1], [1025, 1, 1], [32, 2, 1], [32, 1, 2]] {
            assert!(HistogramRequest::block(Dtype::U8, recipe, shape, 4).is_err());
        }
        for dtype in [Dtype::I8, Dtype::I16, Dtype::U16, Dtype::F32, Dtype::F64] {
            assert!(HistogramRequest::block(dtype, recipe, [64, 1, 1], 4).is_err());
            assert!(HistogramRecipe::new(HistogramAlgorithm::Atomic, dtype, 1, 1).is_err());
        }
        for (bins, output) in [(0, 1), (1, 0), (u32::MAX, 1), (1, u32::MAX)] {
            assert!(
                HistogramRecipe::new(HistogramAlgorithm::Atomic, Dtype::I32, bins, output).is_err()
            );
        }
        let single = HistogramRecipe::new(HistogramAlgorithm::Atomic, Dtype::I32, 1, 1).unwrap();
        assert!(HistogramRequest::block(Dtype::I64, single, [1, 1, 1], i32::MAX as u32).is_ok());
        for items in [0, i32::MAX as u32, u32::MAX] {
            assert!(HistogramRequest::block(Dtype::I64, single, [2, 1, 1], items).is_err());
        }
        let large_output =
            HistogramRecipe::new(HistogramAlgorithm::Atomic, Dtype::I64, 1, i32::MAX as u32)
                .unwrap();
        assert!(HistogramRequest::block(Dtype::U64, large_output, [1, 1, 1], 1).is_ok());
        assert!(HistogramRequest::block(Dtype::U64, large_output, [2, 1, 1], 1).is_err());
    }
}
