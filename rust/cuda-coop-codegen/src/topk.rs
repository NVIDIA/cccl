// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Unsorted Block TopK with preserved inputs and fresh prefix-valid outputs.

use crate::{Dtype, Error, Result};

/// Whether to retain the smallest or largest keys, without sorting them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKSelection {
    /// Retain the smallest keys.
    Min,
    /// Retain the largest keys.
    Max,
}

/// A uniform runtime count or a provider-checked static count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopKCount {
    /// The signed64 native control supplies the count.
    Runtime,
    /// A static count, also passed through the signed64 native control.
    Static(u32),
}
impl TopKCount {
    fn field(self) -> String {
        match self {
            Self::Runtime => "r0000000000".into(),
            Self::Static(n) => format!("s{n:010}"),
        }
    }
    fn parse(field: &str) -> Result<Self> {
        if field.len() != 11 || !field.as_bytes()[1..].iter().all(u8::is_ascii_digit) {
            return Err(Error::message(
                "TopK count requires one mode and ten ASCII digits",
            ));
        }
        let n = field[1..]
            .parse::<u32>()
            .map_err(|_| Error::message("TopK count overflow"))?;
        match (&field[..1], n) {
            ("r", 0) => Ok(Self::Runtime),
            ("s", n) if n <= i32::MAX as u32 => Ok(Self::Static(n)),
            _ => Err(Error::message("invalid TopK count mode or int capacity")),
        }
    }
    fn check(self, tile: u32) -> Result<()> {
        if matches!(self, Self::Static(n) if n > tile) {
            return Err(Error::message("TopK static count exceeds the Block tile"));
        }
        Ok(())
    }
}

/// Unsorted selection and optional valid prefix. Omitted validity means full input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKRecipe {
    /// Smallest or largest keys.
    pub selection: TopKSelection,
    /// Required number of selected keys, bounded by the complete tile.
    pub k: TopKCount,
    /// Optional valid prefix; explicit counts use CUB's partial-tile path.
    pub valid_items: Option<TopKCount>,
}
impl TopKRecipe {
    pub(crate) fn token(self, value: Option<Dtype>) -> String {
        let direction = match self.selection {
            TopKSelection::Min => "min",
            TopKSelection::Max => "max",
        };
        let kind = value.map_or_else(|| "keys".into(), |v| format!("pairs_{}", tag(v)));
        let valid = self
            .valid_items
            .map_or_else(|| "f0000000000".into(), TopKCount::field);
        format!(
            "topk_{direction}_{kind}_k{}_v{valid}_validity_bits_v1",
            self.k.field()
        )
    }
    pub(crate) fn parse(token: &str) -> Result<(Self, Option<Dtype>)> {
        let token = token
            .strip_prefix("topk_")
            .and_then(|s| s.strip_suffix("_validity_bits_v1"))
            .ok_or_else(|| Error::message("invalid TopK operation"))?;
        let (direction, rest) = token
            .split_once('_')
            .ok_or_else(|| Error::message("invalid TopK direction"))?;
        let selection = match direction {
            "min" => TopKSelection::Min,
            "max" => TopKSelection::Max,
            _ => return Err(Error::message("invalid TopK direction")),
        };
        let (kind, counts) = rest
            .split_once("_k")
            .ok_or_else(|| Error::message("missing TopK counts"))?;
        let value = if kind == "keys" {
            None
        } else {
            Some(match kind.strip_prefix("pairs_") {
                Some("i8_") => Dtype::I8,
                Some("u8_") => Dtype::U8,
                Some("i16") => Dtype::I16,
                Some("u16") => Dtype::U16,
                Some("i32") => Dtype::I32,
                Some("u32") => Dtype::U32,
                Some("i64") => Dtype::I64,
                Some("u64") => Dtype::U64,
                Some("f32") => Dtype::F32,
                Some("f64") => Dtype::F64,
                _ => return Err(Error::message("invalid TopK pair value type")),
            })
        };
        let (k, valid) = counts
            .split_once("_v")
            .ok_or_else(|| Error::message("missing TopK valid count"))?;
        let valid_items = if valid == "f0000000000" {
            None
        } else {
            Some(TopKCount::parse(valid)?)
        };
        Ok((
            Self {
                selection,
                k: TopKCount::parse(k)?,
                valid_items,
            },
            value,
        ))
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

/// A validated complete one-dimensional Block TopK specialization.
///
/// All ten numeric key types are supported. Inputs and maps are lane-private,
/// readable for every item even when k or validity is zero, and preserved.
/// Output and map are disjoint fresh buffers of the same type and extent;
/// only the blocked prefix min(k, valid_items) is written and marked valid.
/// Unwritten numeric slots remain unreadable. Counts are uniform signed64,
/// checked against tile capacity before narrowing; static counts must match
/// their controls. Selection is unsorted and equal-key order is unspecified.
/// Floating keys have no specified NaN ordering. Scratch is the complete CUB
/// TempStorage, correctly aligned, disjoint and synchronized before reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKKeysRequest {
    pub(crate) dtype: Dtype,
    pub(crate) recipe: TopKRecipe,
    block: [u32; 3],
    items: u32,
}
impl TopKKeysRequest {
    /// Validate 1D geometry, positive extent and both static count bounds.
    ///
    /// # Errors
    /// Rejects non-1D Blocks, sizes outside 1..=1024, zero extent, int overflow
    /// or a static count beyond the complete tile.
    pub fn block(dtype: Dtype, recipe: TopKRecipe, block: [u32; 3], items: u32) -> Result<Self> {
        if !(1..=1024).contains(&block[0]) || block[1..] != [1, 1] {
            return Err(Error::message(
                "TopK requires a complete one-dimensional Block of 1..=1024 threads",
            ));
        }
        let tile = block[0]
            .checked_mul(items)
            .filter(|n| items > 0 && *n <= i32::MAX as u32)
            .ok_or_else(|| {
                Error::message("TopK requires positive items and a tile fitting CUB int capacity")
            })?;
        recipe.k.check(tile)?;
        if let Some(valid) = recipe.valid_items {
            valid.check(tile)?;
        }
        Ok(Self {
            dtype,
            recipe,
            block,
            items,
        })
    }
    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.block
    }
    pub(crate) fn source(self, symbol: &str) -> String {
        source(self, None, symbol)
    }
}

/// Associated-value TopK with independently typed preserved input and fresh output.
///
/// Both inputs have the same extent. Every key AND value bit is checked before
/// either numeric input is read, including empty selections. Values are opaque
/// numeric bit patterns and move with their keys; selection at equal-key ties
/// is unspecified. Each output retains its own type/alignment and the same
/// selected prefix. All four arrays, maps and shared scratch obey the disjoint
/// output and lifetime requirements of [`TopKKeysRequest`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopKPairsRequest {
    pub(crate) keys: TopKKeysRequest,
    pub(crate) value_dtype: Dtype,
}
impl TopKPairsRequest {
    /// Validate paired selection with an independent numeric value type.
    ///
    /// # Errors
    /// Has the geometry/count restrictions of [`TopKKeysRequest::block`].
    pub fn block(
        key: Dtype,
        value: Dtype,
        recipe: TopKRecipe,
        block: [u32; 3],
        items: u32,
    ) -> Result<Self> {
        Ok(Self {
            keys: TopKKeysRequest::block(key, recipe, block, items)?,
            value_dtype: value,
        })
    }
    pub(crate) fn block_shape(self) -> [u32; 3] {
        self.keys.block
    }
    pub(crate) fn source(self, symbol: &str) -> String {
        source(self.keys, Some(self.value_dtype), symbol)
    }
}

fn source(request: TopKKeysRequest, value: Option<Dtype>, symbol: &str) -> String {
    let (key, _) = request.dtype.native();
    let n = request.items;
    let b = request.block[0];
    let tile = b * n;
    let bytes = n / 8 + u32::from(n % 8 != 0);
    let direction = match request.recipe.selection {
        TopKSelection::Min => "min",
        TopKSelection::Max => "max",
    };
    let full = if request.recipe.valid_items.is_none() {
        "true"
    } else {
        "false"
    };
    let valid_parameter = if full == "true" {
        ""
    } else {
        ", long long valid_items"
    };
    let valid = if full == "true" {
        tile.to_string()
    } else {
        "valid_items".into()
    };
    let mut bounds =
        format!("if (k < 0 || k > {tile} || {valid} < 0 || {valid} > {tile}) __trap();");
    if let TopKCount::Static(k) = request.recipe.k {
        bounds.push_str(&format!(" if (k != {k}) __trap();"));
    }
    if let Some(TopKCount::Static(v)) = request.recipe.valid_items {
        bounds.push_str(&format!(" if (valid_items != {v}) __trap();"));
    }
    let (
        primitive,
        value_parameters,
        value_maps,
        clear_values,
        check_values,
        local_values,
        copy_values,
        kind,
        args,
        write_values,
    ) = if let Some(dtype) = value {
        let (v, _) = dtype.native();
        (
            format!("cub::detail::block_topk<{key}, {b}, {n}, {v}>"),
            format!(", const {v}* values, {v}* output_values"),
            ", const unsigned char* value_bits, unsigned char* output_value_bits",
            "output_value_bits[i] = 0;",
            "if ((value_bits[i / 8] & (1u << (i % 8))) == 0) __trap();",
            format!("{v} local_values[{n}];"),
            "local_values[i] = values[i];",
            "pairs",
            "local_keys, local_values",
            "output_values[i] = local_values[i]; output_value_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));",
        )
    } else {
        (
            format!("cub::detail::block_topk<{key}, {b}, {n}>"),
            String::new(),
            "",
            "",
            "",
            String::new(),
            "",
            "keys",
            "local_keys",
            "",
        )
    };
    format!(
        r#"
#include <cub/block/block_topk.cuh>
namespace {symbol}_detail {{
using primitive = {primitive};
using scratch_storage = primitive::TempStorage;
}}
extern "C" __device__ void {symbol}(const {key}* input, {key}* output,
    unsigned char* scratch{value_parameters}, const unsigned char* input_bits,
    unsigned char* output_bits{value_maps}, long long k{valid_parameter}) {{
  using namespace {symbol}_detail;
  {bounds}
#pragma unroll
  for (int i = 0; i < {bytes}; ++i) {{ output_bits[i] = 0; {clear_values} }}
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{
    if ((input_bits[i / 8] & (1u << (i % 8))) == 0) __trap();
    {check_values}
  }}
  // Uniform empty selections still validate every input slot, but partial CUB
  // requires num_valid > 0. No numeric input read is needed for an empty result.
  if (k == 0 || {valid} == 0) return;
  {key} local_keys[{n}];
  {local_values}
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{ local_keys[i] = input[i]; {copy_values} }}
  primitive operation{{*reinterpret_cast<scratch_storage*>(scratch)}};
  operation.template {direction}_{kind}<{full}>({args}, static_cast<int>(k), static_cast<int>({valid}));
  const long long selected = k < {valid} ? k : {valid};
#pragma unroll
  for (int i = 0; i < {n}; ++i) {{
    if (static_cast<long long>(threadIdx.x) * {n} + i < selected) {{
      output[i] = local_keys[i];
      output_bits[i / 8] |= static_cast<unsigned char>(1u << (i % 8));
      {write_values}
    }}
  }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_geometry_and_closed_transport() {
        let recipe = TopKRecipe {
            selection: TopKSelection::Min,
            k: TopKCount::Static(17),
            valid_items: Some(TopKCount::Static(31)),
        };
        for value in [None, Some(Dtype::I8), Some(Dtype::F64)] {
            assert_eq!(
                TopKRecipe::parse(&recipe.token(value)).unwrap(),
                (recipe, value)
            );
        }
        assert!(TopKKeysRequest::block(Dtype::I32, recipe, [33, 1, 1], 9).is_ok());
        assert!(TopKKeysRequest::block(Dtype::I32, recipe, [4, 4, 4], 4).is_err());
        assert!(TopKKeysRequest::block(Dtype::I32, recipe, [1, 1, 1], 1).is_err());
        for k in [TopKCount::Static(0), TopKCount::Runtime] {
            let empty = TopKRecipe {
                k,
                valid_items: Some(TopKCount::Static(0)),
                ..recipe
            };
            assert!(TopKKeysRequest::block(Dtype::F64, empty, [1, 1, 1], 1).is_ok());
        }
        assert!(TopKKeysRequest::block(Dtype::I32, recipe, [1024, 1, 1], u32::MAX).is_err());
        for bad in ["r0000000001", "s2147483648", "s00000000a0", "s000000001"] {
            assert!(TopKCount::parse(bad).is_err());
        }
    }
}
