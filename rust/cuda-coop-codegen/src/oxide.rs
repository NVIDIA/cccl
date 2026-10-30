// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Host transport for the Oxide library-operation marker.
//!
//! A dependency build script can serve as its already-built provider executable:
//! normal Cargo execution embeds [`registration`] in a device crate constant;
//! `--oxide-library-request-v1` execution passes stdin/stdout to [`serve`];
//! `--oxide-library-request-v2` serves compiler-admitted callable Reduce and
//! Merge Sort Keys comparator requests;
//! `--oxide-library-inputs-v1` passes them to [`serve_inputs`] during macro expansion.
//! Use `cuda_device::library::registration!` in the getter to track these inputs.
//! The compiler reads the registration getter's constant MIR and invokes this
//! host process. This module never invokes Cargo or executes device Rust code.
//! The protocol supports raw i32 and typed validity Load/Store recipes documented
//! by [`Operation`], numeric Block/scalar Warp Scan through [`ScanRecipe`], and
//! root-result scalar/payload Reduce through [`ReduceRecipe`], and partial-output
//! Warp batched Reduce through [`ReduceBatchedRecipe`]. Out-of-place Block/Warp
//! Exchange uses [`ExchangeRecipe`] with no scalar controls; Block Shuffle uses
//! [`ShuffleRecipe`] for one-item shifts with partial fresh output.
//! [`AdjacentDifferenceRecipe`] selects Block neighbor subtraction with typed
//! optional count and predecessor/successor controls.
//! [`DiscontinuityRecipe`] selects Block head/tail flags with one or two i32
//! outputs and optional neighbors of the input type. [`HistogramRecipe`] selects
//! bounded integer samples with an independent counter type and output extent.
//! [`RadixSortKeysRecipe`] selects stable Block key sorting with two signed64
//! uniform bit bounds and fresh blocked output. [`RadixSortPairsRecipe`] also
//! carries an independent numeric value type into a fresh paired output.
//! [`RadixRankKeysRecipe`] selects stable i32 ranks with a static digit width
//! and signed64 begin/end-constraint controls, preserving input-slot order.
//! [`MergeSortKeysRecipe`] selects numeric Block/Warp sorting with full output or a
//! signed64 count and key-typed sentinel for prefix-valid output.
//! Device calls retain all participation, validity and scratch obligations of
//! [`crate::Specialization`]. No allocation or synchronization policy is supplied.
//! Callable Reduce uses i32 full-input Block Raking and the same payload validity
//! and root-result ABI. Callable Merge Sort Keys uses full-input Block i32 and a
//! stateless comparator. Their thunks and borrowed environments come from Oxide's
//! checked callback marker; the provider does not select or execute Rust code.

use crate::{
    AdjacentDifferenceRecipe, AdjacentDifferenceRequest, Algorithm, Artifact, DiscontinuityRecipe,
    DiscontinuityRequest, Dtype, Error, ExchangeRecipe, ExchangeRequest, HistogramAlgorithm,
    HistogramRecipe, HistogramRequest, Layout, MergeSortKeysRecipe, MergeSortKeysRequest,
    MergeSortPairsRecipe, MergeSortPairsRequest, NativeRequest, Operation, Provider,
    RadixRankKeysRecipe, RadixRankKeysRequest, RadixSortKeysRecipe, RadixSortKeysRequest,
    RadixSortPairsRecipe, RadixSortPairsRequest, ReduceAlgorithm, ReduceBatchedRecipe,
    ReduceBatchedRequest, ReduceForm, ReduceOperator, ReduceRecipe, ReduceRequest, Request, Result,
    ScanRecipe, ScanRequest, ShuffleRecipe, ShuffleRequest, TopKCount, TopKKeysRequest,
    TopKPairsRequest, TopKRecipe, TopKSelection,
};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"OXDLIB01";
const MAX_HEADER: usize = 1024 * 1024;
const MAX_LTO: usize = 256 * 1024 * 1024;

#[derive(Serialize)]
struct Registration<'a> {
    version: u32,
    provider: &'a str,
    executable: &'a Path,
    operation: &'a str,
    algorithm: &'a str,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    content = "bits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Scalar {
    Signed(u16),
    Unsigned(u16),
    Float(u16),
}

impl Scalar {
    fn dtype(self) -> Result<Dtype> {
        match self {
            Self::Signed(8) => Ok(Dtype::I8),
            Self::Unsigned(8) => Ok(Dtype::U8),
            Self::Signed(16) => Ok(Dtype::I16),
            Self::Unsigned(16) => Ok(Dtype::U16),
            Self::Signed(32) => Ok(Dtype::I32),
            Self::Unsigned(32) => Ok(Dtype::U32),
            Self::Signed(64) => Ok(Dtype::I64),
            Self::Unsigned(64) => Ok(Dtype::U64),
            Self::Float(32) => Ok(Dtype::F32),
            Self::Float(64) => Ok(Dtype::F64),
            _ => Err(Error::message("unsupported scalar kind or width")),
        }
    }
}
impl From<Dtype> for Scalar {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::I8 => Self::Signed(8),
            Dtype::U8 => Self::Unsigned(8),
            Dtype::I16 => Self::Signed(16),
            Dtype::U16 => Self::Unsigned(16),
            Dtype::I32 => Self::Signed(32),
            Dtype::U32 => Self::Unsigned(32),
            Dtype::I64 => Self::Signed(64),
            Dtype::U64 => Self::Unsigned(64),
            Dtype::F32 => Self::Float(32),
            Dtype::F64 => Self::Float(64),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    content = "scalar",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Argument {
    ConstPointer(Scalar),
    MutPointer(Scalar),
    Value(Scalar),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchRequest<Request = SpecializationRequest> {
    version: u32,
    target: String,
    cuda_root: PathBuf,
    requests: Vec<Request>,
}

/// The compiler batches callback requests separately under protocol version two.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Callback {
    pub(crate) symbol: String,
    arguments: Vec<Argument>,
    result: Scalar,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CallbackRequest {
    operation: String,
    algorithm: String,
    dtype: Scalar,
    block: [u32; 3],
    items_per_thread: u32,
    #[serde(default)]
    warp_width: Option<u32>,
    callback: Callback,
}

impl CallbackRequest {
    fn native(&self) -> Result<NativeRequest> {
        if self.dtype != Scalar::Signed(32) || self.warp_width.is_some() {
            return Err(Error::message(
                "callable recipes require i32 Block payloads",
            ));
        }
        if self.callback.symbol.is_empty()
            || !self
                .callback
                .symbol
                .bytes()
                .enumerate()
                .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
            || self.callback.arguments
                != [
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::Value(Scalar::Signed(32)),
                    Argument::Value(Scalar::Signed(32)),
                ]
            || self.callback.result != Scalar::Signed(32)
        {
            return Err(Error::message(
                "callable recipes require a C identifier and the (const environment*, i32, i32) -> i32 callback ABI",
            ));
        }
        match (self.operation.as_str(), self.algorithm.as_str()) {
            ("reduce_callable_payload_validity_bits_v1", "raking") => ReduceRequest::block(
                Dtype::I32,
                ReduceOperator::Sum,
                ReduceAlgorithm::Raking,
                ReduceForm::Payload,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::Reduce),
            ("merge_sort_keys_callable_payload_validity_bits_v1", "block") => {
                MergeSortKeysRequest::block(
                    Dtype::I32,
                    MergeSortKeysRecipe::AscendingFull,
                    self.block,
                    self.items_per_thread,
                )
                .map(NativeRequest::MergeSortKeys)
            }
            _ => Err(Error::message(
                "unsupported callable operation or algorithm",
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecializationRequest {
    operation: String,
    algorithm: String,
    dtype: Scalar,
    block: [u32; 3],
    items_per_thread: u32,
    #[serde(default)]
    warp_width: Option<u32>,
}

impl SpecializationRequest {
    fn native(&self) -> Result<NativeRequest> {
        if self.operation.starts_with("topk_") {
            if self.algorithm != "block" || self.warp_width.is_some() {
                return Err(Error::message(
                    "TopK requires a Block recipe without Warp width",
                ));
            }
            let (recipe, value) = TopKRecipe::parse(&self.operation)?;
            return if let Some(value) = value {
                TopKPairsRequest::block(
                    self.dtype.dtype()?,
                    value,
                    recipe,
                    self.block,
                    self.items_per_thread,
                )
                .map(NativeRequest::TopKPairs)
            } else {
                TopKKeysRequest::block(
                    self.dtype.dtype()?,
                    recipe,
                    self.block,
                    self.items_per_thread,
                )
                .map(NativeRequest::TopKKeys)
            };
        }
        if self.operation.starts_with("merge_sort_pairs_") {
            let dtype = self.dtype.dtype()?;
            let (recipe, value) = MergeSortPairsRecipe::parse(&self.operation, &self.algorithm)?;
            return match (self.algorithm.as_str(), self.warp_width) {
                ("block", None) => MergeSortPairsRequest::block(
                    dtype,
                    value,
                    recipe,
                    self.block,
                    self.items_per_thread,
                ),
                ("warp", Some(width)) => MergeSortPairsRequest::warp(
                    dtype,
                    value,
                    recipe,
                    self.block,
                    width,
                    self.items_per_thread,
                ),
                _ => Err(Error::message(
                    "Merge Sort Pairs scope and width must agree",
                )),
            }
            .map(NativeRequest::MergeSortPairs);
        }
        if self.operation.starts_with("merge_sort_keys_") {
            let recipe = MergeSortKeysRecipe::parse(&self.operation, &self.algorithm)?;
            let dtype = self.dtype.dtype()?;
            return match (self.algorithm.as_str(), self.warp_width) {
                ("block", None) => {
                    MergeSortKeysRequest::block(dtype, recipe, self.block, self.items_per_thread)
                }
                ("warp", Some(width)) => MergeSortKeysRequest::warp(
                    dtype,
                    recipe,
                    self.block,
                    width,
                    self.items_per_thread,
                ),
                _ => Err(Error::message(
                    "Merge Sort Keys scope and warp width disagree",
                )),
            }
            .map(NativeRequest::MergeSortKeys);
        }
        if self.operation.starts_with("radix_rank_keys_") {
            let recipe = RadixRankKeysRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Radix Rank Keys requires a Block scope"));
            }
            return RadixRankKeysRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::RadixRankKeys);
        }
        if self.operation.starts_with("radix_sort_pairs_") {
            let (recipe, value_dtype) =
                RadixSortPairsRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Radix Sort Pairs requires a Block scope"));
            }
            return RadixSortPairsRequest::block(
                self.dtype.dtype()?,
                value_dtype,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::RadixSortPairs);
        }
        if self.operation.starts_with("radix_sort_keys_") {
            let recipe = RadixSortKeysRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Radix Sort Keys requires a Block scope"));
            }
            return RadixSortKeysRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::RadixSortKeys);
        }
        if self.operation.starts_with("histogram_") {
            let recipe = HistogramRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Histogram requires a Block scope"));
            }
            return HistogramRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::Histogram);
        }
        if self.operation.starts_with("discontinuity_") {
            let recipe = DiscontinuityRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Discontinuity requires a Block scope"));
            }
            return DiscontinuityRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::Discontinuity);
        }
        if self.operation.starts_with("adjacent_difference_") {
            let recipe = AdjacentDifferenceRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Adjacent Difference requires a Block scope"));
            }
            return AdjacentDifferenceRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::AdjacentDifference);
        }
        if self.operation.starts_with("shuffle_") {
            let recipe = ShuffleRecipe::parse(&self.operation, &self.algorithm)?;
            if self.warp_width.is_some() {
                return Err(Error::message("Shuffle requires a Block scope"));
            }
            return ShuffleRequest::block(
                self.dtype.dtype()?,
                recipe,
                self.block,
                self.items_per_thread,
            )
            .map(NativeRequest::Shuffle);
        }
        if self.operation.starts_with("exchange_") {
            let recipe = ExchangeRecipe::parse(&self.operation, &self.algorithm)?;
            let dtype = self.dtype.dtype()?;
            let request = match (recipe, self.warp_width) {
                (ExchangeRecipe::Block(_), None) => {
                    ExchangeRequest::block(dtype, recipe, self.block, self.items_per_thread)?
                }
                (ExchangeRecipe::Warp(_), Some(width)) => {
                    ExchangeRequest::warp(dtype, recipe, self.block, width, self.items_per_thread)?
                }
                _ => return Err(Error::message("Exchange recipe disagrees with Warp scope")),
            };
            return Ok(NativeRequest::Exchange(request));
        }
        if self.operation.starts_with("reduce_batched_") {
            let recipe = ReduceBatchedRecipe::parse(&self.operation, &self.algorithm)?;
            let width = self
                .warp_width
                .ok_or_else(|| Error::message("batched Reduce requires a Warp scope"))?;
            return ReduceBatchedRequest::warp(
                self.dtype.dtype()?,
                recipe,
                self.block,
                width,
                self.items_per_thread,
            )
            .map(NativeRequest::ReduceBatched);
        }
        if self.operation.starts_with("reduce_") {
            let recipe = ReduceRecipe::parse(&self.operation, &self.algorithm)?;
            let dtype = self.dtype.dtype()?;
            let request = match (recipe, self.warp_width) {
                (ReduceRecipe::Block(op, algorithm, form), None) => ReduceRequest::block(
                    dtype,
                    op,
                    algorithm,
                    form,
                    self.block,
                    self.items_per_thread,
                )?,
                (ReduceRecipe::Warp(op, form), Some(width)) => {
                    ReduceRequest::warp(dtype, op, form, self.block, width, self.items_per_thread)?
                }
                _ => return Err(Error::message("Reduce recipe disagrees with Warp scope")),
            };
            return Ok(NativeRequest::Reduce(request));
        }
        if self.operation.starts_with("scan_") {
            let recipe = ScanRecipe::parse(&self.operation, &self.algorithm)?;
            let dtype = self.dtype.dtype()?;
            let request = match (recipe, self.warp_width) {
                (ScanRecipe::Block(operation, algorithm), None) => ScanRequest::block(
                    dtype,
                    operation,
                    algorithm,
                    self.block,
                    self.items_per_thread,
                )?,
                (ScanRecipe::Warp(operation), Some(width)) if self.items_per_thread == 1 => {
                    ScanRequest::warp(dtype, operation, self.block, width)?
                }
                _ => {
                    return Err(Error::message(
                        "Scan recipe disagrees with Warp scope or scalar extent",
                    ));
                }
            };
            return Ok(NativeRequest::Scan(request));
        }
        let operation = match self.operation.as_str() {
            "load_full" => Operation::LoadFull,
            "load_filled" => Operation::LoadFilled,
            "store_full" => Operation::StoreFull,
            "store_partial" => Operation::StorePartial,
            "load_full_validity_bits_v1" => Operation::LoadFullValidity,
            "load_masked_validity_bits_v1" => Operation::LoadMaskedValidity,
            "load_filled_validity_bits_v1" => Operation::LoadFilledValidity,
            "store_full_validity_bits_v1" => Operation::StoreFullValidity,
            "store_partial_validity_bits_v1" => Operation::StorePartialValidity,
            _ => {
                return Err(Error::message(format!(
                    "unsupported operation {:?}",
                    self.operation
                )));
            }
        };
        let algorithm = match self.algorithm.as_str() {
            "direct" => Algorithm::Direct,
            "striped" => Algorithm::Striped,
            "vectorize" => Algorithm::Vectorize,
            "transpose" => Algorithm::Transpose,
            "warp_transpose" => Algorithm::WarpTranspose,
            "warp_transpose_timesliced" => Algorithm::WarpTransposeTimesliced,
            _ => {
                return Err(Error::message(format!(
                    "unsupported algorithm {:?}",
                    self.algorithm
                )));
            }
        };
        let request = Request::new_typed_with_shape(
            self.dtype.dtype()?,
            operation,
            algorithm,
            self.block,
            self.items_per_thread,
        )?;
        match self.warp_width {
            Some(width) => request.with_warp_width(width).map(NativeRequest::LoadStore),
            None => Ok(NativeRequest::LoadStore(request)),
        }
    }
}

#[derive(Serialize)]
struct WireLayout {
    size: u64,
    alignment: u64,
}

impl TryFrom<Layout> for WireLayout {
    type Error = Error;
    fn try_from(layout: Layout) -> Result<Self> {
        Ok(Self {
            size: u64::try_from(layout.size)
                .map_err(|_| Error::message("layout size exceeds u64"))?,
            alignment: u64::try_from(layout.alignment)
                .map_err(|_| Error::message("layout alignment exceeds u64"))?,
        })
    }
}

#[derive(Serialize)]
struct SpecializationResponse {
    request_index: u32,
    identity: [u8; 32],
    symbol: String,
    arguments: Vec<Argument>,
    cub_layout: WireLayout,
    scratch: WireLayout,
    scratch_free: bool,
    mutates_payload: bool,
}

#[derive(Serialize)]
struct BatchResponse {
    version: u32,
    batch_identity: [u8; 32],
    specializations: Vec<SpecializationResponse>,
}

fn arguments(request: NativeRequest) -> Vec<Argument> {
    let request = match request {
        NativeRequest::TopKKeys(request) => {
            let key = Scalar::from(request.dtype);
            let mut args = vec![
                Argument::ConstPointer(key),
                Argument::MutPointer(key),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
            ];
            if request.recipe.valid_items.is_some() {
                args.push(Argument::Value(Scalar::Signed(64)));
            }
            return args;
        }
        NativeRequest::TopKPairs(request) => {
            let key = Scalar::from(request.keys.dtype);
            let value = Scalar::from(request.value_dtype);
            let mut args = vec![
                Argument::ConstPointer(key),
                Argument::MutPointer(key),
                Argument::ConstPointer(value),
                Argument::MutPointer(value),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
            ];
            if request.keys.recipe.valid_items.is_some() {
                args.push(Argument::Value(Scalar::Signed(64)));
            }
            return args;
        }
        NativeRequest::MergeSortPairs(request) => {
            let key = Scalar::from(request.key_dtype);
            let value = Scalar::from(request.value_dtype);
            let mut arguments = vec![
                Argument::ConstPointer(key),
                Argument::MutPointer(key),
                Argument::ConstPointer(value),
                Argument::MutPointer(value),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if request.recipe.partial() {
                arguments.extend([Argument::Value(Scalar::Signed(64)), Argument::Value(key)]);
            }
            return arguments;
        }
        NativeRequest::MergeSortKeys(request) => {
            let dtype = Scalar::from(request.dtype);
            let mut arguments = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if request.recipe.partial() {
                arguments.extend([Argument::Value(Scalar::Signed(64)), Argument::Value(dtype)]);
            }
            return arguments;
        }
        NativeRequest::RadixRankKeys(request) => {
            return vec![
                Argument::ConstPointer(Scalar::from(request.dtype)),
                Argument::MutPointer(Scalar::Signed(32)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
                Argument::Value(Scalar::Signed(64)),
            ];
        }
        NativeRequest::RadixSortPairs(request) => {
            let key = Scalar::from(request.key_dtype);
            let value = Scalar::from(request.value_dtype);
            return vec![
                Argument::ConstPointer(key),
                Argument::MutPointer(key),
                Argument::ConstPointer(value),
                Argument::MutPointer(value),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
                Argument::Value(Scalar::Signed(64)),
            ];
        }
        NativeRequest::RadixSortKeys(request) => {
            let dtype = Scalar::from(request.dtype);
            return vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
                Argument::Value(Scalar::Signed(64)),
            ];
        }
        NativeRequest::Histogram(request) => {
            return vec![
                Argument::ConstPointer(Scalar::from(request.dtype)),
                Argument::MutPointer(Scalar::from(request.recipe.counter)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
        }
        NativeRequest::Discontinuity(request) => {
            let dtype = Scalar::from(request.dtype);
            let mut arguments = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(Scalar::Signed(32)),
            ];
            if request.recipe.paired() {
                arguments.push(Argument::MutPointer(Scalar::Signed(32)));
            }
            arguments.extend([
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ]);
            if request.recipe.paired() {
                arguments.push(Argument::MutPointer(Scalar::Unsigned(8)));
            }
            if request.recipe.predecessor() {
                arguments.push(Argument::Value(dtype));
            }
            if request.recipe.successor() {
                arguments.push(Argument::Value(dtype));
            }
            return arguments;
        }
        NativeRequest::AdjacentDifference(request) => {
            let dtype = Scalar::from(request.dtype);
            let mut arguments = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if request.recipe.counted() {
                arguments.push(Argument::Value(Scalar::Signed(32)));
            }
            if request.recipe.has_neighbor() {
                arguments.push(Argument::Value(dtype));
            }
            return arguments;
        }
        NativeRequest::Shuffle(request) => {
            let dtype = Scalar::from(request.dtype);
            return vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
        }
        NativeRequest::Exchange(request) => {
            let dtype = Scalar::from(request.dtype);
            return vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
        }
        NativeRequest::ReduceBatched(request) => {
            let dtype = Scalar::from(request.dtype);
            return vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
        }
        NativeRequest::Reduce(request) => {
            let dtype = Scalar::from(request.dtype);
            let mut arguments = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if request.recipe.form() == ReduceForm::ScalarPrefix {
                arguments.push(Argument::Value(Scalar::Signed(32)));
            }
            return arguments;
        }
        NativeRequest::Scan(request) => {
            let dtype = Scalar::from(request.dtype);
            let mut arguments = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if request.recipe.operation().seeded() {
                arguments.push(Argument::Value(dtype));
            }
            return arguments;
        }
        NativeRequest::LoadStore(request) => request,
    };
    let operation = request.operation();
    let dtype = Scalar::from(request.dtype());
    let shared_payload =
        operation.validity() && !operation.is_load() && request.algorithm().scratch_free();
    let mut arguments = vec![
        if operation.is_load() {
            Argument::ConstPointer(dtype)
        } else {
            Argument::MutPointer(dtype)
        },
        if shared_payload {
            Argument::ConstPointer(dtype)
        } else {
            Argument::MutPointer(dtype)
        },
    ];
    if operation.validity() {
        arguments.push(if shared_payload {
            Argument::ConstPointer(Scalar::Unsigned(8))
        } else {
            Argument::MutPointer(Scalar::Unsigned(8))
        });
    }
    if operation.counted() {
        arguments.push(Argument::Value(Scalar::Signed(32)));
    }
    if operation.filled() {
        arguments.push(Argument::Value(dtype));
    }
    arguments
}

/// Register the public Direct recipes from cuda-coop's owning build script.
///
/// Normal Cargo builds only emit registrations for the current build-script
/// executable. Oxide later invokes that executable with the existing native
/// request/input protocol. Headers come from this codegen crate's complete CCCL
/// checkout, independent of the consumer's working directory. Host checks and
/// documentation do not load CUDA. Native compilation uses Oxide's selected
/// toolkit and target; no consumer build script is required.
///
/// # Errors
///
/// Returns errors locating this checkout or executable, encoding registrations,
/// or serving a native request. Native request errors are those of [`serve`]
/// and [`serve_inputs`].
pub fn build_script() -> Result<()> {
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    match std::env::args().nth(1).as_deref() {
        Some("--oxide-library-request-v1") => {
            return serve(&checkout, std::io::stdin().lock(), std::io::stdout().lock());
        }
        Some("--oxide-library-request-v2") => {
            return serve_version(
                &checkout,
                std::io::stdin().lock(),
                std::io::stdout().lock(),
                2,
            );
        }
        Some("--oxide-library-inputs-v1") => {
            return serve_inputs(&checkout, std::io::stdin().lock(), std::io::stdout().lock());
        }
        _ => {}
    }
    println!("cargo::rerun-if-changed=build.rs");
    let executable = std::env::current_exe()?;
    for (operation, name) in [
        (Operation::LoadFullValidity, "LOAD_FULL"),
        (Operation::LoadMaskedValidity, "LOAD_MASKED"),
        (Operation::LoadFilledValidity, "LOAD_FILLED"),
        (Operation::StoreFullValidity, "STORE_FULL"),
        (Operation::StorePartialValidity, "STORE_PARTIAL"),
    ] {
        let value = registration(&executable, operation, Algorithm::Direct)?;
        println!("cargo::rustc-env=COOP_VALIDITY_{name}={value}");
    }
    Ok(())
}

/// Encode a constant registration for one native recipe.
///
/// `executable` is the live, already-built dependency build script, normally
/// obtained from `std::env::current_exe()`. Embed the returned string in the
/// device crate through `cargo::rustc-env`; a named getter must return that
/// constant directly. The marker uses the getter's function-item type.
///
/// # Errors
///
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn registration(
    executable: &Path,
    operation: Operation,
    algorithm: Algorithm,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: operation.token(),
        algorithm: match algorithm {
            Algorithm::Direct => "direct",
            Algorithm::Striped => "striped",
            Algorithm::Vectorize => "vectorize",
            Algorithm::Transpose => "transpose",
            Algorithm::WarpTranspose => "warp_transpose",
            Algorithm::WarpTransposeTimesliced => "warp_transpose_timesliced",
        },
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one Scan recipe for the existing dependency provider executable.
///
/// The Block recipe selects a Block Scan algorithm; the Warp recipe has no
/// explicit algorithm. Dtype, geometry and extent arrive with specialization.
/// Seeded exclusive operations take one runtime value of exactly that dtype;
/// inclusive and zero-seeded sum take no scalar controls.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn scan_registration(executable: &Path, recipe: ScanRecipe) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.operation().token(),
        algorithm: recipe.algorithm_token(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Register full-input i32 Block Reduce with a compiler-admitted Rust callable.
///
/// The compiler supplies the concrete thunk and environment through protocol
/// version two. Input validity and root-only output use the payload ABI.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn reduce_callable_registration(executable: &Path) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: "reduce_callable_payload_validity_bits_v1",
        algorithm: "raking",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Register full-input i32 Block Merge Sort Keys with a stateless Rust comparator.
///
/// The compiler supplies a genuine i32-return thunk under protocol version two;
/// the native comparator converts its result to bool. Output retains all keys.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn merge_sort_keys_callable_registration(executable: &Path) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: "merge_sort_keys_callable_payload_validity_bits_v1",
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one Reduce recipe for the existing dependency provider executable.
///
/// ScalarPrefix adds one signed 32-bit uniform count. Scalar and Payload add
/// no controls; their distinct identities preserve scalar-only count support.
/// Warp recipes use implicit algorithm selection. Dtype, geometry and input
/// extent arrive with specialization; the output always has one slot.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn reduce_registration(executable: &Path, recipe: ReduceRecipe) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(),
        algorithm: recipe.algorithm_token(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a Warp batched-reduction recipe for the existing provider executable.
///
/// Dtype, exact Block geometry, Warp width and batch count arrive with
/// specialization. Each lane's output capacity is derived as ceil(batches/width).
/// There are four typed payload pointers and no scalar controls; scratch is
/// inserted by the existing library-call ABI.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn reduce_batched_registration(
    executable: &Path,
    recipe: ReduceBatchedRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(),
        algorithm: "warp",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a Block or Warp Exchange recipe for the existing provider executable.
///
/// Dtype, exact Block geometry, optional Warp width and payload extent arrive
/// with specialization. Four typed payload pointers describe preserved input
/// and fresh same-extent output with packed validity; there are no scalar
/// controls. Scratch is inserted by the existing library-call ABI.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn exchange_registration(executable: &Path, recipe: ExchangeRecipe) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: recipe.algorithm_token(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a one-item Block Shuffle recipe for the existing provider executable.
///
/// Dtype, exact Block geometry and payload extent arrive with specialization.
/// Four typed payload pointers describe preserved input and fresh same-extent
/// output with packed validity, including its one invalid boundary slot. There
/// are no scalar controls; scratch is inserted by the existing library-call ABI.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn shuffle_registration(executable: &Path, recipe: ShuffleRecipe) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of eight tracked Block TopK registration templates.
///
/// `pairs` selects two input/output planes; `partial` adds a signed64 validity
/// control after k. The API's family-local constant formatter replaces only
/// the closed count fields and, for pairs, the three-byte value tag. The key
/// dtype, exact Block and extent arrive through the genuine checked marker.
/// Scratch is injected at slot 2: keys use six/seven native arguments and
/// pairs ten/eleven. See [`TopKKeysRequest`] and [`TopKPairsRequest`] for input,
/// validity, selection and shared-storage obligations.
///
/// # Errors
/// Rejects a nonabsolute executable or JSON encoding failure.
pub fn topk_registration_template(
    executable: &Path,
    selection: TopKSelection,
    pairs: bool,
    partial: bool,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    let recipe = TopKRecipe {
        selection,
        k: TopKCount::Runtime,
        valid_items: partial.then_some(TopKCount::Runtime),
    };
    // i32 occupies exactly the same three-byte field as every sealed value tag.
    let operation = recipe.token(pairs.then_some(Dtype::I32));
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &operation,
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of four Block Merge Sort Keys order/prefix registrations.
///
/// Key dtype, exact XYZ and per-lane extent arrive with specialization. Four
/// typed pointers describe preserved input and fresh output with independent
/// packed validity. Full recipes have no controls; partial recipes append a
/// signed64 count and sentinel of exactly the key type. Scratch is inserted at
/// slot 2, giving five or seven native arguments. See [`MergeSortKeysRequest`]
/// for input validity, sentinel ordering, scratch and participation obligations.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn merge_sort_keys_registration(
    executable: &Path,
    recipe: MergeSortKeysRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of four physical/logical Warp Merge Sort Keys registrations.
///
/// Uses the same typed five/seven-argument native ABI as
/// [`merge_sort_keys_registration`]. The genuine checked Warp marker supplies
/// the logical width and exact enclosing XYZ Block at specialization. Only
/// widths 1/2/4/8/16/32 and complete physical warps are supported. Controls agree
/// within each group; the compiler supplies its independent scratch replica.
/// See [`MergeSortKeysRequest`] for input, validity and sentinel obligations.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn warp_merge_sort_keys_registration(
    executable: &Path,
    recipe: MergeSortKeysRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "warp",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of forty independently typed Block Merge Sort Pairs registrations.
///
/// Key dtype, exact XYZ and shared extent arrive with specialization. Value
/// dtype is selected here; eight payload pointers preserve both input types
/// and their separate validity maps. Scratch at slot 2 gives nine native
/// arguments, or eleven when a partial recipe appends `(i64 count, K sentinel)`.
/// See [`MergeSortPairsRequest`] for participation, validity and scratch rules.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn merge_sort_pairs_registration(
    executable: &Path,
    recipe: MergeSortPairsRecipe,
    value_dtype: Dtype,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(value_dtype),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of forty independently typed Warp Merge Sort Pairs registrations.
///
/// Key dtype, exact XYZ and shared extent arrive with specialization. Value
/// dtype is selected here; eight payload pointers preserve both input types
/// and their separate validity maps. Scratch at slot 2 gives nine native
/// arguments, or eleven when a partial recipe appends `(i64 count, K sentinel)`.
/// See [`MergeSortPairsRequest`] for participation, validity and scratch rules.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn warp_merge_sort_pairs_registration(
    executable: &Path,
    recipe: MergeSortPairsRecipe,
    value_dtype: Dtype,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(value_dtype),
        algorithm: "warp",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of sixteen Block Radix Rank Keys order/width registrations.
///
/// `recipe` selects ascending/descending order and 1..=8 radix bits. Key type,
/// exact XYZ and per-lane extent arrive with specialization. Four pointers
/// describe preserved keys, fresh i32 ranks and independent validity maps.
/// Two signed64 uniform controls follow: begin bit and an explicit static end
/// constraint, or -1 for omitted end. Scratch is inserted at slot 2, giving
/// seven native arguments. See [`RadixRankKeysRequest`] for the native contract.
///
/// # Errors
/// Rejects widths outside 1..=8, a nonabsolute executable or an unencodable path.
pub fn radix_rank_keys_registration(
    executable: &Path,
    recipe: RadixRankKeysRecipe,
) -> Result<String> {
    recipe.validate()?;
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a Block Radix Sort Keys recipe for the existing provider executable.
///
/// Key dtype, exact XYZ geometry and per-lane extent arrive with specialization.
/// Four typed pointers describe preserved keys and fresh same-extent output
/// with independent packed validity maps. Two signed64 uniform controls follow:
/// beginning bit and exclusive ending bit. The existing ABI inserts scratch
/// at slot 2. Bounds are runtime values, never part of specialization identity.
/// See [`RadixSortKeysRequest`] for bounds, layout and participation obligations.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn radix_sort_keys_registration(
    executable: &Path,
    recipe: RadixSortKeysRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode one of the twenty typed Block Radix Sort Pairs registrations.
///
/// `recipe` selects order and `value_dtype` selects the independent value type.
/// Key dtype, common per-lane extent and exact XYZ arrive with specialization.
/// Eight typed pointers describe key/value inputs, fresh outputs and four
/// independent validity maps. Two signed64 uniform bit bounds follow; the
/// existing native ABI inserts scratch at slot 2, for eleven arguments total.
/// See [`RadixSortPairsRequest`] for the complete native contract.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn radix_sort_pairs_registration(
    executable: &Path,
    recipe: RadixSortPairsRecipe,
    value_dtype: Dtype,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: &recipe.token(value_dtype),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a Block Adjacent Difference recipe for the existing provider executable.
///
/// Dtype, exact Block geometry and payload extent arrive with specialization.
/// Four typed payload pointers describe preserved input and fully initialized
/// fresh output with packed validity. Partial recipes append an `i32` count;
/// recipes with a tile predecessor or successor append that value in the exact
/// input dtype, after the count when present. Scratch is inserted by the
/// existing library-call ABI. See [`AdjacentDifferenceRequest`] for capacities,
/// boundary behavior and participation requirements.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn adjacent_difference_registration(
    executable: &Path,
    recipe: AdjacentDifferenceRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a Block Discontinuity recipe for the existing provider executable.
///
/// Input dtype, exact Block geometry and payload extent arrive with
/// specialization. Single-result recipes use four typed payload pointers;
/// paired recipes use six, ordered input, heads, tails, input validity, heads
/// validity, tails validity. Outputs are i32 regardless of input dtype.
/// Optional predecessor and successor controls follow, in that order, with
/// exactly the input type. Scratch is inserted at native parameter two by the
/// existing ABI. See [`DiscontinuityRequest`] for initialization and reuse.
///
/// # Errors
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn discontinuity_registration(
    executable: &Path,
    recipe: DiscontinuityRecipe,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: recipe.token(),
        algorithm: "block",
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

/// Encode a tracked Histogram registration template for one algorithm.
///
/// The operation contains `xxx` for a sealed i32/u32/i64/u64 counter tag and
/// ten zero digits each for the total bin count and per-lane output extent.
/// A device getter fills those fixed-width fields in its associated constant,
/// retaining the [`serve_inputs`] dependency query from `registration!`.
/// The unfilled template is not a valid specialization.
///
/// Input type/extent and exact Block shape arrive through the genuine marker.
/// Output type/extent are independent. Native arguments are input, output,
/// scratch, input bitmap, output bitmap; there are no scalar controls.
/// See [`HistogramRequest`] for preflight, output validity and scratch reuse.
///
/// # Errors
/// Rejects a nonabsolute provider executable or an unencodable path.
pub fn histogram_registration_template(
    executable: &Path,
    algorithm: HistogramAlgorithm,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: crate::histogram::TEMPLATE_OPERATION,
        algorithm: algorithm.token(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRequest {
    version: u32,
    cuda_root: PathBuf,
    target: String,
}

#[derive(Serialize)]
struct InputResponse {
    version: u32,
    paths: Vec<PathBuf>,
    environment: Vec<String>,
}

/// Declare native inputs for Oxide's registration macro without loading CUDA.
///
/// `checkout` selects the same CCCL tree as [`serve`]. The bounded version-one
/// JSON request contains `cuda_root` and `target`; the response lists header
/// directories and NVRTC libraries. The macro tracks files and directories
/// recursively before rustc emits dep-info, including new header files.
/// No GPU, native compiler invocation or library loading occurs here.
/// Cargo's timestamp freshness rules apply; replacements with unchanged or older
/// timestamps can remain undetected. Rebuild after restoring or replacing inputs
/// this way, including a symlink switched to an older installation.
///
/// # Errors
///
/// Rejects malformed, oversized or unsupported requests, nonabsolute toolkit
/// paths, invalid targets, and inaccessible checkout paths.
pub fn serve_inputs(checkout: &Path, input: impl Read, mut output: impl Write) -> Result<()> {
    let mut bytes = Vec::new();
    input
        .take((MAX_HEADER + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_HEADER {
        return Err(Error::message("provider input request exceeds 1 MiB"));
    }
    let request: InputRequest = serde_json::from_slice(&bytes)
        .map_err(|error| Error::message(format!("invalid provider input request: {error}")))?;
    if request.version != VERSION
        || !request.cuda_root.is_absolute()
        || !request.target.starts_with("compute_")
        || request.target.len() <= 8
        || !request.target[8..]
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    {
        return Err(Error::message(
            "provider inputs require version 1, an absolute toolkit path and compute_XX target",
        ));
    }
    let checkout = checkout
        .canonicalize()
        .map_err(|error| Error::input(checkout, error))?;
    let paths = crate::headers::Headers::roots(&checkout, &request.cuda_root)
        .into_iter()
        .map(|(path, _)| path)
        .chain(crate::nvrtc::Compiler::input_paths(&request.cuda_root))
        .collect();
    let response = serde_json::to_vec(&InputResponse {
        version: VERSION,
        paths,
        environment: Vec::new(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider inputs: {error}")))?;
    if response.len() > MAX_HEADER {
        return Err(Error::message("provider input response exceeds 1 MiB"));
    }
    output.write_all(&response)?;
    Ok(())
}

/// Compile one versioned stdin request and write its framed native response.
///
/// `checkout` selects the CCCL checkout whose headers the provider captures.
/// The request supplies an explicit toolkit path and target. Stdout contains
/// only the frame; the caller reports errors on stderr and exits nonzero.
/// The 24-byte prefix is `OXDLIB01`, JSON header length, and raw LTO length
/// (both lengths are little-endian u64), followed by the two byte buffers.
/// Requests and response headers are limited to 1 MiB; LTO is limited to 256 MiB.
/// Runtime arguments are a scalar/raw-pointer tuple; scratch is inserted at
/// parameter two and uses a generic byte pointer after shared address conversion.
///
/// # Errors
///
/// Rejects malformed/version-mismatched requests, unsupported types/recipes,
/// invalid geometry, and duplicate specializations. Native compiler errors
/// retain NVRTC diagnostics. No response is written until compilation succeeds.
pub fn serve(checkout: &Path, input: impl Read, output: impl Write) -> Result<()> {
    serve_version(checkout, input, output, VERSION)
}

fn serve_version(
    checkout: &Path,
    input: impl Read,
    mut output: impl Write,
    version: u32,
) -> Result<()> {
    let mut bytes = Vec::new();
    input
        .take((MAX_HEADER + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_HEADER {
        return Err(Error::message("provider request exceeds 1 MiB"));
    }
    if version == 2 {
        let request: BatchRequest<CallbackRequest> =
            serde_json::from_slice(&bytes).map_err(|error| {
                Error::message(format!("invalid callback provider request: {error}"))
            })?;
        if request.version != version {
            return Err(Error::message(
                "callback provider requires protocol version 2",
            ));
        }
        let requests = request
            .requests
            .iter()
            .map(CallbackRequest::native)
            .collect::<Result<Vec<_>>>()?;
        let callbacks = request
            .requests
            .into_iter()
            .map(|request| request.callback)
            .collect::<Vec<_>>();
        let artifact = Provider::from_checkout(checkout, &request.cuda_root)?
            .compile_native_with_callbacks(&requests, &request.target, Some(&callbacks))?;
        return write_response(&requests, artifact, &mut output, version);
    }
    let request: BatchRequest = serde_json::from_slice(&bytes)
        .map_err(|error| Error::message(format!("invalid provider request: {error}")))?;
    if request.version != VERSION {
        return Err(Error::message(format!(
            "unsupported provider protocol {}; expected {VERSION}",
            request.version
        )));
    }
    let requests: Vec<_> = request
        .requests
        .iter()
        .map(SpecializationRequest::native)
        .collect::<Result<_>>()?;
    let artifact = Provider::from_checkout(checkout, &request.cuda_root)?
        .compile_native(&requests, &request.target)?;
    write_response(&requests, artifact, &mut output, version)
}

fn write_response(
    requests: &[NativeRequest],
    artifact: Artifact,
    output: &mut impl Write,
    version: u32,
) -> Result<()> {
    if artifact.ltoir.is_empty() || artifact.ltoir.len() > MAX_LTO {
        return Err(Error::message(
            "native LTO buffer must contain 1..=256 MiB bytes",
        ));
    }
    let specializations = artifact
        .specializations
        .into_iter()
        .zip(requests)
        .enumerate()
        .map(|(index, (entry, request))| {
            Ok(SpecializationResponse {
                request_index: u32::try_from(index)
                    .map_err(|_| Error::message("request index exceeds u32"))?,
                identity: entry.identity,
                symbol: entry.symbol,
                // Both protocols describe only the caller's payload/uniform
                // tuple. Oxide inserts scratch and the V2 environment itself.
                arguments: arguments(*request),
                cub_layout: entry.cub_layout.try_into()?,
                scratch: entry.scratch.try_into()?,
                scratch_free: entry.scratch_free,
                mutates_payload: entry.mutates_payload,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = serde_json::to_vec(&BatchResponse {
        version,
        batch_identity: artifact.identity,
        specializations,
    })
    .map_err(|error| Error::message(format!("cannot encode provider response: {error}")))?;
    if header.len() > MAX_HEADER {
        return Err(Error::message("provider response header exceeds 1 MiB"));
    }
    output.write_all(if version == 2 { b"OXDLIB02" } else { MAGIC })?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&(artifact.ltoir.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&artifact.ltoir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    fn callable_request(symbol: &str, items: u32) -> serde_json::Value {
        serde_json::json!({
            "operation":"reduce_callable_payload_validity_bits_v1", "algorithm":"raking",
            "dtype":{"kind":"signed","bits":32}, "block":[128,1,1], "items_per_thread":items,
            "callback":{
                "symbol":symbol,
                "arguments":[
                    {"kind":"const_pointer","scalar":{"kind":"unsigned","bits":8}},
                    {"kind":"value","scalar":{"kind":"signed","bits":32}},
                    {"kind":"value","scalar":{"kind":"signed","bits":32}}
                ],
                "result":{"kind":"signed","bits":32}
            }
        })
    }

    #[test]
    fn callable_protocol_rejects_wrong_recipe_or_thunk_before_toolkit() {
        let base = callable_request("rust_reduce_thunk", 4);
        assert!(serde_json::from_value::<SpecializationRequest>(base.clone()).is_err());
        let mut missing = base.clone();
        missing.as_object_mut().unwrap().remove("callback");
        assert!(serde_json::from_value::<CallbackRequest>(missing).is_err());
        let parsed: CallbackRequest = serde_json::from_value(base.clone()).unwrap();
        assert!(parsed.native().is_ok());
        for (path, value) in [
            (
                "/operation",
                serde_json::json!("reduce_sum_payload_validity_bits_v1"),
            ),
            ("/algorithm", serde_json::json!("warp_reductions")),
            ("/dtype/bits", serde_json::json!(64)),
            ("/block/0", serde_json::json!(0)),
            ("/items_per_thread", serde_json::json!(0)),
            ("/callback/symbol", serde_json::json!("bad;symbol")),
            (
                "/callback/arguments/0/kind",
                serde_json::json!("mut_pointer"),
            ),
            ("/callback/result/bits", serde_json::json!(64)),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let request = serde_json::json!({"version":2, "target":"compute_120", "cuda_root":"/absent", "requests":[request]});
            let mut output = Vec::new();
            let error = serve_version(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                &mut output,
                2,
            )
            .unwrap_err();
            assert!(!error.to_string().contains("No such file"), "{error}");
            assert!(output.is_empty());
        }
    }

    #[test]
    #[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
    fn native_callable_reduce_identity_and_v2_response()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let requests = [
            callable_request("rust_add", 1),
            callable_request("rust_add", 4),
            callable_request("rust_max", 4),
        ]
        .into_iter()
        .map(serde_json::from_value::<CallbackRequest>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
        let native = requests
            .iter()
            .map(CallbackRequest::native)
            .collect::<Result<Vec<_>>>()?;
        let callbacks = requests
            .into_iter()
            .map(|request| request.callback)
            .collect::<Vec<_>>();
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let provider = Provider::from_checkout(&checkout, std::env::var("CUDA_HOME")?)?;
        let target =
            std::env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
        let artifact =
            provider.compile_native_with_callbacks(&native, &target, Some(&callbacks))?;
        let identities = artifact
            .specializations
            .iter()
            .map(|s| s.identity)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(identities.len(), 3);
        assert!(
            artifact
                .specializations
                .iter()
                .all(|s| !s.scratch_free && !s.mutates_payload && s.scratch.size > 0)
        );
        let builtin = provider.compile_native(&native[..1], &target)?;
        assert!(!identities.contains(&builtin.specializations[0].identity));
        let mut response = Vec::new();
        write_response(&native, artifact, &mut response, 2)?;
        assert_eq!(&response[..8], b"OXDLIB02");
        let length = u64::from_le_bytes(response[8..16].try_into()?) as usize;
        let header: serde_json::Value = serde_json::from_slice(&response[24..24 + length])?;
        assert_eq!(header["version"], 2);
        for entry in header["specializations"].as_array().unwrap() {
            assert_eq!(
                entry["arguments"],
                serde_json::json!([
                    {"kind":"const_pointer","scalar":{"kind":"signed","bits":32}},
                    {"kind":"mut_pointer","scalar":{"kind":"signed","bits":32}},
                    {"kind":"const_pointer","scalar":{"kind":"unsigned","bits":8}},
                    {"kind":"mut_pointer","scalar":{"kind":"unsigned","bits":8}}
                ])
            );
        }
        Ok(())
    }

    fn comparator_request(symbol: &str, items: u32) -> serde_json::Value {
        let mut request = callable_request(symbol, items);
        request["operation"] =
            serde_json::json!("merge_sort_keys_callable_payload_validity_bits_v1");
        request["algorithm"] = serde_json::json!("block");
        request
    }

    #[test]
    fn callable_merge_sort_is_closed_before_toolkit() {
        let mut base = comparator_request("rust_compare", 4);
        base["warp_width"] = serde_json::Value::Null;
        let parsed: CallbackRequest = serde_json::from_value(base.clone()).unwrap();
        assert!(matches!(
            parsed.native().unwrap(),
            NativeRequest::MergeSortKeys(_)
        ));
        assert!(serde_json::from_value::<SpecializationRequest>(base.clone()).is_err());
        for (path, value) in [
            (
                "/operation",
                serde_json::json!("merge_sort_keys_ascending_partial_validity_bits_v1"),
            ),
            ("/algorithm", serde_json::json!("warp")),
            ("/dtype/kind", serde_json::json!("unsigned")),
            ("/dtype/bits", serde_json::json!(64)),
            ("/warp_width", serde_json::json!(8)),
            ("/block/0", serde_json::json!(33)),
            ("/items_per_thread", serde_json::json!(0)),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let parsed: CallbackRequest = serde_json::from_value(request).unwrap();
            assert!(parsed.native().is_err(), "{path}");
        }
    }

    #[test]
    #[ignore = "requires CUDA_HOME and a Linux CUDA toolkit with NVRTC LTO support"]
    fn native_callable_merge_sort_identity_and_layout()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let requests = [
            comparator_request("rust_even_first", 1),
            comparator_request("rust_even_first", 4),
            comparator_request("rust_descending", 4),
        ]
        .into_iter()
        .map(serde_json::from_value::<CallbackRequest>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
        let native = requests
            .iter()
            .map(CallbackRequest::native)
            .collect::<Result<Vec<_>>>()?;
        let callbacks = requests
            .into_iter()
            .map(|request| request.callback)
            .collect::<Vec<_>>();
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let provider = Provider::from_checkout(&checkout, std::env::var("CUDA_HOME")?)?;
        let target =
            std::env::var("CUDA_COOP_TEST_TARGET").unwrap_or_else(|_| "compute_120".into());
        let artifact =
            provider.compile_native_with_callbacks(&native, &target, Some(&callbacks))?;
        let builtin = provider.compile_native(&native[..2], &target)?;
        let identities = artifact
            .specializations
            .iter()
            .map(|s| s.identity)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(identities.len(), 3);
        for (index, entry) in artifact.specializations.iter().enumerate() {
            let original = &builtin.specializations[index.min(1)];
            assert_ne!(entry.identity, original.identity);
            assert_eq!(entry.cub_layout, original.cub_layout);
            assert_eq!(entry.scratch, original.scratch);
            assert!(!entry.scratch_free && !entry.mutates_payload && entry.scratch.size > 0);
        }
        let mut response = Vec::new();
        write_response(&native, artifact, &mut response, 2)?;
        assert_eq!(&response[..8], b"OXDLIB02");
        let length = u64::from_le_bytes(response[8..16].try_into()?) as usize;
        let header: serde_json::Value = serde_json::from_slice(&response[24..24 + length])?;
        for entry in header["specializations"].as_array().unwrap() {
            assert_eq!(
                entry["arguments"],
                serde_json::json!([
                    {"kind":"const_pointer","scalar":{"kind":"signed","bits":32}},
                    {"kind":"mut_pointer","scalar":{"kind":"signed","bits":32}},
                    {"kind":"const_pointer","scalar":{"kind":"unsigned","bits":8}},
                    {"kind":"mut_pointer","scalar":{"kind":"unsigned","bits":8}}
                ])
            );
        }
        Ok(())
    }

    #[test]
    fn topk_mixed_abi_and_static_admission() {
        use super::*;
        let recipe = TopKRecipe {
            selection: TopKSelection::Min,
            k: TopKCount::Static(17),
            valid_items: Some(TopKCount::Static(31)),
        };
        let request = SpecializationRequest {
            operation: recipe.token(Some(Dtype::F64)),
            algorithm: "block".into(),
            dtype: Scalar::Signed(32),
            block: [64, 1, 1],
            items_per_thread: 4,
            warp_width: None,
        };
        let args = arguments(request.native().unwrap());
        assert_eq!(
            args,
            vec![
                Argument::ConstPointer(Scalar::Signed(32)),
                Argument::MutPointer(Scalar::Signed(32)),
                Argument::ConstPointer(Scalar::Float(64)),
                Argument::MutPointer(Scalar::Float(64)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
                Argument::Value(Scalar::Signed(64)),
                Argument::Value(Scalar::Signed(64))
            ]
        );
        let too_small = SpecializationRequest {
            block: [1, 1, 1],
            items_per_thread: 1,
            ..request
        };
        assert!(too_small.native().is_err());
        let full = TopKRecipe {
            selection: TopKSelection::Max,
            k: TopKCount::Runtime,
            valid_items: None,
        };
        let request = SpecializationRequest {
            operation: full.token(None),
            block: [33, 1, 1],
            items_per_thread: 9,
            ..too_small
        };
        assert_eq!(arguments(request.native().unwrap()).len(), 5);
        assert!(
            SpecializationRequest {
                warp_width: Some(32),
                ..request
            }
            .native()
            .is_err()
        );
    }

    use super::*;
    use crate::{
        ReduceAlgorithm, ReduceBatchedLayout, ReduceOperator, ScanAlgorithm, ScanOperation,
        ScanOperator,
    };

    #[test]
    fn merge_sort_pairs_registration_preserves_independent_payload_and_sentinel() {
        for value in [
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
            for recipe in [
                MergeSortPairsRecipe::AscendingFull,
                MergeSortPairsRecipe::DescendingFull,
                MergeSortPairsRecipe::AscendingPartial,
                MergeSortPairsRecipe::DescendingPartial,
            ] {
                for warp in [false, true] {
                    let encoded = if warp {
                        warp_merge_sort_pairs_registration(Path::new("/provider"), recipe, value)
                    } else {
                        merge_sort_pairs_registration(Path::new("/provider"), recipe, value)
                    }
                    .unwrap();
                    let registration: serde_json::Value = serde_json::from_str(&encoded).unwrap();
                    let mut request = SpecializationRequest {
                        operation: registration["operation"].as_str().unwrap().into(),
                        algorithm: registration["algorithm"].as_str().unwrap().into(),
                        dtype: Scalar::Float(64),
                        block: [4, 4, 4],
                        items_per_thread: 9,
                        warp_width: warp.then_some(8),
                    };
                    let native = request.native().unwrap();
                    let expected = if warp {
                        MergeSortPairsRequest::warp(Dtype::F64, value, recipe, [4, 4, 4], 8, 9)
                    } else {
                        MergeSortPairsRequest::block(Dtype::F64, value, recipe, [4, 4, 4], 9)
                    }
                    .unwrap();
                    assert_eq!(native, NativeRequest::MergeSortPairs(expected));
                    let mut abi = vec![
                        Argument::ConstPointer(Scalar::Float(64)),
                        Argument::MutPointer(Scalar::Float(64)),
                        Argument::ConstPointer(Scalar::from(value)),
                        Argument::MutPointer(Scalar::from(value)),
                        Argument::ConstPointer(Scalar::Unsigned(8)),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                        Argument::ConstPointer(Scalar::Unsigned(8)),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                    ];
                    if recipe.partial() {
                        abi.extend([
                            Argument::Value(Scalar::Signed(64)),
                            Argument::Value(Scalar::Float(64)),
                        ]);
                    }
                    assert_eq!(arguments(native), abi);
                    assert!(!native.mutates_payload() && !native.scratch_free());
                    request.warp_width = if warp { None } else { Some(8) };
                    assert!(request.native().is_err());
                }
            }
        }
        for operation in [
            "merge_sort_pairs_ascending_full_f16_validity_bits_v1",
            "merge_sort_pairs_ascending_full_f64_validity_bits_v2",
            "merge_sort_pairs_ascending_partial_validity_bits_v1",
        ] {
            assert!(MergeSortPairsRecipe::parse(operation, "block").is_err());
        }
        assert!(
            MergeSortPairsRecipe::parse(
                "merge_sort_pairs_ascending_full_f64_validity_bits_v1",
                "custom"
            )
            .is_err()
        );
        assert!(
            merge_sort_pairs_registration(
                Path::new("relative"),
                MergeSortPairsRecipe::AscendingFull,
                Dtype::F64
            )
            .is_err()
        );
        assert!(
            warp_merge_sort_pairs_registration(
                Path::new("relative"),
                MergeSortPairsRecipe::AscendingFull,
                Dtype::F64
            )
            .is_err()
        );
    }

    #[test]
    fn merge_sort_keys_registration_preserves_count_and_sentinel_types() {
        for (recipe, dtype) in [
            (MergeSortKeysRecipe::AscendingFull, Scalar::Signed(8)),
            (MergeSortKeysRecipe::DescendingFull, Scalar::Float(32)),
            (MergeSortKeysRecipe::AscendingPartial, Scalar::Unsigned(64)),
            (MergeSortKeysRecipe::DescendingPartial, Scalar::Float(64)),
        ] {
            let registration: serde_json::Value = serde_json::from_str(
                &merge_sort_keys_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype,
                block: [4, 4, 4],
                items_per_thread: 9,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::MergeSortKeys(
                    MergeSortKeysRequest::block(dtype.dtype().unwrap(), recipe, [4, 4, 4], 9)
                        .unwrap()
                )
            );
            let mut expected = vec![
                Argument::ConstPointer(dtype),
                Argument::MutPointer(dtype),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if recipe.partial() {
                expected.extend([Argument::Value(Scalar::Signed(64)), Argument::Value(dtype)]);
            }
            assert_eq!(arguments(native), expected);
            assert!(!native.mutates_payload() && !native.scratch_free());
            request.warp_width = Some(32);
            assert!(request.native().is_err());
            request.warp_width = None;
            request.block = [3, 5, 2];
            assert!(request.native().is_err());

            let warp: serde_json::Value = serde_json::from_str(
                &warp_merge_sort_keys_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            request.operation = warp["operation"].as_str().unwrap().into();
            request.algorithm = warp["algorithm"].as_str().unwrap().into();
            request.block = [4, 4, 6];
            assert!(request.native().is_err()); // No authenticated width.
            request.warp_width = Some(8);
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::MergeSortKeys(
                    MergeSortKeysRequest::warp(dtype.dtype().unwrap(), recipe, [4, 4, 6], 8, 9)
                        .unwrap()
                )
            );
            assert_eq!(arguments(native), expected);
            assert!(!native.mutates_payload() && !native.scratch_free());
            request.warp_width = Some(17);
            assert!(request.native().is_err());
        }
        assert!(
            merge_sort_keys_registration(Path::new("relative"), MergeSortKeysRecipe::AscendingFull)
                .is_err()
        );
        assert!(
            warp_merge_sort_keys_registration(
                Path::new("relative"),
                MergeSortKeysRecipe::AscendingFull
            )
            .is_err()
        );
        for (operation, algorithm) in [
            ("merge_sort_keys_ascending_full_validity_bits_v1", "other"),
            ("merge_sort_keys_ascending_full_validity_bits_v2", "block"),
            ("merge_sort_keys_custom_full_validity_bits_v1", "block"),
            ("merge_sort_pairs_ascending_full_validity_bits_v1", "block"),
        ] {
            assert!(MergeSortKeysRecipe::parse(operation, algorithm).is_err());
        }
    }

    #[test]
    fn radix_sort_pairs_registration_preserves_independent_value_abi() {
        for value in [
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
            for recipe in [
                RadixSortPairsRecipe::Ascending,
                RadixSortPairsRecipe::Descending,
            ] {
                let registration: serde_json::Value = serde_json::from_str(
                    &radix_sort_pairs_registration(Path::new("/provider"), recipe, value).unwrap(),
                )
                .unwrap();
                let mut request = SpecializationRequest {
                    operation: registration["operation"].as_str().unwrap().into(),
                    algorithm: registration["algorithm"].as_str().unwrap().into(),
                    dtype: Scalar::Signed(32),
                    block: [4, 4, 4],
                    items_per_thread: 4,
                    warp_width: None,
                };
                let native = request.native().unwrap();
                assert_eq!(
                    native,
                    NativeRequest::RadixSortPairs(
                        RadixSortPairsRequest::block(Dtype::I32, value, recipe, [4, 4, 4], 4)
                            .unwrap()
                    )
                );
                let value = Scalar::from(value);
                assert_eq!(
                    arguments(native),
                    vec![
                        Argument::ConstPointer(Scalar::Signed(32)),
                        Argument::MutPointer(Scalar::Signed(32)),
                        Argument::ConstPointer(value),
                        Argument::MutPointer(value),
                        Argument::ConstPointer(Scalar::Unsigned(8)),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                        Argument::ConstPointer(Scalar::Unsigned(8)),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                        Argument::Value(Scalar::Signed(64)),
                        Argument::Value(Scalar::Signed(64)),
                    ]
                );
                assert_eq!(native.block_shape(), [4, 4, 4]);
                assert!(!native.mutates_payload() && !native.scratch_free());
                request.warp_width = Some(32);
                assert!(request.native().is_err());
                request.warp_width = None;
                request.dtype = Scalar::Float(32);
                assert!(request.native().is_err());
                request.dtype = Scalar::Signed(64);
                request.block = [1, 1, 1];
                request.items_per_thread = 65535;
                assert!(request.native().is_ok());
                request.block = [2, 1, 1];
                assert!(request.native().is_err());
                request.items_per_thread = 4;
                request.algorithm = "warp".into();
                assert!(request.native().is_err());
                request.algorithm = "block".into();
                request.operation = "radix_sort_pairs_ascending_f16_validity_bits_v1".into();
                assert!(request.native().is_err());
            }
        }
    }

    #[test]
    fn radix_rank_registration_preserves_width_and_independent_output_abi() {
        for bits in 1..=8 {
            for recipe in [
                RadixRankKeysRecipe::Ascending(bits),
                RadixRankKeysRecipe::Descending(bits),
            ] {
                let registration: serde_json::Value = serde_json::from_str(
                    &radix_rank_keys_registration(Path::new("/provider"), recipe).unwrap(),
                )
                .unwrap();
                let mut request = SpecializationRequest {
                    operation: registration["operation"].as_str().unwrap().into(),
                    algorithm: registration["algorithm"].as_str().unwrap().into(),
                    dtype: Scalar::Signed(64),
                    block: [4, 4, 4],
                    items_per_thread: 9,
                    warp_width: None,
                };
                let native = request.native().unwrap();
                assert_eq!(
                    native,
                    NativeRequest::RadixRankKeys(
                        RadixRankKeysRequest::block(Dtype::I64, recipe, [4, 4, 4], 9).unwrap()
                    )
                );
                assert_eq!(
                    arguments(native),
                    vec![
                        Argument::ConstPointer(Scalar::Signed(64)),
                        Argument::MutPointer(Scalar::Signed(32)),
                        Argument::ConstPointer(Scalar::Unsigned(8)),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                        Argument::Value(Scalar::Signed(64)),
                        Argument::Value(Scalar::Signed(64)),
                    ]
                );
                assert!(!native.mutates_payload() && !native.scratch_free());
                request.warp_width = Some(32);
                assert!(request.native().is_err());
                request.warp_width = None;
                request.dtype = Scalar::Float(64);
                assert!(request.native().is_err());
            }
        }
        for bits in [0, 9, u32::MAX] {
            for recipe in [
                RadixRankKeysRecipe::Ascending(bits),
                RadixRankKeysRecipe::Descending(bits),
            ] {
                assert!(radix_rank_keys_registration(Path::new("/provider"), recipe).is_err());
                assert!(RadixRankKeysRequest::block(Dtype::I32, recipe, [32, 1, 1], 1).is_err());
            }
        }
        assert!(
            radix_rank_keys_registration(Path::new("relative"), RadixRankKeysRecipe::Ascending(4))
                .is_err()
        );
        for (operation, algorithm) in [
            ("radix_rank_keys_ascending_0_validity_bits_v1", "block"),
            ("radix_rank_keys_descending_9_validity_bits_v1", "block"),
            ("radix_rank_keys_ascending_04_validity_bits_v1", "block"),
            ("radix_rank_keys_ascending_4_validity_bits_v2", "block"),
            ("radix_rank_keys_ascending_4_validity_bits_v1", "warp"),
        ] {
            assert!(RadixRankKeysRecipe::parse(operation, algorithm).is_err());
        }
    }

    #[test]
    fn radix_sort_keys_registration_preserves_wide_controls_and_scope() {
        for (recipe, dtype) in [
            (RadixSortKeysRecipe::Ascending, Scalar::Signed(32)),
            (RadixSortKeysRecipe::Descending, Scalar::Unsigned(32)),
            (RadixSortKeysRecipe::Ascending, Scalar::Signed(64)),
            (RadixSortKeysRecipe::Descending, Scalar::Unsigned(64)),
        ] {
            let registration: serde_json::Value = serde_json::from_str(
                &radix_sort_keys_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype,
                block: [4, 4, 4],
                items_per_thread: 9,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::RadixSortKeys(
                    RadixSortKeysRequest::block(dtype.dtype().unwrap(), recipe, [4, 4, 4], 9)
                        .unwrap()
                )
            );
            assert_eq!(
                arguments(native),
                vec![
                    Argument::ConstPointer(dtype),
                    Argument::MutPointer(dtype),
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::Unsigned(8)),
                    Argument::Value(Scalar::Signed(64)),
                    Argument::Value(Scalar::Signed(64)),
                ]
            );
            assert!(!native.mutates_payload() && !native.scratch_free());
            request.warp_width = Some(32);
            assert!(request.native().is_err());
            request.warp_width = None;
            request.dtype = Scalar::Float(64);
            assert!(request.native().is_err());
        }
        assert!(
            radix_sort_keys_registration(Path::new("relative"), RadixSortKeysRecipe::Ascending)
                .is_err()
        );
        assert!(
            RadixSortKeysRecipe::parse("radix_sort_keys_ascending_validity_bits_v1", "warp")
                .is_err()
        );
        assert!(
            RadixSortKeysRecipe::parse("radix_sort_keys_ascending_validity_bits_v2", "block")
                .is_err()
        );
        assert!(
            RadixSortKeysRecipe::parse("radix_sort_pairs_ascending_validity_bits_v1", "block")
                .is_err()
        );
    }

    #[test]
    fn histogram_template_decoding_and_independent_counter_abi() {
        for (algorithm, counter, bins, output_items) in [
            (HistogramAlgorithm::Sort, Dtype::U64, 65, 2),
            (HistogramAlgorithm::Atomic, Dtype::I32, 1, 9),
            (HistogramAlgorithm::Sort, Dtype::I64, 33, 1),
            (HistogramAlgorithm::Atomic, Dtype::U32, 65, 3),
        ] {
            let template =
                histogram_registration_template(Path::new("/provider"), algorithm).unwrap();
            let mut registration: serde_json::Value = serde_json::from_str(&template).unwrap();
            assert_eq!(
                registration["operation"],
                crate::histogram::TEMPLATE_OPERATION
            );
            let recipe = HistogramRecipe::new(algorithm, counter, bins, output_items).unwrap();
            registration["operation"] = recipe.token().into();
            let request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: Scalar::Unsigned(8),
                block: [64, 1, 1],
                items_per_thread: 4,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::Histogram(
                    HistogramRequest::block(Dtype::U8, recipe, [64, 1, 1], 4).unwrap()
                )
            );
            assert_eq!(
                arguments(native),
                vec![
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::from(counter)),
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::Unsigned(8)),
                ]
            );
            assert!(
                SpecializationRequest {
                    warp_width: Some(32),
                    ..request
                }
                .native()
                .is_err()
            );
        }
        for operation in [
            crate::histogram::TEMPLATE_OPERATION,
            "histogram_u64_bins65_k0000000002_validity_bits_v1",
            "histogram_u64_bins0000000065_k0000000002_validity_bits_v2",
            "histogram_f64_bins0000000065_k0000000002_validity_bits_v1",
            "histogram_u64_bins0000000000_k0000000002_validity_bits_v1",
            "histogram_u64_bins2147483648_k0000000002_validity_bits_v1",
            "histogram_u64_bins9999999999_k0000000002_validity_bits_v1",
            "histogram_u64_bins0000000065_k+000000002_validity_bits_v1",
            "histogram_u64_bins0000000065_k0000000000_validity_bits_v1",
        ] {
            assert!(HistogramRecipe::parse(operation, "sort").is_err());
        }
        let valid = "histogram_u64_bins0000000065_k0000000002_validity_bits_v1";
        assert!(HistogramRecipe::parse(valid, "block").is_err());
        assert!(
            histogram_registration_template(Path::new("relative"), HistogramAlgorithm::Atomic)
                .is_err()
        );
    }

    #[test]
    fn discontinuity_registration_preserves_mixed_output_abi() {
        use DiscontinuityRecipe::*;
        for (recipe, paired, neighbors) in [
            (Heads, false, 0),
            (HeadsPredecessor, false, 1),
            (Tails, false, 0),
            (TailsSuccessor, false, 1),
            (HeadsAndTails, true, 0),
            (HeadsAndTailsPredecessor, true, 1),
            (HeadsAndTailsSuccessor, true, 1),
            (HeadsAndTailsBoth, true, 2),
        ] {
            let registration: serde_json::Value = serde_json::from_str(
                &discontinuity_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: Scalar::Float(64),
                block: [4, 3, 2],
                items_per_thread: 4,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::Discontinuity(
                    DiscontinuityRequest::block(Dtype::F64, recipe, [4, 3, 2], 4).unwrap()
                )
            );
            let mut expected = vec![
                Argument::ConstPointer(Scalar::Float(64)),
                Argument::MutPointer(Scalar::Signed(32)),
            ];
            if paired {
                expected.push(Argument::MutPointer(Scalar::Signed(32)));
            }
            expected.extend([
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ]);
            if paired {
                expected.push(Argument::MutPointer(Scalar::Unsigned(8)));
            }
            expected.extend((0..neighbors).map(|_| Argument::Value(Scalar::Float(64))));
            assert_eq!(arguments(native), expected);
            assert!(!native.scratch_free() && !native.mutates_payload());
            assert_eq!(registration["version"], 1);
            assert_eq!(registration["provider"], "cuda_coop");
            request.warp_width = Some(32);
            assert!(request.native().is_err());
        }
        assert!(discontinuity_registration(Path::new("relative"), Heads).is_err());
        assert!(
            DiscontinuityRecipe::parse("discontinuity_heads_successor_validity_bits_v1", "block")
                .is_err()
        );
        assert!(
            DiscontinuityRecipe::parse("discontinuity_tails_predecessor_validity_bits_v1", "block")
                .is_err()
        );
        assert!(DiscontinuityRecipe::parse(Heads.token(), "warp").is_err());
        assert!(DiscontinuityRecipe::parse(Heads.token(), "sort").is_err());
    }

    #[test]
    fn adjacent_difference_registration_preserves_typed_controls() {
        use AdjacentDifferenceRecipe::*;
        for (recipe, controls) in [
            (LeftFull, vec![]),
            (LeftFullPredecessor, vec![Scalar::Float(64)]),
            (LeftPartial, vec![Scalar::Signed(32)]),
            (
                LeftPartialPredecessor,
                vec![Scalar::Signed(32), Scalar::Float(64)],
            ),
            (RightFull, vec![]),
            (RightFullSuccessor, vec![Scalar::Float(64)]),
            (RightPartial, vec![Scalar::Signed(32)]),
        ] {
            let registration: serde_json::Value = serde_json::from_str(
                &adjacent_difference_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: Scalar::Float(64),
                block: [4, 3, 2],
                items_per_thread: 9,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::AdjacentDifference(
                    AdjacentDifferenceRequest::block(Dtype::F64, recipe, [4, 3, 2], 9).unwrap()
                )
            );
            let mut expected = vec![
                Argument::ConstPointer(Scalar::Float(64)),
                Argument::MutPointer(Scalar::Float(64)),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            expected.extend(controls.into_iter().map(Argument::Value));
            assert_eq!(arguments(native), expected);
            assert!(!native.scratch_free() && !native.mutates_payload());
            assert_eq!(registration["version"], 1);
            assert_eq!(registration["provider"], "cuda_coop");
            request.warp_width = Some(32);
            assert!(request.native().is_err());
        }
        assert!(adjacent_difference_registration(Path::new("relative"), LeftFull).is_err());
        assert!(
            AdjacentDifferenceRecipe::parse(
                "adjacent_difference_right_partial_successor_validity_bits_v1",
                "block"
            )
            .is_err()
        );
        assert!(AdjacentDifferenceRecipe::parse(LeftFull.token(), "warp").is_err());
        assert!(AdjacentDifferenceRecipe::parse(RightFull.token(), "transpose").is_err());
    }

    #[test]
    fn shuffle_registration_preserves_scope_extent_and_pointer_abi() {
        for recipe in [ShuffleRecipe::Down, ShuffleRecipe::Up] {
            let registration: serde_json::Value = serde_json::from_str(
                &shuffle_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: Scalar::Float(64),
                block: [4, 3, 2],
                items_per_thread: 9,
                warp_width: None,
            };
            let native = request.native().unwrap();
            assert_eq!(
                native,
                NativeRequest::Shuffle(
                    ShuffleRequest::block(Dtype::F64, recipe, [4, 3, 2], 9).unwrap()
                )
            );
            assert_eq!(
                arguments(native),
                vec![
                    Argument::ConstPointer(Scalar::Float(64)),
                    Argument::MutPointer(Scalar::Float(64)),
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::Unsigned(8)),
                ]
            );
            assert!(!native.scratch_free() && !native.mutates_payload());
            assert_eq!(registration["version"], 1);
            assert_eq!(registration["provider"], "cuda_coop");
            request.warp_width = Some(32);
            assert!(request.native().is_err());
        }
        assert!(shuffle_registration(Path::new("relative"), ShuffleRecipe::Down).is_err());
        assert!(ShuffleRecipe::parse("shuffle_rotate", "block").is_err());
        assert!(ShuffleRecipe::parse(ShuffleRecipe::Down.token(), "warp").is_err());
        assert!(ShuffleRecipe::parse(ShuffleRecipe::Up.token(), "transpose").is_err());
    }

    #[test]
    fn exchange_registration_preserves_scope_extent_and_pointer_abi() {
        use crate::ExchangeMode::{BlockedToStriped, StripedToBlocked};
        for (recipe, width) in [
            (ExchangeRecipe::Block(StripedToBlocked), None),
            (ExchangeRecipe::Block(BlockedToStriped), None),
            (ExchangeRecipe::Warp(StripedToBlocked), Some(8)),
            (ExchangeRecipe::Warp(BlockedToStriped), Some(8)),
        ] {
            let registration: serde_json::Value = serde_json::from_str(
                &exchange_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let mut request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: Scalar::Float(64),
                block: [8, 4, 2],
                items_per_thread: 9,
                warp_width: width,
            };
            let native = request.native().unwrap();
            let expected = match width {
                None => ExchangeRequest::block(Dtype::F64, recipe, [8, 4, 2], 9).unwrap(),
                Some(width) => {
                    ExchangeRequest::warp(Dtype::F64, recipe, [8, 4, 2], width, 9).unwrap()
                }
            };
            assert_eq!(native, NativeRequest::Exchange(expected));
            assert_eq!(
                arguments(native),
                vec![
                    Argument::ConstPointer(Scalar::Float(64)),
                    Argument::MutPointer(Scalar::Float(64)),
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::Unsigned(8)),
                ]
            );
            assert!(!native.scratch_free() && !native.mutates_payload());
            assert_eq!(registration["version"], 1);
            assert_eq!(registration["provider"], "cuda_coop");
            request.warp_width = if width.is_some() { None } else { Some(8) };
            assert!(request.native().is_err());
        }
        assert!(
            exchange_registration(
                Path::new("relative"),
                ExchangeRecipe::Block(StripedToBlocked)
            )
            .is_err()
        );
        assert!(ExchangeRecipe::parse("exchange_ranked", "block").is_err());
        assert!(
            ExchangeRecipe::parse(ExchangeRecipe::Block(StripedToBlocked).token(), "shuffle")
                .is_err()
        );
    }

    #[test]
    fn batched_registration_preserves_derived_extent_and_pointer_abi() {
        for (dtype, op, layout, width, batches, outputs) in [
            (
                Dtype::I8,
                ReduceOperator::Sum,
                ReduceBatchedLayout::Striped,
                32,
                1,
                1,
            ),
            (
                Dtype::U16,
                ReduceOperator::BitAnd,
                ReduceBatchedLayout::Blocked,
                8,
                10,
                2,
            ),
            (
                Dtype::F64,
                ReduceOperator::Max,
                ReduceBatchedLayout::Striped,
                1,
                9,
                9,
            ),
        ] {
            let recipe = ReduceBatchedRecipe::new(op, layout);
            let registration: serde_json::Value = serde_json::from_str(
                &reduce_batched_registration(Path::new("/provider"), recipe).unwrap(),
            )
            .unwrap();
            let scalar = Scalar::from(dtype);
            let request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: scalar,
                block: [4, 4, 4],
                items_per_thread: batches,
                warp_width: Some(width),
            }
            .native()
            .unwrap();
            let NativeRequest::ReduceBatched(native) = request else {
                panic!("batched recipe decoded as another family");
            };
            assert_eq!(native.outputs_per_thread(), outputs);
            assert_eq!(native.recipe, recipe);
            assert!(!request.scratch_free());
            assert!(!request.mutates_payload());
            assert_eq!(
                arguments(request),
                vec![
                    Argument::ConstPointer(scalar),
                    Argument::MutPointer(scalar),
                    Argument::ConstPointer(Scalar::Unsigned(8)),
                    Argument::MutPointer(Scalar::Unsigned(8)),
                ]
            );
            assert_eq!(registration["provider"], "cuda_coop");
            assert_eq!(registration["version"], 1);
        }
        let recipe = ReduceBatchedRecipe::new(ReduceOperator::Sum, ReduceBatchedLayout::Striped);
        assert!(reduce_batched_registration(Path::new("relative"), recipe).is_err());
        let last_padded = (i32::MAX as u32 / 32) * 32;
        assert_eq!(
            ReduceBatchedRequest::warp(Dtype::I32, recipe, [32, 1, 1], 32, last_padded)
                .unwrap()
                .outputs_per_thread(),
            last_padded / 32,
        );
        assert!(
            ReduceBatchedRequest::warp(Dtype::I32, recipe, [32, 1, 1], 32, last_padded + 1,)
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_batched_reduce_before_opening_toolkit() {
        let base = serde_json::json!({"version":1,"target":"compute_120","cuda_root":"/absent", "requests":[{
            "operation":"reduce_batched_bit_xor_blocked_validity_bits_v1","algorithm":"warp",
            "dtype":{"kind":"signed","bits":32},"block":[4,4,4],"items_per_thread":10,"warp_width":8
        }]});
        for (path, value, diagnostic) in [
            (
                "/requests/0/dtype",
                serde_json::json!({"kind":"float","bits":32}),
                "integer",
            ),
            (
                "/requests/0/algorithm",
                serde_json::json!("raking"),
                "Warp algorithm",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("reduce_batched_sum_payload_validity_bits_v1"),
                "output layout",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("reduce_batched_unknown_striped_validity_bits_v1"),
                "operator",
            ),
            (
                "/requests/0/items_per_thread",
                serde_json::json!(0),
                "positive batches",
            ),
            (
                "/requests/0/items_per_thread",
                serde_json::json!(u32::MAX),
                "padded extent",
            ),
            (
                "/requests/0/warp_width",
                serde_json::json!(null),
                "Warp scope",
            ),
            (
                "/requests/0/warp_width",
                serde_json::json!(17),
                "power-of-two",
            ),
            (
                "/requests/0/warp_width",
                serde_json::json!(0),
                "power-of-two",
            ),
            (
                "/requests/0/block",
                serde_json::json!([48, 1, 1]),
                "complete physical warps",
            ),
            (
                "/requests/0/block",
                serde_json::json!([u32::MAX, 4, 4]),
                "positive axes",
            ),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let error = serve(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn reduce_registration_preserves_scalar_count_and_payload_abi() {
        for (dtype, recipe, items) in [
            (
                Dtype::I8,
                ReduceRecipe::Block(
                    ReduceOperator::Sum,
                    ReduceAlgorithm::WarpReductions,
                    ReduceForm::Payload,
                ),
                1,
            ),
            (
                Dtype::U16,
                ReduceRecipe::Block(
                    ReduceOperator::Min,
                    ReduceAlgorithm::Raking,
                    ReduceForm::ScalarPrefix,
                ),
                1,
            ),
            (
                Dtype::F64,
                ReduceRecipe::Warp(ReduceOperator::Max, ReduceForm::Scalar),
                1,
            ),
            (
                Dtype::I32,
                ReduceRecipe::Warp(ReduceOperator::BitXor, ReduceForm::Payload),
                9,
            ),
        ] {
            let registration: serde_json::Value =
                serde_json::from_str(&reduce_registration(Path::new("/provider"), recipe).unwrap())
                    .unwrap();
            let scalar = Scalar::from(dtype);
            let request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: scalar,
                block: [4, 4, 4],
                items_per_thread: items,
                warp_width: matches!(recipe, ReduceRecipe::Warp(..)).then_some(17),
            }
            .native()
            .unwrap();
            let mut expected = vec![
                Argument::ConstPointer(scalar),
                Argument::MutPointer(scalar),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if recipe.form() == ReduceForm::ScalarPrefix {
                expected.push(Argument::Value(Scalar::Signed(32)));
            }
            assert_eq!(arguments(request), expected);
            assert_eq!(registration["provider"], "cuda_coop");
        }
    }

    #[test]
    fn rejects_invalid_reduce_before_opening_toolkit() {
        let base = serde_json::json!({"version":1,"target":"compute_120","cuda_root":"/absent", "requests":[{
            "operation":"reduce_bit_xor_payload_validity_bits_v1","algorithm":"raking",
            "dtype":{"kind":"signed","bits":32},"block":[8,4,2],"items_per_thread":9,"warp_width":null
        }]});
        for (path, value, diagnostic) in [
            (
                "/requests/0/dtype",
                serde_json::json!({"kind":"float","bits":32}),
                "integer",
            ),
            (
                "/requests/0/algorithm",
                serde_json::json!("raking_memoize"),
                "Reduce algorithm",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("reduce_sum_prefix_validity_bits_v1"),
                "exactly one",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("reduce_sum_payload_prefix_validity_bits_v1"),
                "Reduce operator",
            ),
            (
                "/requests/0/items_per_thread",
                serde_json::json!(0),
                "positive items",
            ),
            (
                "/requests/0/warp_width",
                serde_json::json!(17),
                "Warp scope",
            ),
            (
                "/requests/0/block",
                serde_json::json!([u32::MAX, 4, 4]),
                "positive axes",
            ),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let error = serve(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
        for (block, width, diagnostic) in [
            ([48, 1, 1], 17, "complete physical warps"),
            ([64, 1, 1], 15, "guarded prefix"),
        ] {
            let mut request = base.clone();
            request["requests"][0]["algorithm"] = "warp".into();
            request["requests"][0]["block"] = serde_json::json!(block);
            request["requests"][0]["warp_width"] = width.into();
            let error = serve(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn scan_registration_preserves_typed_seed_and_empty_controls() {
        for (dtype, recipe) in [
            (
                Dtype::I8,
                ScanRecipe::Block(
                    ScanOperation::Exclusive(ScanOperator::Min),
                    ScanAlgorithm::Raking,
                ),
            ),
            (
                Dtype::U16,
                ScanRecipe::Warp(ScanOperation::Exclusive(ScanOperator::Sum)),
            ),
            (
                Dtype::F64,
                ScanRecipe::Warp(ScanOperation::Inclusive(ScanOperator::Max)),
            ),
            (
                Dtype::I32,
                ScanRecipe::Block(ScanOperation::ExclusiveSum, ScanAlgorithm::WarpScans),
            ),
        ] {
            let registration: serde_json::Value =
                serde_json::from_str(&scan_registration(Path::new("/provider"), recipe).unwrap())
                    .unwrap();
            let scalar = Scalar::from(dtype);
            let request = SpecializationRequest {
                operation: registration["operation"].as_str().unwrap().into(),
                algorithm: registration["algorithm"].as_str().unwrap().into(),
                dtype: scalar,
                block: [4, 4, 4],
                items_per_thread: 1,
                warp_width: matches!(recipe, ScanRecipe::Warp(_)).then_some(8),
            }
            .native()
            .unwrap();
            let mut expected = vec![
                Argument::ConstPointer(scalar),
                Argument::MutPointer(scalar),
                Argument::ConstPointer(Scalar::Unsigned(8)),
                Argument::MutPointer(Scalar::Unsigned(8)),
            ];
            if recipe.operation().seeded() {
                expected.push(Argument::Value(scalar));
            }
            assert_eq!(arguments(request), expected);
            assert_eq!(registration["provider"], "cuda_coop");
        }
    }

    #[test]
    fn rejects_invalid_scan_before_opening_toolkit() {
        let base = serde_json::json!({"version":1,"target":"compute_120","cuda_root":"/absent", "requests":[{
            "operation":"scan_inclusive_bit_xor_validity_bits_v1","algorithm":"raking",
            "dtype":{"kind":"signed","bits":32},"block":[8,4,2],"items_per_thread":9,"warp_width":null
        }]});
        for (path, value, diagnostic) in [
            (
                "/requests/0/dtype",
                serde_json::json!({"kind":"float","bits":32}),
                "integer",
            ),
            (
                "/requests/0/algorithm",
                serde_json::json!("transpose"),
                "Scan algorithm",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("scan_exclusive_min_validity_bits_v1"),
                "seed form",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("scan_inclusive_sum_seeded_validity_bits_v1"),
                "Scan operator",
            ),
            (
                "/requests/0/items_per_thread",
                serde_json::json!(0),
                "positive items",
            ),
            (
                "/requests/0/warp_width",
                serde_json::json!(8),
                "scope or scalar extent",
            ),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let error = serve(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
        for (block, items, width, diagnostic) in [
            ([48, 1, 1], 1, 8, "complete physical warps"),
            ([64, 1, 1], 4, 8, "scalar extent"),
            ([64, 1, 1], 1, 3, "power of two"),
        ] {
            let mut request = base.clone();
            request["requests"][0]["algorithm"] = "warp".into();
            request["requests"][0]["block"] = serde_json::json!(block);
            request["requests"][0]["items_per_thread"] = items.into();
            request["requests"][0]["warp_width"] = width.into();
            let error = serve(
                Path::new("/absent"),
                request.to_string().as_bytes(),
                Vec::new(),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
        assert!(
            ScanRequest::block(
                Dtype::I32,
                ScanOperation::ExclusiveSum,
                ScanAlgorithm::WarpScans,
                [8, 3, 2],
                1
            )
            .is_err()
        );
    }

    #[test]
    fn typed_validity_abi_preserves_width_and_separate_flags() {
        // Old byte-per-slot registrations must not bind to the packed format.
        assert!(
            SpecializationRequest {
                operation: "load_filled_validity_v1".into(),
                algorithm: "direct".into(),
                dtype: Scalar::Signed(32),
                block: [128, 1, 1],
                items_per_thread: 4,
                warp_width: None,
            }
            .native()
            .is_err()
        );
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
            let scalar = Scalar::from(dtype);
            assert_eq!(scalar.dtype().unwrap(), dtype);
            for algorithm in [
                "direct",
                "striped",
                "vectorize",
                "transpose",
                "warp_transpose",
                "warp_transpose_timesliced",
            ] {
                let request = SpecializationRequest {
                    operation: "load_filled_validity_bits_v1".into(),
                    algorithm: algorithm.into(),
                    dtype: scalar,
                    block: [128, 1, 1],
                    items_per_thread: 4,
                    warp_width: None,
                }
                .native()
                .unwrap();
                assert_eq!(
                    arguments(request),
                    vec![
                        Argument::ConstPointer(scalar),
                        Argument::MutPointer(scalar),
                        Argument::MutPointer(Scalar::Unsigned(8)),
                        Argument::Value(Scalar::Signed(32)),
                        Argument::Value(scalar),
                    ]
                );
            }
        }
        for operation in [
            Operation::StoreFullValidity,
            Operation::StorePartialValidity,
        ] {
            for algorithm in [
                Algorithm::Direct,
                Algorithm::Striped,
                Algorithm::Vectorize,
                Algorithm::Transpose,
                Algorithm::WarpTranspose,
                Algorithm::WarpTransposeTimesliced,
            ] {
                let request = Request::new_typed(Dtype::I16, operation, algorithm, 128, 4).unwrap();
                let mut expected = vec![
                    Argument::MutPointer(Scalar::Signed(16)),
                    if algorithm.scratch_free() {
                        Argument::ConstPointer(Scalar::Signed(16))
                    } else {
                        Argument::MutPointer(Scalar::Signed(16))
                    },
                    if algorithm.scratch_free() {
                        Argument::ConstPointer(Scalar::Unsigned(8))
                    } else {
                        Argument::MutPointer(Scalar::Unsigned(8))
                    },
                ];
                if operation.counted() {
                    expected.push(Argument::Value(Scalar::Signed(32)));
                }
                assert_eq!(arguments(NativeRequest::LoadStore(request)), expected);
            }
        }
        for scalar in [Scalar::Signed(1), Scalar::Unsigned(128), Scalar::Float(16)] {
            assert!(scalar.dtype().is_err());
        }
    }

    #[test]
    fn input_query_needs_no_installed_toolkit() {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"));
        let toolkit = std::env::temp_dir().join("cuda-coop-input-query-no-toolkit");
        let request = serde_json::json!({"version":1,"cuda_root":toolkit,"target":"compute_120"});
        let mut bytes = Vec::new();
        serve_inputs(checkout, request.to_string().as_bytes(), &mut bytes).unwrap();
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["version"], 1);
        assert_eq!(response["paths"].as_array().unwrap().len(), 6);
        assert!(
            response["paths"]
                .as_array()
                .unwrap()
                .iter()
                .all(|path| Path::new(path.as_str().unwrap()).is_absolute())
        );
        assert!(response["environment"].as_array().unwrap().is_empty());
        for request in [
            serde_json::json!({"version":2,"cuda_root":toolkit,"target":"compute_120"}),
            serde_json::json!({"version":1,"cuda_root":"relative","target":"compute_120"}),
            serde_json::json!({"version":1,"cuda_root":toolkit,"target":"invalid"}),
        ] {
            assert!(serve_inputs(checkout, request.to_string().as_bytes(), Vec::new()).is_err());
        }
        assert!(serve_inputs(checkout, vec![b' '; MAX_HEADER + 1].as_slice(), Vec::new()).is_err());
    }

    #[test]
    fn registration_binds_native_recipe() {
        let value: serde_json::Value = serde_json::from_str(
            &registration(
                Path::new("/provider"),
                Operation::LoadFilled,
                Algorithm::Transpose,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["operation"], "load_filled");
        assert_eq!(value["algorithm"], "transpose");
        assert!(
            registration(
                Path::new("relative"),
                Operation::LoadFull,
                Algorithm::Direct
            )
            .is_err()
        );
    }

    #[test]
    fn preserves_multidimensional_raw_and_typed_requests() {
        for (operation, dtype) in [
            ("load_full", Scalar::Signed(32)),
            ("store_partial_validity_bits_v1", Scalar::Unsigned(64)),
        ] {
            let request = SpecializationRequest {
                operation: operation.into(),
                algorithm: "transpose".into(),
                dtype,
                block: [8, 4, 4],
                items_per_thread: 4,
                warp_width: None,
            };
            let NativeRequest::LoadStore(native) = request.native().unwrap() else {
                panic!("expected Load/Store")
            };
            assert_eq!(native.block_shape(), [8, 4, 4]);
            assert_eq!(native.block_threads(), 128);
            assert_eq!(native.dtype(), request.dtype.dtype().unwrap());
        }
    }

    #[test]
    fn rejects_request_before_opening_toolkit() {
        let base = serde_json::json!({"version":1,"target":"compute_120","cuda_root":"/absent", "requests":[{
            "operation":"load_full","algorithm":"direct","dtype":{"kind":"signed","bits":32},"block":[128,1,1],"items_per_thread":4
        }]});
        for (path, value, diagnostic) in [
            (
                "/version",
                serde_json::json!(2),
                "unsupported provider protocol",
            ),
            (
                "/requests/0/dtype/bits",
                serde_json::json!(64),
                "signed i32",
            ),
            ("/requests/0/block/1", serde_json::json!(0), "block threads"),
            (
                "/requests/0/operation",
                serde_json::json!("unknown"),
                "unsupported operation",
            ),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let mut output = Vec::new();
            let error = serve(
                Path::new("/absent"),
                serde_json::to_vec(&request).unwrap().as_slice(),
                &mut output,
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
            assert!(output.is_empty());
        }
    }
}
