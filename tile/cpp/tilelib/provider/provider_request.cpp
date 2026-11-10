// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/provider_request.h"

#include <climits>
#include <string>
#include <utility>
#include <variant>

#include "provider/families/emitter.h"
#include "provider/registry.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
MetadataResult malformed(FamilyId family, const char* message)
{
  return MetadataResult::malformed(std::string("[LP-TOPOLOGY-001] ") + family_name(family) + ": " + message);
}

MetadataResult unsupported(FamilyId family, const char* message)
{
  return MetadataResult::unsupported(std::string("[LP-UNSUPPORTED-001] ") + family_name(family) + ": " + message);
}
} // namespace

MetadataResult validate_logical_extents(FamilyId family, int64_t extent, int64_t decoded_extent)
{
  const bool rld = family == FamilyId::run_length_decode || family == FamilyId::run_length_decode_with_offsets;
  if (extent <= 0 || (rld && decoded_extent <= 0))
  {
    return malformed(family, rld ? "RunN and DecodedN must be positive" : "N must be positive");
  }
  return MetadataResult::success();
}

MetadataResult make_logical_request(
  FamilyId family, PrimitiveSpec primitive, int64_t extent, Target target, std::optional<LogicalRequest>* out)
{
  if (!out)
  {
    return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] null logical request destination");
  }
  out->reset();
  if (family_id(primitive) != family || !target.sm_arch.valid())
  {
    return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] invalid logical request family or target");
  }
  if (const auto* const reduce = std::get_if<ReduceSpec>(&primitive))
  {
    if (reduce->batches < 0 || (reduce->batches > 0 && extent % reduce->batches != 0))
    {
      return malformed(family, "Batches must be positive and divide N");
    }
  }
  const auto* const rld = std::get_if<RunLengthDecodeSpec>(&primitive);
  const auto topology   = validate_logical_extents(family, extent, rld ? rld->decoded_extent : 0);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (const auto* const exchange = std::get_if<ExchangeSpec>(&primitive))
  {
    if (exchange->block_threads <= 0 || extent % exchange->block_threads != 0)
    {
      return malformed(family, "Exchange requires a positive BlockThreads width that divides N");
    }
    if (exchange->block_threads > 1024 || exchange->block_threads % 32 != 0)
    {
      return malformed(family, "Exchange BlockThreads must be a multiple of 32 in [32, 1024]");
    }
  }
  *out = LogicalRequest{std::move(primitive), extent, target};
  return MetadataResult::success();
}

// Derive per-thread extents and validate a candidate without compiling it.
// Exchange retains its semantic width; Run Length Decode must divide both
// logical extents. Family source emission is a final consistency check on the
// selected options. Publish the immutable result only after those checks pass;
// NVRTC and resource queries belong to later phases.
MetadataResult
select_implementation(const LogicalRequest& demand, int64_t execution_threads, std::optional<ImplementationSpec>* out)
{
  if (!out)
  {
    return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] null implementation destination");
  }
  out->reset();
  const auto family          = family_id(demand.primitive());
  const auto* const exchange = std::get_if<ExchangeSpec>(&demand.primitive());
  const auto* const reduce   = std::get_if<ReduceSpec>(&demand.primitive());
  const bool batched_reduce  = reduce && reduce->batches > 0;
  const bool warp_reduce     = batched_reduce && reduce->algorithm == "warp_reductions";
  const int64_t threads      = warp_reduce ? 32 : (exchange ? exchange->block_threads : execution_threads);
  if (threads <= 0)
  {
    return unsupported(family, "automatic topology has no valid block-thread candidate");
  }
  const int64_t reduction_extent = batched_reduce ? demand.extent() / reduce->batches : demand.extent();
  const int64_t valid_threads    = batched_reduce && reduction_extent < threads ? reduction_extent : threads;
  if (valid_threads <= 0 || reduction_extent % valid_threads != 0 || (batched_reduce && threads % valid_threads != 0))
  {
    return unsupported(family, "automatic topology cannot divide N by the selected block size");
  }
  const int64_t items = reduction_extent / valid_threads;
  if (threads > 1024 || items > INT32_MAX || threads % 32 != 0)
  {
    return unsupported(family, "topology is outside the provider capability limits");
  }
  Topology topology{static_cast<int>(threads), static_cast<int>(items), 0};
  if (warp_reduce && reduce->dtype == "f64" && reduce->batches % 2 == 0 && reduction_extent >= 2
      && reduction_extent < 32 && (reduction_extent & (reduction_extent - 1)) == 0)
  {
    // Short FP64 rows share one CUB batched warp reduction at their logical
    // width. Other types and longer rows keep independent invocations so
    // additional worker warps can process rows concurrently.
    topology.reduction_batches_per_invocation = 2;
  }
  if (const auto* const rld = std::get_if<RunLengthDecodeSpec>(&demand.primitive()))
  {
    if (rld->decoded_extent % threads != 0)
    {
      return unsupported(family, "DecodedN must be divisible by BlockThreads");
    }
    const int64_t decoded_items = rld->decoded_extent / threads;
    if (decoded_items <= 0 || decoded_items > INT32_MAX)
    {
      return unsupported(family, "DecodedItemsPerThread is outside the provider capability limits");
    }
    topology.decoded_items_per_thread = static_cast<int>(decoded_items);
  }
  if (const auto* const histogram = std::get_if<HistogramSpec>(&demand.primitive()))
  {
    if (histogram->bins % threads != 0)
    {
      return unsupported(family, "Bins must be divisible by BlockThreads");
    }
  }
  if (const auto* const shuffle = std::get_if<ShuffleSpec>(&demand.primitive()))
  {
    if (shuffle->mode != "up" && shuffle->mode != "down" && items != 1)
    {
      return unsupported(family, "offset and rotate require ItemsPerThread == 1");
    }
  }
  if (std::holds_alternative<MergeSortSpec>(demand.primitive()) && !merge_sort_block_threads_are_supported(threads))
  {
    return unsupported(family, "BlockThreads must be a power of two in [32, 1024]");
  }

  ImplementationSpec implementation{demand, topology};
  std::string source;
  std::string error;
  const auto status =
    emit_family_source(SourceEmissionRequest{implementation, family_descriptor(family).impl_symbol}, &source, &error);
  if (status != SourceEmissionStatus::emitted || source.empty())
  {
    return MetadataResult::internal_error(
      std::string("[LP-PROVIDER-INTERNAL-001] ") + family_name(family)
      + ": source validation failed: " + (error.empty() ? "source emitter returned no diagnostic" : error));
  }
  *out = std::move(implementation);
  return MetadataResult::success();
}
} // namespace cuda::coop::tile::detail::provider
