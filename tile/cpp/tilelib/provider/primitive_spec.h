// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace cuda::coop::tile::detail::provider
{
enum class FamilyId
{
  sum,
  reduce,
  scan,
  histogram,
  histogram_256,
  run_length_decode,
  run_length_decode_with_offsets,
  discontinuity_flags,
  discontinuity_heads_and_tails,
  adjacent_difference,
  shuffle,
  exchange,
  radix_rank,
  merge_sort_keys,
  merge_sort_pairs,
  radix_sort_keys,
  radix_sort_pairs,
  topk_keys,
  topk_pairs,
  sum_batched,
  reduce_batched,
  count,
};

inline constexpr size_t family_count = static_cast<size_t>(FamilyId::count);

struct Topology
{
  int block_threads                    = 0;
  int items_per_thread                 = 0;
  int decoded_items_per_thread         = 0;
  int reduction_batches_per_invocation = 1;
};

enum class TargetArchitectureFamily
{
  none,
  a,
  f,
};

struct TargetArchitecture
{
  int code                        = 0;
  TargetArchitectureFamily family = TargetArchitectureFamily::none;

  [[nodiscard]] constexpr bool valid() const noexcept
  {
    return code > 0
        && (family == TargetArchitectureFamily::none || family == TargetArchitectureFamily::a
            || family == TargetArchitectureFamily::f);
  }

  [[nodiscard]] std::string sm_name() const;
  [[nodiscard]] std::string compute_name() const;

  friend constexpr bool operator==(const TargetArchitecture& lhs, const TargetArchitecture& rhs) noexcept
  {
    return lhs.code == rhs.code && lhs.family == rhs.family;
  }

  friend constexpr bool operator!=(const TargetArchitecture& lhs, const TargetArchitecture& rhs) noexcept
  {
    return !(lhs == rhs);
  }
};

struct Target
{
  TargetArchitecture sm_arch;
  int64_t cache_epoch = 0;
};

struct ReduceSpec
{
  bool generic = false;
  // Zero preserves the original input-shaped broadcast ABI.
  int64_t batches = 0;
  std::string dtype;
  std::string algorithm;
  std::string binary_op;
};

struct ScanSpec
{
  bool generic = false;
  std::string dtype;
  std::string mode;
  std::string algorithm;
  std::string scan_op;
  bool initial_present = false;
  bool prefix_present  = false;
};

struct HistogramSpec
{
  bool fixed_256 = false;
  int bins       = 0;
  std::string dtype;
};

struct RunLengthDecodeSpec
{
  bool with_offsets = false;
  std::string value_dtype;
  std::string length_dtype;
  int64_t decoded_extent = 0;
};

struct DiscontinuitySpec
{
  bool heads_and_tails = false;
  std::string mode;
  std::string dtype;
};

struct AdjacentDifferenceSpec
{
  std::string mode;
  std::string dtype;
};

struct ExchangeSpec
{
  int64_t block_threads = 0;
  std::string mode;
  std::string dtype;
};

struct ShuffleSpec
{
  std::string mode;
  std::string dtype;
};

struct RadixRankSpec
{
  std::string dtype;
  bool descending         = false;
  int radix_bits          = 4;
  bool memoize_outer_scan = true;
  std::string inner_scan_algorithm;
  std::string smem_config;
};

struct MergeSortSpec
{
  bool pairs = false;
  std::string key_dtype;
  std::string value_dtype;
  bool descending   = false;
  bool partial_tile = false;
};

struct RadixSortSpec
{
  bool pairs = false;
  std::string key_dtype;
  std::string value_dtype;
  bool descending         = false;
  int radix_bits          = 4;
  bool memoize_outer_scan = true;
  std::string inner_scan_algorithm;
  std::string smem_config;
};

struct TopKSpec
{
  bool pairs = false;
  std::string key_dtype;
  std::string value_dtype;
  bool descending = false;
  bool full_tile  = true;
};

using PrimitiveSpec =
  std::variant<ReduceSpec,
               ScanSpec,
               HistogramSpec,
               RunLengthDecodeSpec,
               DiscontinuitySpec,
               AdjacentDifferenceSpec,
               ExchangeSpec,
               ShuffleSpec,
               RadixRankSpec,
               MergeSortSpec,
               RadixSortSpec,
               TopKSpec>;

struct MetadataResult;

// Operation semantics and logical extents, independent of a selected worker
// count. Only the binding factory can construct a request; consumers see const
// data and cannot invalidate its invariants after admission.
class LogicalRequest
{
public:
  [[nodiscard]] const PrimitiveSpec& primitive() const noexcept
  {
    return primitive_;
  }
  [[nodiscard]] int64_t extent() const noexcept
  {
    return extent_;
  }
  [[nodiscard]] const Target& target() const noexcept
  {
    return target_;
  }
  [[nodiscard]] bool automatic_topology() const noexcept
  {
    return !std::holds_alternative<ExchangeSpec>(primitive_);
  }

private:
  LogicalRequest(PrimitiveSpec primitive, int64_t extent, Target target)
      : primitive_(std::move(primitive))
      , extent_(extent)
      , target_(target)
  {}

  friend MetadataResult make_logical_request(FamilyId, PrimitiveSpec, int64_t, Target, std::optional<LogicalRequest>*);

  PrimitiveSpec primitive_;
  int64_t extent_;
  Target target_;
};

// A concrete, validated CUB realization. Source emission, resource probing and
// cache identity consume this type, never an automatic or partially bound call.
class ImplementationSpec
{
public:
  [[nodiscard]] const LogicalRequest& demand() const noexcept
  {
    return demand_;
  }
  [[nodiscard]] const PrimitiveSpec& primitive() const noexcept
  {
    return demand_.primitive();
  }
  [[nodiscard]] const Target& target() const noexcept
  {
    return demand_.target();
  }
  [[nodiscard]] const Topology& topology() const noexcept
  {
    return topology_;
  }

private:
  ImplementationSpec(LogicalRequest demand, Topology topology)
      : demand_(std::move(demand))
      , topology_(topology)
  {}

  friend MetadataResult select_implementation(const LogicalRequest&, int64_t, std::optional<ImplementationSpec>*);

  LogicalRequest demand_;
  Topology topology_;
};

[[nodiscard]] bool uses_warp_execution_scope(const ImplementationSpec& request) noexcept;
FamilyId family_id(const PrimitiveSpec& primitive);
const char* family_name(FamilyId family);
const char* request_payload(FamilyId family);
const char* required_header(FamilyId family);
} // namespace cuda::coop::tile::detail::provider
