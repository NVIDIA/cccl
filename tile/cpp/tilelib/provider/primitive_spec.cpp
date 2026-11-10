// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/primitive_spec.h"

#include "provider/registry.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

[[nodiscard]] char target_architecture_family_suffix(TargetArchitectureFamily family) noexcept
{
  switch (family)
  {
    case TargetArchitectureFamily::a:
      return 'a';
    case TargetArchitectureFamily::f:
      return 'f';
    case TargetArchitectureFamily::none:
      return '\0';
  }
  return '\0';
}
} // namespace

bool uses_warp_execution_scope(const ImplementationSpec& request) noexcept
{
  if (const auto* const reduce = std::get_if<ReduceSpec>(&request.primitive()))
  {
    return reduce->batches > 0 && reduce->algorithm == "warp_reductions";
  }
  // The Tile frontend already encodes a 32-item merge sort as one item per
  // participant across 32 participants. Other families retain their CTA ABI.
  return request.topology().block_threads == 32 && request.topology().items_per_thread == 1
      && std::holds_alternative<MergeSortSpec>(request.primitive());
}

std::string TargetArchitecture::sm_name() const
{
  if (!valid())
  {
    return {};
  }
  std::string name = "sm_" + std::to_string(code);
  if (const char suffix = target_architecture_family_suffix(family); suffix != '\0')
  {
    name.push_back(suffix);
  }
  return name;
}

std::string TargetArchitecture::compute_name() const
{
  if (!valid())
  {
    return {};
  }
  std::string name = "compute_" + std::to_string(code);
  if (const char suffix = target_architecture_family_suffix(family); suffix != '\0')
  {
    name.push_back(suffix);
  }
  return name;
}

FamilyId family_id(const PrimitiveSpec& primitive)
{
  return std::visit(
    overloaded{
      [](const ReduceSpec& spec) {
        if (spec.batches > 0)
        {
          return spec.generic ? FamilyId::reduce_batched : FamilyId::sum_batched;
        }
        return spec.generic ? FamilyId::reduce : FamilyId::sum;
      },
      [](const ScanSpec&) {
        return FamilyId::scan;
      },
      [](const HistogramSpec& spec) {
        return spec.fixed_256 ? FamilyId::histogram_256 : FamilyId::histogram;
      },
      [](const RunLengthDecodeSpec& spec) {
        return spec.with_offsets ? FamilyId::run_length_decode_with_offsets : FamilyId::run_length_decode;
      },
      [](const DiscontinuitySpec& spec) {
        return spec.heads_and_tails ? FamilyId::discontinuity_heads_and_tails : FamilyId::discontinuity_flags;
      },
      [](const AdjacentDifferenceSpec&) {
        return FamilyId::adjacent_difference;
      },
      [](const ExchangeSpec&) {
        return FamilyId::exchange;
      },
      [](const ShuffleSpec&) {
        return FamilyId::shuffle;
      },
      [](const RadixRankSpec&) {
        return FamilyId::radix_rank;
      },
      [](const MergeSortSpec& spec) {
        return spec.pairs ? FamilyId::merge_sort_pairs : FamilyId::merge_sort_keys;
      },
      [](const RadixSortSpec& spec) {
        return spec.pairs ? FamilyId::radix_sort_pairs : FamilyId::radix_sort_keys;
      },
      [](const TopKSpec& spec) {
        return spec.pairs ? FamilyId::topk_pairs : FamilyId::topk_keys;
      }},
    primitive);
}

const char* family_name(FamilyId family)
{
  return family_descriptor(family).diagnostic_name;
}

const char* request_payload(FamilyId family)
{
  return family_descriptor(family).payload;
}

const char* required_header(FamilyId family)
{
  return family_descriptor(family).required_header;
}
} // namespace cuda::coop::tile::detail::provider
