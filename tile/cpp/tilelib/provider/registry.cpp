// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/registry.h"

#include <array>
#include <cstdlib>
#include <cstring>

#include "provider/families/adjacent_difference.h"
#include "provider/families/discontinuity.h"
#include "provider/families/exchange.h"
#include "provider/families/reduce.h"
#include "provider/families/scan.h"
#include "provider/families/shuffle.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
static_assert(family_count == 21, "the provider contract must contain exactly 21 primitive families");

constexpr std::array<FamilyDescriptor, 10> registry{{
  {FamilyId::sum,
   "cuda_coop_tile_sum",
   "cuda_coop_tile_sum_impl",
   "sum",
   "reduce_sum",
   "cub/block/block_reduce.cuh",
   generated::schema_for("cuda_coop_tile_sum"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_reduce_metadata,
   emit_reduce_source},
  {FamilyId::reduce,
   "cuda_coop_tile_reduce",
   "cuda_coop_tile_reduce_impl",
   "reduce",
   "reduce_generic",
   "cub/block/block_reduce.cuh",
   generated::schema_for("cuda_coop_tile_reduce"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_reduce_metadata,
   emit_reduce_source},
  {FamilyId::scan,
   "cuda_coop_tile_scan",
   "cuda_coop_tile_scan_impl",
   "scan",
   "scan",
   "cub/block/block_scan.cuh",
   generated::schema_for("cuda_coop_tile_scan"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_scan_metadata,
   emit_scan_source},
  {FamilyId::discontinuity_flags,
   "cuda_coop_tile_discontinuity_flags",
   "cuda_coop_tile_discontinuity_flags_impl",
   "discontinuity",
   "discontinuity",
   "cub/block/block_discontinuity.cuh",
   generated::schema_for("cuda_coop_tile_discontinuity_flags"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_discontinuity_metadata,
   emit_discontinuity_source},
  {FamilyId::discontinuity_heads_and_tails,
   "cuda_coop_tile_discontinuity_heads_and_tails",
   "cuda_coop_tile_discontinuity_heads_and_tails_impl",
   "discontinuity",
   "discontinuity",
   "cub/block/block_discontinuity.cuh",
   generated::schema_for("cuda_coop_tile_discontinuity_heads_and_tails"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_discontinuity_metadata,
   emit_discontinuity_source},
  {FamilyId::adjacent_difference,
   "cuda_coop_tile_adjacent_difference",
   "cuda_coop_tile_adjacent_difference_impl",
   "adjacent-difference",
   "adjacent_difference",
   "cub/block/block_adjacent_difference.cuh",
   generated::schema_for("cuda_coop_tile_adjacent_difference"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_adjacent_difference_metadata,
   emit_adjacent_difference_source},
  {FamilyId::shuffle,
   "cuda_coop_tile_shuffle",
   "cuda_coop_tile_shuffle_impl",
   "shuffle",
   "shuffle",
   "cub/block/block_shuffle.cuh",
   generated::schema_for("cuda_coop_tile_shuffle"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_shuffle_metadata,
   emit_shuffle_source},
  {FamilyId::exchange,
   "cuda_coop_tile_exchange",
   "cuda_coop_tile_exchange_impl",
   "exchange",
   "exchange",
   "cub/block/block_exchange.cuh",
   generated::schema_for("cuda_coop_tile_exchange"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_exchange_metadata,
   emit_exchange_source},
  {FamilyId::sum_batched,
   "cuda_coop_tile_sum_batched",
   "cuda_coop_tile_sum_batched_impl",
   "sum-batched",
   "reduce_sum_batched",
   "cub/warp/warp_reduce.cuh",
   generated::schema_for("cuda_coop_tile_sum_batched"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_reduce_metadata,
   emit_reduce_source},
  {FamilyId::reduce_batched,
   "cuda_coop_tile_reduce_batched",
   "cuda_coop_tile_reduce_batched_impl",
   "reduce-batched",
   "reduce_generic_batched",
   "cub/warp/warp_reduce.cuh",
   generated::schema_for("cuda_coop_tile_reduce_batched"),
   FamilyRankingPolicy::block_threads_centered_128,
   parse_reduce_metadata,
   emit_reduce_source},
}};

constexpr bool registry_is_complete()
{
  if (registry.size() != 10)
  {
    return false;
  }
  for (size_t index = 0; index < registry.size(); ++index)
  {
    const FamilyDescriptor& descriptor = registry[index];
    if (!descriptor.decl_symbol || !descriptor.impl_symbol || !descriptor.diagnostic_name || !descriptor.payload
        || !descriptor.required_header || !descriptor.schema || !descriptor.parse_metadata || !descriptor.emit_source)
    {
      return false;
    }
  }
  return true;
}

constexpr bool c_strings_equal(const char* lhs, const char* rhs)
{
  if (!lhs || !rhs)
  {
    return lhs == rhs;
  }
  while (*lhs != '\0' && *rhs != '\0')
  {
    if (*lhs != *rhs)
    {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == *rhs;
}

constexpr bool registry_symbols_are_unique()
{
  for (size_t lhs = 0; lhs < registry.size(); ++lhs)
  {
    if (c_strings_equal(registry[lhs].decl_symbol, registry[lhs].impl_symbol))
    {
      return false;
    }
    for (size_t rhs = lhs + 1; rhs < registry.size(); ++rhs)
    {
      if (c_strings_equal(registry[lhs].decl_symbol, registry[rhs].decl_symbol)
          || c_strings_equal(registry[lhs].impl_symbol, registry[rhs].impl_symbol)
          || c_strings_equal(registry[lhs].decl_symbol, registry[rhs].impl_symbol)
          || c_strings_equal(registry[lhs].impl_symbol, registry[rhs].decl_symbol))
      {
        return false;
      }
    }
  }
  return true;
}

static_assert(registry_is_complete(), "every registered family must have a complete descriptor");
static_assert(registry_symbols_are_unique(), "provider declaration and implementation symbols must be unique");
} // namespace

const FamilyDescriptor* family_registry()
{
  return registry.data();
}

size_t family_registry_size()
{
  return registry.size();
}

const FamilyDescriptor& family_descriptor(FamilyId family)
{
  for (const FamilyDescriptor& descriptor : registry)
  {
    if (descriptor.family == family)
    {
      return descriptor;
    }
  }
  std::abort();
}

const FamilyDescriptor* find_family_descriptor(const char* symbol, size_t size) noexcept
{
  if (!symbol)
  {
    return nullptr;
  }
  for (const FamilyDescriptor& descriptor : registry)
  {
    const size_t expected_size = std::strlen(descriptor.decl_symbol);
    if (size == expected_size && std::memcmp(symbol, descriptor.decl_symbol, size) == 0)
    {
      return &descriptor;
    }
  }
  return nullptr;
}

namespace
{
constexpr char current_implementation_namespace[] = "__lpv4_";

bool has_decimal_uniquifier(const char* suffix, size_t size) noexcept
{
  if (!suffix || size == 0)
  {
    return false;
  }
  for (size_t i = 0; i < size; ++i)
  {
    if (suffix[i] < '0' || suffix[i] > '9')
    {
      return false;
    }
  }
  return true;
}

bool has_current_implementation_namespace(const char* symbol, size_t size, const char* base) noexcept
{
  if (!symbol || !base)
  {
    return false;
  }
  const size_t base_size      = std::strlen(base);
  const size_t namespace_size = sizeof(current_implementation_namespace) - 1;
  return size >= base_size + namespace_size && std::memcmp(symbol, base, base_size) == 0
      && std::memcmp(symbol + base_size, current_implementation_namespace, namespace_size) == 0;
}
} // namespace

bool symbol_is_concrete_specialization_of(const char* symbol, size_t size, const char* base) noexcept
{
  if (!symbol || !base)
  {
    return false;
  }
  const size_t base_size = std::strlen(base);
  if (size <= base_size + 1 || std::memcmp(symbol, base, base_size) != 0 || symbol[base_size] != '_')
  {
    return false;
  }
  if (symbol[base_size + 1] == '_')
  {
    return size > base_size + 2;
  }
  return has_decimal_uniquifier(symbol + base_size + 1, size - base_size - 1);
}

const FamilyDescriptor* find_family_descriptor_for_implementation(const char* symbol, size_t size) noexcept
{
  for (const FamilyDescriptor& descriptor : registry)
  {
    // Late lookup establishes provider ownership, not final identity.  Claim
    // malformed symbols in the current LPv4 namespace so validation can report
    // identity corruption, while leaving retired or foreign namespaces alone.
    if (has_current_implementation_namespace(symbol, size, descriptor.impl_symbol))
    {
      return &descriptor;
    }
  }
  return nullptr;
}

bool declaration_symbol_matches_family(
  const FamilyDescriptor& descriptor, const char* symbol, size_t size, bool require_concrete_suffix) noexcept
{
  if (require_concrete_suffix)
  {
    // A SymbolTable uniquifier may look like a family suffix.  In particular,
    // `cuda_coop_tile_histogram_256` is a canonical sibling declaration, not a
    // concrete spelling of `cuda_coop_tile_histogram`.  Exact registry symbols
    // always retain their registry meaning; late callbacks require a spelling
    // that is unambiguously derived from this descriptor.
    if (find_family_descriptor(symbol, size))
    {
      return false;
    }
    return symbol_is_concrete_specialization_of(symbol, size, descriptor.decl_symbol);
  }
  const size_t expected_size = std::strlen(descriptor.decl_symbol);
  return symbol && size == expected_size && std::memcmp(symbol, descriptor.decl_symbol, size) == 0;
}

int64_t family_candidate_preference_score(const FamilyDescriptor& descriptor, const ImplementationSpec& request)
{
  switch (descriptor.ranking_policy)
  {
    case FamilyRankingPolicy::block_threads_centered_128: {
      const int64_t block_threads = request.topology().block_threads;
      if (block_threads <= 0 || block_threads > 1024)
      {
        return 0;
      }

      constexpr int64_t preferred_threads = 128;
      const int64_t distance =
        block_threads >= preferred_threads ? block_threads - preferred_threads : preferred_threads - block_threads;
      // Proximity to 128 is the primary ordering. Prefer the smaller topology
      // when two candidates are equally distant so every viable topology has
      // a deterministic provider score.
      const int64_t proximity = 1024 - distance;
      const int64_t tie_break = 1024 - block_threads;
      return proximity * 1024 + tie_break;
    }
  }
  std::abort();
}

std::optional<size_t> argument_dtype_parameter_index(FamilyId family, size_t argument_index, size_t num_arguments)
{
  const auto& schema = *family_descriptor(family).schema;
  if (num_arguments != schema.num_operands || argument_index >= num_arguments)
  {
    return std::nullopt;
  }
  const auto parameter_index = schema.dtype_parameter_indices[argument_index];
  return parameter_index >= 0 ? std::optional<size_t>{static_cast<size_t>(parameter_index)} : std::nullopt;
}
} // namespace cuda::coop::tile::detail::provider
