// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "tile_ir/linkage_provider/LinkageProviderAPI.h"

namespace cuda::coop::tile::detail::provider
{
namespace detail
{
inline std::string copy_string(TileIRStringView value)
{
  return value.data && value.len != 0 ? std::string(value.data, value.len) : std::string{};
}

inline bool valid_string(TileIRStringView value)
{
  return value.data || value.len == 0;
}
} // namespace detail

// Borrow a discovery callsite and compilation context for metadata parsing.
// The compiler owns both records and all arrays they reference. Keep this view
// inside the callback; copy strings and validated values into LogicalRequest
// before retaining anything. A missing discovery worker count is represented
// by zero, so it cannot be treated as a final kernel execution width.
class DiscoveryCallView
{
public:
  static constexpr bool is_resolved = false;

  DiscoveryCallView(const TileIRLPCallsite& callsite, const TileIRLPCompilationContext& context)
      : callsite_(callsite)
      , context_(context)
  {}

  const TileIRForeignDecl* declaration() const
  {
    return callsite_.decl;
  }

  size_t constant_count() const
  {
    return callsite_.num_const_arguments;
  }

  size_t argument_count() const
  {
    return callsite_.num_arguments;
  }

  size_t result_count() const
  {
    return callsite_.num_results;
  }

  const TileIRForeignType* arguments() const
  {
    return callsite_.arguments;
  }

  const TileIRForeignType* results() const
  {
    return callsite_.results;
  }

  bool int_constant(size_t index, int64_t* value) const
  {
    if (!value || !callsite_.const_arguments || index >= constant_count()
        || callsite_.const_arguments[index].kind != TILEIR_SYMBOLIC_EXPRESSION_INT)
    {
      return false;
    }
    *value = callsite_.const_arguments[index].int_val;
    return true;
  }

  bool string_constant(size_t index, std::string* value) const
  {
    if (!value || !callsite_.const_arguments || index >= constant_count()
        || callsite_.const_arguments[index].kind != TILEIR_SYMBOLIC_EXPRESSION_STRING
        || !detail::valid_string(callsite_.const_arguments[index].string_val))
    {
      return false;
    }
    *value = detail::copy_string(callsite_.const_arguments[index].string_val);
    return true;
  }

  bool dtype_constant(size_t index, TileIRDataType* value) const
  {
    if (!value || !callsite_.const_arguments || index >= constant_count()
        || callsite_.const_arguments[index].kind != TILEIR_SYMBOLIC_EXPRESSION_DTYPE)
    {
      return false;
    }
    *value = callsite_.const_arguments[index].dtype_val;
    return true;
  }

  int64_t execution_threads() const
  {
    return context_.num_warps_per_cta > 0 ? context_.num_warps_per_cta * 32 : 0;
  }

  TileIRStringView architecture() const
  {
    return context_.target_architecture;
  }

  bool has_resolved_cta_scope() const
  {
    return true;
  }

  bool has_resolved_warp_scope() const
  {
    return false;
  }

  bool topology_matches(int64_t block_threads) const
  {
    return execution_threads() <= 0 || execution_threads() == block_threads;
  }

  uint64_t callsite_id() const
  {
    return callsite_.id;
  }

private:
  const TileIRLPCallsite& callsite_;
  const TileIRLPCompilationContext& context_;
};

// Adapt a compiler-selected call to the same parser interface as discovery.
// The view borrows the resolved call. Its execution scope is now authoritative:
// a warp-scoped implementation has 32 participants even inside a wider CTA.
// Resolved declarations no longer carry symbolic constant-parameter schemas.
class ResolvedCallView
{
public:
  static constexpr bool is_resolved = true;

  explicit ResolvedCallView(const TileIRResolvedCall& call)
      : call_(call)
  {}

  const TileIRForeignDecl* declaration() const
  {
    return call_.decl;
  }

  size_t constant_count() const
  {
    return call_.num_const_arguments;
  }

  size_t argument_count() const
  {
    return call_.num_arguments;
  }

  size_t result_count() const
  {
    return call_.decl ? call_.decl->num_results : 0;
  }

  const TileIRForeignType* arguments() const
  {
    return call_.arguments;
  }

  const TileIRForeignType* results() const
  {
    return call_.decl ? call_.decl->results : nullptr;
  }

  bool int_constant(size_t index, int64_t* value) const
  {
    if (!value || !call_.const_arguments || index >= constant_count()
        || call_.const_arguments[index].kind != TILEIR_CONSTANT_ARGUMENT_INT)
    {
      return false;
    }
    *value = call_.const_arguments[index].int_val;
    return true;
  }

  bool string_constant(size_t index, std::string* value) const
  {
    if (!value || !call_.const_arguments || index >= constant_count()
        || call_.const_arguments[index].kind != TILEIR_CONSTANT_ARGUMENT_STRING
        || !detail::valid_string(call_.const_arguments[index].string_val))
    {
      return false;
    }
    *value = detail::copy_string(call_.const_arguments[index].string_val);
    return true;
  }

  bool dtype_constant(size_t index, TileIRDataType* value) const
  {
    if (!value || !call_.const_arguments || index >= constant_count()
        || call_.const_arguments[index].kind != TILEIR_CONSTANT_ARGUMENT_DTYPE)
    {
      return false;
    }
    *value = call_.const_arguments[index].dtype_val;
    return true;
  }

  int64_t execution_threads() const
  {
    // A warp-scoped implementation can be selected inside a physically wider
    // CTA, so its execution width is the participant warp rather than the CTA.
    if (call_.resolved_execution_context.scope == TILEIR_EXEC_SCOPE_WARP)
    {
      return 32;
    }
    return call_.resolved_execution_context.num_warps_per_cta > 0
           ? call_.resolved_execution_context.num_warps_per_cta * 32
           : 0;
  }

  TileIRStringView architecture() const
  {
    return call_.resolved_execution_context.architecture;
  }

  bool has_resolved_cta_scope() const
  {
    return call_.resolved_execution_context.scope == TILEIR_EXEC_SCOPE_CTA;
  }

  bool has_resolved_warp_scope() const
  {
    return call_.resolved_execution_context.scope == TILEIR_EXEC_SCOPE_WARP;
  }

  bool topology_matches(int64_t block_threads) const
  {
    return execution_threads() == block_threads;
  }

  uint64_t callsite_id() const
  {
    return 0;
  }

private:
  const TileIRResolvedCall& call_;
};

// Dispatch a family parser to either borrowed call representation.
// Copying this variant copies the views, not the compiler-owned ABI records.
class MetadataCallView
{
public:
  explicit MetadataCallView(const DiscoveryCallView& call)
      : call_(call)
  {}

  explicit MetadataCallView(const ResolvedCallView& call)
      : call_(call)
  {}

  template <class Visitor>
  decltype(auto) visit(Visitor&& visitor) const
  {
    return std::visit(std::forward<Visitor>(visitor), call_);
  }

private:
  std::variant<DiscoveryCallView, ResolvedCallView> call_;
};
} // namespace cuda::coop::tile::detail::provider
