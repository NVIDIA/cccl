// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "provider/primitive_spec.h"
#include "tile_ir/linkage_provider/LinkageProviderAPI.h"

namespace cuda::coop::tile::detail::provider
{
class MetadataCallView;
struct CallMetadata;
struct FamilyDescriptor;
struct TargetArchitecture;

// Metadata parsing is the provider's trust and normalization boundary between
// Tile IR's C ABI and LogicalRequest. Candidate selection then constructs the
// ImplementationSpec consumed by source emission, resource probes and caches.
// Discovery binds a call once before trying worker counts; ranking and
// instantiation rebind the external resolved call through the same path.
//
// A parser verifies that an owned declaration agrees with its symbol, schema,
// constants, argument/result types, topology, and target. Centralizing those
// checks prevents malformed input from being mistaken for an unavailable
// specialization or from reaching caches and code generation with a partial
// interpretation. CallMetadata is populated only for supported input and is
// reset for every other outcome.
enum class MetadataStatus
{
  // The request is valid, implemented, and represented in CallMetadata.
  supported,
  // The symbol belongs to another provider. Registry lookup normally
  // synthesizes this outcome before a family parser is selected.
  not_owned,
  // The request is valid, but this provider does not implement the requested
  // algorithm, mode, dtype, topology, capacity, or resource specialization.
  unsupported,
  // The request contradicts the owned declaration or the Tile IR ABI schema.
  malformed,
  // Provider registry, identity, typed-request, or source-emission invariants
  // failed after the input had otherwise been admitted.
  internal_error,
};

// Callbacks use the status for control flow and propagate diagnostic through
// the LP's structured error payload. Unsupported, malformed, and internal
// results always carry a nonempty, categorized diagnostic; supported and
// not-owned results do not need one.
struct MetadataResult
{
  MetadataStatus status  = MetadataStatus::internal_error;
  std::string diagnostic = "[LP-PROVIDER-INTERNAL-001] metadata result was not initialized";

  [[nodiscard]] bool is_supported() const noexcept
  {
    return status == MetadataStatus::supported;
  }

  [[nodiscard]] static MetadataResult success();
  [[nodiscard]] static MetadataResult not_owned();
  [[nodiscard]] static MetadataResult unsupported(std::string diagnostic);
  [[nodiscard]] static MetadataResult malformed(std::string diagnostic);
  [[nodiscard]] static MetadataResult internal_error(std::string diagnostic);
};

// Family parsers are registered in FamilyDescriptor. Their target has already
// been resolved and the generated declaration schema checked. They bind either
// a discovery or resolved call through MetadataCallView, preserving logical
// extents and explicit constraints without selecting workers. On failure the
// optional logical request is reset.
using FamilyMetadataParser = MetadataResult (*)(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);

[[nodiscard]] MetadataResult parse_reduce_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_scan_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_histogram_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_run_length_decode_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_discontinuity_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_adjacent_difference_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_shuffle_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_exchange_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_radix_rank_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_merge_sort_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_radix_sort_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);
[[nodiscard]] MetadataResult parse_topk_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out);

// Resolve the target only after registry lookup has established Coop
// ownership. A nonempty compilation/resolved context is authoritative;
// otherwise the provider environment fallback is used. Invalid authoritative
// input is malformed rather than permission to try a lower-precedence source.
[[nodiscard]] MetadataResult
resolve_request_target(TileIRStringView architecture, const FamilyDescriptor& entry, TargetArchitecture* out);

// Discovery overloads normalize an LP callsite. The first resolves its target;
// the second reuses a target resolved once by an auto-topology search.
[[nodiscard]] MetadataResult parse_call_metadata(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& compilation_context,
  const FamilyDescriptor& entry,
  int64_t auto_block_threads_override,
  CallMetadata* out);
[[nodiscard]] MetadataResult parse_call_metadata(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& compilation_context,
  const FamilyDescriptor& entry,
  int64_t auto_block_threads_override,
  TargetArchitecture target_architecture,
  CallMetadata* out);

// Resolved calls are revalidated during ranking and instantiation so selected
// metadata cannot bypass declaration, target, or execution-scope checks. TileIR
// has already concretized and renamed the declaration at this boundary, and no
// longer carries its constant-parameter names. Resolved parsing therefore
// validates the ordered concrete argument count and kinds plus the concrete
// declaration types/access, while discovery remains responsible for validating
// the canonical constant-parameter schema.
[[nodiscard]] MetadataResult parse_call_metadata(
  const TileIRResolvedCall& call, const FamilyDescriptor& entry, size_t call_index, CallMetadata* out);

// Discovery binds and validates one logical call before enumerating candidates.
[[nodiscard]] MetadataResult bind_call_request(
  const TileIRLPCallsite&,
  const TileIRLPCompilationContext&,
  const FamilyDescriptor&,
  TargetArchitecture,
  std::optional<LogicalRequest>*);
[[nodiscard]] MetadataResult materialize_call_request(
  const TileIRLPCallsite&,
  const TileIRLPCompilationContext&,
  const FamilyDescriptor&,
  const LogicalRequest&,
  int64_t candidate_threads,
  CallMetadata*);

[[nodiscard]] bool merge_sort_block_threads_are_supported(int64_t block_threads);

[[nodiscard]] bool argument_is_scalar_by_value(
  const FamilyDescriptor& entry, size_t argument_index, const TileIRForeignType& type, TileIRLPAccessKind access);
} // namespace cuda::coop::tile::detail::provider
