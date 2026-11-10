// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "provider/declaration_schemas.h"
#include "provider/metadata_parser.h"
#include "provider/primitive_spec.h"

namespace cuda::coop::tile::detail::provider
{
enum class FamilyRankingPolicy
{
  block_threads_centered_128,
};

using FamilySourceEmitter = bool (*)(const ImplementationSpec&, const std::string&, std::string*, std::string*);

// Connect a canonical declaration to its parser, emitter, and ranking policy.
// The registry and generated schemas have static storage. A descriptor borrows
// those tables; it carries no per-call state. Family-specific code validates
// semantic options after the shared parser verifies the declaration ABI.
struct FamilyDescriptor
{
  FamilyId family;
  const char* decl_symbol;
  const char* impl_symbol;
  const char* diagnostic_name;
  const char* payload;
  const char* required_header;
  const generated::DeclarationSchema* schema;
  FamilyRankingPolicy ranking_policy;
  FamilyMetadataParser parse_metadata;
  FamilySourceEmitter emit_source;
};

const FamilyDescriptor* family_registry();
size_t family_registry_size();
const FamilyDescriptor& family_descriptor(FamilyId family);
const FamilyDescriptor* find_family_descriptor(const char* symbol, size_t size) noexcept;
const FamilyDescriptor* find_family_descriptor_for_implementation(const char* symbol, size_t size) noexcept;
bool symbol_is_concrete_specialization_of(const char* symbol, size_t size, const char* base) noexcept;
bool declaration_symbol_matches_family(
  const FamilyDescriptor& descriptor, const char* symbol, size_t size, bool require_concrete_suffix) noexcept;
int64_t family_candidate_preference_score(const FamilyDescriptor& descriptor, const ImplementationSpec& request);

std::optional<size_t> argument_dtype_parameter_index(FamilyId family, size_t argument_index, size_t num_arguments);
} // namespace cuda::coop::tile::detail::provider
