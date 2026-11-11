// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <optional>

#include "provider/metadata_parser.h"
#include "provider/primitive_spec.h"

namespace cuda::coop::tile::detail::provider
{
[[nodiscard]] MetadataResult validate_logical_extents(FamilyId family, int64_t extent, int64_t decoded_extent = 0);

// Family binding has already checked the ABI and operation options. Freeze the
// logical shape and operation semantics before enumerating workers.
[[nodiscard]] MetadataResult make_logical_request(
  FamilyId family, PrimitiveSpec primitive, int64_t extent, Target target, std::optional<LogicalRequest>* out);

// Select one candidate without modifying the demand. Per-thread extents derive
// from workers; Exchange preserves its semantic permutation width.
[[nodiscard]] MetadataResult
select_implementation(const LogicalRequest& demand, int64_t execution_threads, std::optional<ImplementationSpec>* out);
} // namespace cuda::coop::tile::detail::provider
