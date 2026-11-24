// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#include "provider/primitive_spec.h"

namespace cuda::coop::tile::detail::provider
{
enum class SourceEmissionStatus
{
  unhandled,
  emitted,
  invalid,
};

struct SourceEmissionRequest
{
  const ImplementationSpec& provider;
  // Discovery uses an internal probe identifier; instantiation passes the
  // compiler-selected implementation symbol verbatim. The emitter does not
  // derive or alias callable identity.
  std::string callable_symbol;
};

// Emit source for a validated concrete implementation through its family.
// The caller supplies the exact callable symbol and owns the destination
// strings. Treat source as usable only when the status is emitted; family
// failure need not leave the destination empty. This stage performs no NVRTC
// compilation and allocates no device scratch.
[[nodiscard]] SourceEmissionStatus
emit_family_source(const SourceEmissionRequest& request, std::string* source, std::string* error);
} // namespace cuda::coop::tile::detail::provider
