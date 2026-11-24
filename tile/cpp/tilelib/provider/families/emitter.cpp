// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/emitter.h"

#include "provider/registry.h"

namespace cuda::coop::tile::detail::provider
{
SourceEmissionStatus emit_family_source(const SourceEmissionRequest& request, std::string* source, std::string* error)
{
  if (!source)
  {
    if (error)
    {
      *error = "null source destination";
    }
    return SourceEmissionStatus::invalid;
  }
  if (request.callable_symbol.empty())
  {
    if (error)
    {
      *error = "callable symbol is empty";
    }
    return SourceEmissionStatus::invalid;
  }

  const FamilySourceEmitter emitter = family_descriptor(family_id(request.provider.primitive())).emit_source;
  if (!emitter)
  {
    return SourceEmissionStatus::unhandled;
  }
  return emitter(request.provider, request.callable_symbol, source, error)
         ? SourceEmissionStatus::emitted
         : SourceEmissionStatus::invalid;
}
} // namespace cuda::coop::tile::detail::provider
