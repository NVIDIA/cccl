// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#include "provider/primitive_spec.h"

namespace cuda::coop::tile::detail::provider
{
[[nodiscard]] bool emit_histogram_source(
  const ImplementationSpec& request, const std::string& entry_name, std::string* source, std::string* error);
[[nodiscard]] bool emit_run_length_decode_source(
  const ImplementationSpec& request, const std::string& entry_name, std::string* source, std::string* error);
} // namespace cuda::coop::tile::detail::provider
