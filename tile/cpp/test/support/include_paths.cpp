// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "provider/runtime/runtime.h"

namespace cuda::coop::tile::detail::provider
{
void emit_trace_line(const std::string&) {}
} // namespace cuda::coop::tile::detail::provider

int main()
{
  namespace provider = cuda::coop::tile::detail::provider;
  std::string error;
  if (!provider::validate_provider_configuration(&error))
  {
    std::cerr << error << '\n';
    return 1;
  }
  for (const auto& path : provider::resolve_provider_include_paths())
  {
    std::cout << path << '\n';
  }
}
