//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include <cccl/c/pch.h>
#include <hostjit/jit_compiler.hpp>

size_t cccl_hostjit_pch_cache_dir(char* out, size_t out_size)
try
{
  const std::string dir = hostjit::pchCacheDir();
  if (dir.empty())
  {
    if (out && out_size > 0)
    {
      out[0] = '\0';
    }
    return 0;
  }

  const size_t needed = dir.size() + 1;
  if (out && out_size > 0)
  {
    const size_t copied = (needed <= out_size) ? dir.size() : out_size - 1;
    std::memcpy(out, dir.data(), copied);
    out[copied] = '\0';
  }
  return needed;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_hostjit_pch_cache_dir(): %s\n", exc.what());
  return 0;
}
