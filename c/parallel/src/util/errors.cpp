//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "errors.h"

#include <stdexcept>

void check(nvrtcResult result)
{
  if (result != NVRTC_SUCCESS)
  {
    throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(result));
  }
}

void check(CUresult result)
{
  if (result != CUDA_SUCCESS)
  {
    const char* str = nullptr;
    cuGetErrorString(result, &str);
    throw std::runtime_error(std::string("CUDA error: ") + str);
  }
}

void check(nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS)
  {
    throw std::runtime_error(std::string("nvJitLink error: ") + std::to_string(result));
  }
}
