//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define CCCL_DISABLE_EXCEPTIONS

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/cassert>

#include <nv/target>

__host__ __device__ constexpr int exception_value()
{
  return 42;
}

struct ExceptionBase
{
  int value = exception_value();

  [[nodiscard]] __host__ __device__ static const char* what() noexcept
  {
    return "ExceptionBase";
  }
};

struct Exception : ExceptionBase
{};

__host__ __device__ void test()
{
  _CCCL_TRY
  {
    _CCCL_THROW(Exception());
  }
  _CCCL_CATCH ([[maybe_unused]] Exception e)
  {
  }
  _CCCL_CATCH ([[maybe_unused]] ExceptionBase e)
  {
  }
  _CCCL_CATCH_ALL {}
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
