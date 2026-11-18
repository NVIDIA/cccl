//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// REQUIRES: nvcc
// This regression exercises NVCC's separate host and device compilation passes.
// UNSUPPORTED: nvcc-12, nvcc-13.0, nvcc-13.1, nvcc-13.2, nvcc-13.3
// The status policies are exposed through <cuda/barrier> with CUDA 13.4 and later.
// UNSUPPORTED: pre-sm-70
// <cuda/barrier> requires SM70 or newer.
// UNSUPPORTED: enable-tile
// This regression exercises CUDA C++ compilation, not tile code.

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
// Declare utility CPOs in both passes so this test isolates status-policy registration.
#include <cuda/std/utility>

#include "test_macros.h"

// Including the status policies only in the device pass must not create host-registration references to their tags.
#if defined(__CUDA_ARCH__)
#  include <cuda/barrier>

static_assert(cuda::std::is_empty_v<cuda::return_status_t>);
static_assert(cuda::std::is_empty_v<cuda::ignore_status_t>);
static_assert(cuda::std::is_trivially_copy_constructible_v<cuda::return_status_t>);
static_assert(cuda::std::is_trivially_copy_constructible_v<cuda::ignore_status_t>);
static_assert(cuda::std::is_same_v<decltype(cuda::return_status), const cuda::return_status_t>);
static_assert(cuda::std::is_same_v<decltype(cuda::ignore_status), const cuda::ignore_status_t>);

TEST_DEVICE_FUNC int select_status(cuda::return_status_t)
{
  return 1;
}

TEST_DEVICE_FUNC int select_status(cuda::ignore_status_t)
{
  return 2;
}
#endif // defined(__CUDA_ARCH__)

int main(int, char**)
{
#if defined(__CUDA_ARCH__)
  assert(select_status(cuda::return_status) == 1);
  assert(select_status(cuda::ignore_status) == 2);
#endif // defined(__CUDA_ARCH__)
  return 0;
}
