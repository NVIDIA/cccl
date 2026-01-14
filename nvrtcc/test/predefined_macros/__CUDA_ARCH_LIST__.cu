//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "../common/check_predefined_macros.h"

template <int... Args>
struct ArchList
{};

template <int... Vs, int... Refs>
__host__ __device__ constexpr bool test_cuda_arch_list(ArchList<Vs...>, ArchList<Refs...>) noexcept
{
  static_assert(sizeof...(Vs) == sizeof...(Refs));
  return ((Vs == Refs) && ...);
}

static_assert(test_cuda_arch_list(ArchList<__CUDA_ARCH_LIST__>{}, ArchList<750, 800, 890, 900>{}));
