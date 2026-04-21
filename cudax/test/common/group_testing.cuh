//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_GROUP_CUH
#define COMMON_GROUP_CUH

#include <cuda/barrier>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <cuda/experimental/group.cuh>

#include "testing.cuh"

namespace
{
template <class T, cuda::std::size_t Id>
__device__ T global_barriers_storage;

//! @brief Returns reference to an array of N cuda::barrier objects with suitable thread scope for level allocated in
//!        suitable address space (shared or device memory). Id parameter can be used to create unique object.
template <cuda::std::size_t N, cuda::std::size_t Id = 0, class Level>
__device__ auto& get_barriers(const Level& level) noexcept
{
  constexpr auto scope = cudax::__minimum_required_scope_for<Level>();

  using Barrier         = cuda::barrier<scope>;
  using BarriersStorage = cuda::std::aligned_storage_t<N * sizeof(Barrier), alignof(Barrier)>;

  if constexpr (scope >= cuda::thread_scope_block)
  {
    __shared__ BarriersStorage shared_barriers_storage;
    return reinterpret_cast<Barrier(&)[N]>(shared_barriers_storage);
  }
  else
  {
    return reinterpret_cast<Barrier(&)[N]>(global_barriers_storage<BarriersStorage, Id>);
  }
}
} // namespace

#endif // COMMON_GROUP_CUH
