//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file
 * @brief Implementation of green context views
 */

#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

#if _CCCL_CTK_AT_LEAST(12, 4)

namespace cuda::experimental::places
{
template <typename T>
struct hash;

// Green contexts are only supported since CUDA 12.4
/**
 * @brief View of a green context and a pool of CUDA streams
 */
class green_ctx_view
{
public:
  green_ctx_view(CUgreenCtx g_ctx, stream_pool pool, int devid)
      : g_ctx(g_ctx)
      , pool(::cuda::experimental::stf::mv(pool))
      , devid(devid)
  {}

  CUgreenCtx g_ctx;
  stream_pool pool;
  int devid;

  bool operator==(const green_ctx_view& other) const
  {
    return (g_ctx == other.g_ctx) && (devid == other.devid);
  }

  bool operator<(const green_ctx_view& other) const
  {
    if (g_ctx != other.g_ctx)
    {
      return g_ctx < other.g_ctx;
    }
    return devid < other.devid;
  }
};

/**
 * @brief Specialization of `places::hash` for `green_ctx_view`
 */
template <>
struct hash<green_ctx_view>
{
  ::std::size_t operator()(const green_ctx_view& k) const
  {
    return ::cuda::experimental::stf::hash_all(k.g_ctx, k.devid);
  }
};
} // end namespace cuda::experimental::places

#endif // _CCCL_CTK_AT_LEAST(12, 4)
