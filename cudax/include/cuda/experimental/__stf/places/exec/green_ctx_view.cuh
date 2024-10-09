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

/**
 * @file
 * @brief Implementation of green context views
 */

#include "cudastf/__stf/internal/async_resources_handle.h"
#include "cudastf/__stf/utility/hash.h"

#if CUDA_VERSION >= 12040
namespace cuda::experimental::stf {

// Green contexts are only supported since CUDA 12.4
/**
 * @brief View of a green context and a pool of CUDA streams
 */
class green_ctx_view {
public:
    green_ctx_view(CUgreenCtx g_ctx, ::std::shared_ptr<stream_pool> pool, int devid)
            : g_ctx(g_ctx), pool(mv(pool)), devid(devid) {}

    CUgreenCtx g_ctx;
    ::std::shared_ptr<stream_pool> pool;
    int devid;

    bool operator==(const green_ctx_view& other) const {
        return (g_ctx == other.g_ctx) && (pool.get() == other.pool.get()) && (devid == other.devid);
    }
};

}  // end namespace cuda::experimental::stf

template <>
struct std::hash<cuda::experimental::stf::green_ctx_view> {
    ::std::size_t operator()(const cuda::experimental::stf::green_ctx_view& k) const {
        return cuda::experimental::stf::hash_all(k.g_ctx, k.pool, k.devid);
    }
};

#endif
