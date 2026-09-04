//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief The STF utilities the sharded tier calls unqualified:
 * `cuda_safe_call`, `cuda_try` and the `each` index range.
 *
 * The tier calls them unqualified from inside `cuda::experimental::sharded`,
 * which reaches them only through a using-declaration in that namespace.
 * Providing it here — next to the includes of the headers that define them —
 * lets every header that uses them say so with one include, instead of
 * depending on some other header having been included first.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

namespace cuda::experimental::sharded
{
using ::cuda::experimental::stf::cuda_safe_call;
using ::cuda::experimental::stf::cuda_try;
using ::cuda::experimental::stf::each;
} // namespace cuda::experimental::sharded
