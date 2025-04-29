//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_ENV
#define __CUDAX_ASYNC_DETAIL_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
using _CUDA_VSTD::execution::env;
using _CUDA_VSTD::execution::env_of_t;
using _CUDA_VSTD::execution::get_env;
using _CUDA_VSTD::execution::prop;

using _CUDA_VSTD::execution::__nothrow_queryable_with;
using _CUDA_VSTD::execution::__query_result_t;
using _CUDA_VSTD::execution::__queryable_with;
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_ENV
