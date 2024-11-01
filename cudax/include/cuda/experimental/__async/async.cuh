//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_ASYNC
#define __CUDAX_ASYNC_DETAIL_ASYNC

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Include this first
#include <cuda/experimental/__detail/config.cuh>

// Include the other implementation headers:
#include <cuda/experimental/__async/basic_sender.cuh>
#include <cuda/experimental/__async/conditional.cuh>
#include <cuda/experimental/__async/continue_on.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/just.cuh>
#include <cuda/experimental/__async/just_from.cuh>
#include <cuda/experimental/__async/let_value.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/read_env.cuh>
#include <cuda/experimental/__async/run_loop.cuh>
#include <cuda/experimental/__async/sequence.cuh>
#include <cuda/experimental/__async/start_detached.cuh>
#include <cuda/experimental/__async/start_on.cuh>
#include <cuda/experimental/__async/stop_token.cuh>
#include <cuda/experimental/__async/sync_wait.cuh>
#include <cuda/experimental/__async/then.cuh>
#include <cuda/experimental/__async/thread_context.cuh>
#include <cuda/experimental/__async/when_all.cuh>
#include <cuda/experimental/__async/write_env.cuh>

#endif // __CUDAX_ASYNC_DETAIL_ASYNC
