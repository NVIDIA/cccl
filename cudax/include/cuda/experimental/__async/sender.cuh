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
#include <cuda/experimental/__detail/config.cuh> // IWYU pragma: export

// Include the other implementation headers:
#include <cuda/experimental/__async/sender/conditional.cuh>    // IWYU pragma: export
#include <cuda/experimental/__async/sender/continue_on.cuh>    // IWYU pragma: export
#include <cuda/experimental/__async/sender/cpos.cuh>           // IWYU pragma: export
#include <cuda/experimental/__async/sender/just.cuh>           // IWYU pragma: export
#include <cuda/experimental/__async/sender/just_from.cuh>      // IWYU pragma: export
#include <cuda/experimental/__async/sender/let_value.cuh>      // IWYU pragma: export
#include <cuda/experimental/__async/sender/queries.cuh>        // IWYU pragma: export
#include <cuda/experimental/__async/sender/read_env.cuh>       // IWYU pragma: export
#include <cuda/experimental/__async/sender/run_loop.cuh>       // IWYU pragma: export
#include <cuda/experimental/__async/sender/sequence.cuh>       // IWYU pragma: export
#include <cuda/experimental/__async/sender/start_detached.cuh> // IWYU pragma: export
#include <cuda/experimental/__async/sender/start_on.cuh>       // IWYU pragma: export
#include <cuda/experimental/__async/sender/stop_token.cuh>     // IWYU pragma: export
#include <cuda/experimental/__async/sender/sync_wait.cuh>      // IWYU pragma: export
#include <cuda/experimental/__async/sender/then.cuh>           // IWYU pragma: export
#include <cuda/experimental/__async/sender/thread_context.cuh> // IWYU pragma: export
#include <cuda/experimental/__async/sender/visit.cuh>          // IWYU pragma: export
#include <cuda/experimental/__async/sender/when_all.cuh>       // IWYU pragma: export
#include <cuda/experimental/__async/sender/write_env.cuh>      // IWYU pragma: export

#endif // __CUDAX_ASYNC_DETAIL_ASYNC
