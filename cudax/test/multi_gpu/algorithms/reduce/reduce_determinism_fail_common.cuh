//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_DETERMINISM_FAIL_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_DETERMINISM_FAIL_COMMON_CUH

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/execution>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

namespace cudax = ::cuda::experimental;

// Calls `reduce` with an environment that carries `determinism`. Nothing here runs. A null
// communicator, a null stream and null iterators only have to satisfy the constraints on `reduce`,
// so that the body is instantiated and the static_assert fires.
template <class Determinism>
void reduce_with_determinism(Determinism determinism)
{
  cudax::nccl_communicator_ref comm{::ncclComm_t{}};
  auto env = ::cuda::std::execution::env{::cuda::stream_ref{::cudaStream_t{}}, ::cuda::execution::require(determinism)};
  int* ptr{};

  cudax::reduce(cudax::broadcasted, comm, env, ptr, ::cuda::std::size_t{0}, ptr);
}

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_DETERMINISM_FAIL_COMMON_CUH
