//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/execution>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

int main()
{
  namespace cudax = ::cuda::experimental;

  cudax::mgmn::nccl_communicator_ref comm{::ncclComm_t{}};
  auto env = ::cuda::std::execution::env{
    ::cuda::stream_ref{::cudaStream_t{}}, ::cuda::execution::require(::cuda::execution::determinism::gpu_to_gpu)};
  int* ptr{};

  // expected-error {{"Only run_to_run and not_guaranteed reductions are currently supported"}}
  cudax::mgmn::reduce(cudax::broadcasted, comm, env, ptr, ::cuda::std::size_t{0}, ptr);

  return EXIT_FAILURE;
}
