//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__multi_gpu/concepts.h>

#include "concepts_common.cuh"

namespace
{
namespace cudax = ::cuda::experimental;
namespace types = cudax_multi_gpu_concepts;

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::synchronous_communicator<types::synchronous_communicator_model>);
  static_assert(!cudax::synchronous_communicator<types::synchronous_communicator_without_send>);
  static_assert(!cudax::synchronous_communicator<types::synchronous_communicator_without_recv>);
  static_assert(!cudax::synchronous_communicator<types::synchronous_communicator_with_throwing_rank>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
