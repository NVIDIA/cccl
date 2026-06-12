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
#include <cuda/experimental/__multi_gpu/nccl_communicator.h>

namespace
{
namespace cudax = ::cuda::experimental;

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::synchronous_communicator<cudax::nccl_communicator>);
  static_assert(cudax::communicator<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_reduce<cudax::nccl_communicator>);
  static_assert(cudax::__has_reduce<cudax::nccl_communicator, int*>);

  static_assert(cudax::__has_synchronous_all_reduce<cudax::nccl_communicator>);
  static_assert(cudax::__has_all_reduce<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_gather<cudax::nccl_communicator>);
  static_assert(cudax::__has_gather<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_gather_v<cudax::nccl_communicator>);
  static_assert(cudax::__has_gather_v<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_all_gather<cudax::nccl_communicator>);
  static_assert(cudax::__has_all_gather<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_broadcast<cudax::nccl_communicator>);
  static_assert(cudax::__has_broadcast<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_all_to_all<cudax::nccl_communicator>);
  static_assert(cudax::__has_all_to_all<cudax::nccl_communicator>);

  static_assert(cudax::__has_synchronous_all_to_all_v<cudax::nccl_communicator>);
  static_assert(cudax::__has_all_to_all_v<cudax::nccl_communicator>);

  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
