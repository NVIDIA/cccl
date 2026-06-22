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

#include <testing.cuh>

#include "concepts_common.cuh"

namespace
{
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct no_send : types::basic_communicator_model
{
  template <class Tp>
  void recv(group_guard_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
};

struct no_recv : types::basic_communicator_model
{
  template <class Tp>
  void send(group_guard_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
};

_CCCL_END_NV_DIAG_SUPPRESS()
} // namespace

C2H_TEST("communicator concept", "[multi_gpu][concepts]")
{
  STATIC_REQUIRE(cudax::__communicator<types::communicator_model>);
  STATIC_REQUIRE(!cudax::__communicator<no_send>);
  STATIC_REQUIRE(!cudax::__communicator<no_recv>);
}
