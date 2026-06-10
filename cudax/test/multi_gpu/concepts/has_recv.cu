//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/concepts.h>

#include "concepts_common.cuh"

namespace
{
namespace cudax = ::cuda::experimental;
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct recv_returns_int : types::synchronous_communicator_model
{
  template <class Tp>
  int recv(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::__has_recv<types::communicator_model>);
  static_assert(cudax::__has_recv<types::communicator_model, long*>);
  static_assert(!cudax::__has_recv<types::communicator_without_recv>);

  static_assert(!cudax::__has_recv<recv_returns_int>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
