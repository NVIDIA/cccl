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

#include <cuda/experimental/__multi_gpu/concepts.h>

#include "collective_concepts_common.cuh"

namespace
{
namespace cudax = ::cuda::experimental;
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct all_gather_returns_int : types::communicator_model
{
  template <class Tp>
  int all_gather(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::stream_ref);
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::__has_all_gather<types::collective_communicator_model>);
  static_assert(cudax::__has_all_gather<types::collective_communicator_model, long*>);
  static_assert(!cudax::__has_all_gather<types::communicator_model>);

  static_assert(!cudax::__has_all_gather<all_gather_returns_int>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
