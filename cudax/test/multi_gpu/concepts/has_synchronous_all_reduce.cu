//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/experimental/__multi_gpu/concepts.h>

#include "collective_concepts_common.cuh"

namespace
{
namespace cudax = ::cuda::experimental;
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct all_reduce_sync_rejects_void : types::synchronous_communicator_model
{
  template <class Tp, class Op, ::cuda::std::enable_if_t<!::cuda::std::is_void_v<Tp>, int> = 0>
  void all_reduce_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op);
};

struct all_reduce_sync_returns_int : types::synchronous_communicator_model
{
  template <class Tp, class Op>
  int all_reduce_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op);
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::__has_synchronous_all_reduce<types::collective_communicator_model>);
  static_assert(cudax::__has_synchronous_all_reduce<types::collective_communicator_model, long*>);
  static_assert(!cudax::__has_synchronous_all_reduce<types::synchronous_communicator_model>);

  static_assert(!cudax::__has_synchronous_all_reduce<all_reduce_sync_returns_int>);
  static_assert(cudax::__has_synchronous_all_reduce<all_reduce_sync_rejects_void, int*>);
  static_assert(!cudax::__has_synchronous_all_reduce<all_reduce_sync_rejects_void, void*>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
