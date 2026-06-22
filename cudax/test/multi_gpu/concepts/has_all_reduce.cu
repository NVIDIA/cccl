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
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/experimental/__multi_gpu/concepts.h>

#include <testing.cuh>

#include "collective_concepts_common.cuh"

namespace
{
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct all_reduce_rejects_void : types::communicator_model
{
  template <class Tp, class Op, ::cuda::std::enable_if_t<!::cuda::std::is_void_v<Tp>, int> = 0>
  void all_reduce(group_guard_type&, Tp*, Tp*, ::cuda::std::size_t, Op, ::cuda::stream_ref);
};

struct all_reduce_returns_int : types::communicator_model
{
  template <class Tp, class Op>
  int all_reduce(group_guard_type&, Tp*, Tp*, ::cuda::std::size_t, Op, ::cuda::stream_ref);
};

_CCCL_END_NV_DIAG_SUPPRESS()
} // namespace

C2H_TEST("__has_all_reduce concept", "[multi_gpu][concepts]")
{
  STATIC_REQUIRE(cudax::__has_all_reduce<types::collective_communicator_model>);
  STATIC_REQUIRE(cudax::__has_all_reduce<types::collective_communicator_model, long*>);
  STATIC_REQUIRE(!cudax::__has_all_reduce<types::communicator_model>);

  STATIC_REQUIRE(!cudax::__has_all_reduce<all_reduce_returns_int>);
  STATIC_REQUIRE(cudax::__has_all_reduce<all_reduce_rejects_void, int*>);
  STATIC_REQUIRE(!cudax::__has_all_reduce<all_reduce_rejects_void, void*>);
}
