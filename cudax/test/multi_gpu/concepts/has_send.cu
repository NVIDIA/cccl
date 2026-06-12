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

struct no_send
{
  using native_handle_type = int;
  using group_token_type   = types::group_token;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct send_returns_int : no_send
{
  template <class Tp>
  int send(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::__has_send<types::communicator_model>);
  static_assert(cudax::__has_send<types::communicator_model, long*>);

  static_assert(!cudax::__has_send<no_send>);
  static_assert(!cudax::__has_send<send_returns_int>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
