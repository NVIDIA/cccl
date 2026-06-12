//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_TEST_MULTI_GPU_CONCEPTS_COMMON_CUH
#define _CUDAX_TEST_MULTI_GPU_CONCEPTS_COMMON_CUH

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

namespace cudax_multi_gpu_concepts
{
struct group_token
{};

struct synchronous_communicator_model
{
  using native_handle_type = int;
  using group_token_type   = group_token;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct communicator_model : synchronous_communicator_model
{
  template <class Tp>
  void send(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
  template <class Tp>
  void recv(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);
};
} // namespace cudax_multi_gpu_concepts

#endif // _CUDAX_TEST_MULTI_GPU_CONCEPTS_COMMON_CUH
