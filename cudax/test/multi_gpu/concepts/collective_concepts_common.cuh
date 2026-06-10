//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX_TEST_MULTI_GPU_COLLECTIVE_CONCEPTS_COMMON_CUH
#define _CUDAX_TEST_MULTI_GPU_COLLECTIVE_CONCEPTS_COMMON_CUH

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include "concepts_common.cuh"

namespace cudax_multi_gpu_concepts
{
struct collective_communicator_model : communicator_model
{
  template <class Tp, class Op>
  void reduce_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op, ::cuda::std::int32_t);
  template <class Tp, class Op>
  void reduce(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op, ::cuda::std::int32_t, ::cuda::stream_ref);

  template <class Tp, class Op>
  void all_reduce_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op);
  template <class Tp, class Op>
  void all_reduce(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, Op, ::cuda::stream_ref);

  template <class Tp>
  void gather_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void gather(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);

  template <class Tp>
  void gather_v_sync(
    group_token_type&,
    Tp*,
    ::cuda::std::size_t,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*,
    ::cuda::std::int32_t);
  template <class Tp>
  void gather_v(
    group_token_type&,
    Tp*,
    ::cuda::std::size_t,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*,
    ::cuda::std::int32_t,
    ::cuda::stream_ref);

  template <class Tp>
  void all_gather_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t);
  template <class Tp>
  void all_gather(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::stream_ref);

  template <class Tp>
  void broadcast_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void broadcast(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t, ::cuda::stream_ref);

  template <class Tp>
  void all_to_all_sync(group_token_type&, Tp*, Tp*, ::cuda::std::size_t);
  template <class Tp>
  void all_to_all(group_token_type&, Tp*, Tp*, ::cuda::std::size_t, ::cuda::stream_ref);

  template <class Tp>
  void all_to_all_v_sync(
    group_token_type&,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*);
  template <class Tp>
  void all_to_all_v(
    group_token_type&,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*,
    Tp*,
    const ::cuda::std::size_t*,
    const ::cuda::std::size_t*,
    ::cuda::stream_ref);
};
} // namespace cudax_multi_gpu_concepts

#endif // _CUDAX_TEST_MULTI_GPU_COLLECTIVE_CONCEPTS_COMMON_CUH
