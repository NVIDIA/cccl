//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

// TODO: Remove pointless comparison suppression when compiler fixes short-circuiting

#include <cuda/pipeline>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

template <typename T_size, typename T_thread_rank>
struct single_thread_test_group
{
  __host__ __device__ void sync() const {}
  __host__ __device__ T_size size() const
  {
    return 1;
  }
  __host__ __device__ T_thread_rank thread_rank() const
  {
    return 0;
  }
};

template <cuda::thread_scope Scope, typename T_size, typename T_thread_rank>
__host__ __device__ void test_fully_specialized()
{
  auto group = single_thread_test_group<T_size, T_thread_rank>{};

  cuda::pipeline_shared_state<Scope, 1> shared_state;
  auto pipe_partitioned_implicit          = make_pipeline(group, &shared_state, 1);
  auto pipe_partitioned_explicit_consumer = make_pipeline(group, &shared_state, cuda::pipeline_role::consumer);
  auto pipe_partitioned_explicit_producer = make_pipeline(group, &shared_state, cuda::pipeline_role::producer);
  auto pipe_unified                       = make_pipeline(group, &shared_state);

  char src[2] = {1, 2};
  char dst[2] = {0, 0};

  pipe_unified.producer_acquire();
  memcpy_async(group, &dst[0], &src[0], 1, pipe_unified);
  memcpy_async(group, &dst[1], &src[1], cuda::aligned_size_t<1>(1), pipe_unified);
  pipe_unified.producer_commit();
  pipe_unified.consumer_wait();
  assert(dst[0] == 1);
  assert(dst[1] == 2);
  pipe_unified.consumer_release();
}

template <cuda::thread_scope Scope, typename T_size>
__host__ __device__ void test_select_thread_rank_type()
{
  test_fully_specialized<Scope, T_size, bool>();
  test_fully_specialized<Scope, T_size, int8_t>();
  test_fully_specialized<Scope, T_size, int16_t>();
  test_fully_specialized<Scope, T_size, int32_t>();
  test_fully_specialized<Scope, T_size, int64_t>();
  test_fully_specialized<Scope, T_size, uint8_t>();
  test_fully_specialized<Scope, T_size, uint16_t>();
  test_fully_specialized<Scope, T_size, uint32_t>();
  test_fully_specialized<Scope, T_size, uint64_t>();
}

template <cuda::thread_scope Scope>
__host__ __device__ void test_select_size_type()
{
  test_select_thread_rank_type<Scope, bool>();
  test_select_thread_rank_type<Scope, int8_t>();
  test_select_thread_rank_type<Scope, int16_t>();
  test_select_thread_rank_type<Scope, int32_t>();
  test_select_thread_rank_type<Scope, int64_t>();
  test_select_thread_rank_type<Scope, uint8_t>();
  test_select_thread_rank_type<Scope, uint16_t>();
  test_select_thread_rank_type<Scope, uint32_t>();
  test_select_thread_rank_type<Scope, uint64_t>();
}
