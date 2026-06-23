//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__device/all_devices.h>
#include <cuda/memory_pool>
#include <cuda/std/type_traits>

#include <testing.cuh>
#include <utility.cuh>

namespace
{
namespace test
{
template <typename PoolType>
inline constexpr bool is_pinned_memory_pool_type =
#if _CCCL_CTK_AT_LEAST(12, 9)
  cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::pinned_memory_pool>
  || cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::pinned_memory_pool_ref>
  || cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::shared_pinned_memory_pool>;
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 9) ^^^ / vvv _CCCL_CTK_BELOW(12, 9) vvv
  false;
#endif // ^^^ _CCCL_CTK_BELOW(12, 9) ^^^

template <typename PoolType>
inline constexpr bool is_managed_memory_pool_type =
#if _CCCL_CTK_AT_LEAST(13, 0)
  cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::managed_memory_pool>
  || cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::managed_memory_pool_ref>
  || cuda::std::is_same_v<cuda::std::remove_cv_t<PoolType>, cuda::shared_managed_memory_pool>;
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  false;
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

template <typename PoolType>
inline constexpr bool is_memory_pool_type =
  cuda::std::is_base_of_v<cuda::__memory_pool_base, cuda::std::remove_cv_t<PoolType>>;

template <typename PoolType>
void skip_if_unsupported_memory_pool()
{
  static_assert(is_memory_pool_type<PoolType>,
                "skip_if_unsupported_memory_pool requires a memory pool, memory pool ref, or shared memory pool type");

  if (!cuda::device_attributes::memory_pools_supported(cuda::devices[0]))
  {
    SKIP("stream-ordered memory pools are not supported");
  }
#if _CCCL_CTK_AT_LEAST(12, 9)
  if constexpr (is_pinned_memory_pool_type<PoolType>)
  {
    if (!cuda::__is_host_memory_pool_supported())
    {
      SKIP("host memory pools are not supported");
    }
  }
#endif // _CCCL_CTK_AT_LEAST(12, 9)
#if _CCCL_CTK_AT_LEAST(13, 0)
  if constexpr (is_managed_memory_pool_type<PoolType>)
  {
    if (!cuda::device_attributes::concurrent_managed_access(cuda::devices[0]))
    {
      SKIP("managed memory pools are not supported");
    }
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)
}
} // namespace test
} // namespace

template <typename ResourceType>
void test_deallocate_async([[maybe_unused]] ResourceType& resource)
{
  /* disable until we move the launch API to libcudacxx
  cudax::stream stream{cuda::device_ref{0}};
  test::pinned<int> i(0);
  cuda::atomic_ref atomic_i(*i);

  int* allocation = static_cast<int*>(resource.allocate_sync(sizeof(int)));

  cudax::launch(stream, test::one_thread_dims, test::spin_until_80{}, i.get());
  cudax::launch(stream, test::one_thread_dims, test::assign_42{}, allocation);
  cudax::launch(stream, test::one_thread_dims, test::verify_42{}, allocation);

  resource.deallocate(stream, allocation, sizeof(int));

  atomic_i.store(80);
  stream.sync();
  */
}
