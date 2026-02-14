//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/tune.h>

struct get_reduce_tuning_query_t
{};

template <class Derived>
struct reduce_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_reduce_tuning_query_t&) const noexcept -> Derived
  {
    return static_cast<const Derived&>(*this);
  }
};

template <int BlockThreads>
struct reduce : reduce_tuning<reduce<BlockThreads>>
{
  template <class T>
  struct type
  {
    struct max_policy
    {
      struct reduce_policy
      {
        static constexpr int block_threads = BlockThreads / sizeof(T);
      };
    };
  };
};

struct get_scan_tuning_query_t
{};

struct scan_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_scan_tuning_query_t&) const noexcept
  {
    return *this;
  }

  struct type
  {
    struct max_policy
    {
      struct reduce_policy
      {
        static constexpr int block_threads = 1;
      };
    };
  };
};

__host__ __device__ void test()
{
  constexpr int nominal_block_threads = 256;
  constexpr int block_threads         = nominal_block_threads / sizeof(int);

  using env_t           = decltype(cuda::execution::__tune(reduce<nominal_block_threads>{}, scan_tuning{}));
  using tuning_t        = cuda::std::execution::__query_result_t<env_t, cuda::execution::__get_tuning_t>;
  using reduce_tuning_t = cuda::std::execution::__query_result_t<tuning_t, get_reduce_tuning_query_t>;
  using scan_tuning_t   = cuda::std::execution::__query_result_t<tuning_t, get_scan_tuning_query_t>;
  using reduce_policy_t = reduce_tuning_t::type<int>;
  using scan_policy_t   = scan_tuning_t::type;

  static_assert(reduce_policy_t::max_policy::reduce_policy::block_threads == block_threads);
  static_assert(scan_policy_t::max_policy::reduce_policy::block_threads == 1);
}

int main(int, char**)
{
  test();

  return 0;
}
