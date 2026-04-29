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

#include "test_macros.h"

struct reduce_policy
{
  int block_threads;
};

template <int BlockThreads, class T>
struct reduce_policy_selector
{
  TEST_FUNC constexpr auto operator()(cuda::arch_id /*arch*/) const -> reduce_policy
  {
    return {BlockThreads / sizeof(T)};
  }
};

struct scan_policy
{
  int block_threads = 1;
};

struct scan_policy_selector
{
  TEST_FUNC constexpr auto operator()(cuda::arch_id /*arch*/) const -> scan_policy
  {
    return {};
  }
};

TEST_FUNC void test()
{
  constexpr int nominal_block_threads = 256;
  constexpr int block_threads         = nominal_block_threads / sizeof(int);

  using env_t =
    decltype(cuda::execution::tune(reduce_policy_selector<nominal_block_threads, int>{}, scan_policy_selector{}));
  using tuning_t        = cuda::std::execution::__query_result_t<env_t, cuda::execution::__get_tuning_t>;
  using reduce_policy_t = cuda::std::execution::__query_result_t<tuning_t, reduce_policy>;
  using scan_policy_t   = cuda::std::execution::__query_result_t<tuning_t, scan_policy>;

  static_assert(reduce_policy_t{}(cuda::arch_id::sm_75).block_threads == block_threads);
  static_assert(scan_policy_t{}(cuda::arch_id::sm_75).block_threads == 1);
}

int main(int, char**)
{
  test();

  return 0;
}
