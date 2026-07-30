//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.tune.h>

#include "test_macros.h"

struct reduce_policy
{
  int threads_per_block;
};

template <int ThreadsPerBlock, class T>
struct reduce_policy_selector
{
  TEST_FUNC constexpr auto operator()(cuda::compute_capability) const -> reduce_policy
  {
    return {ThreadsPerBlock / sizeof(T)};
  }
};

struct scan_policy
{
  int threads_per_block = 1;
};

struct scan_policy_selector
{
  TEST_FUNC constexpr auto operator()(cuda::compute_capability) const -> scan_policy
  {
    return {};
  }
};

TEST_FUNC void test()
{
  constexpr int nominal_threads_per_block = 256;
  constexpr int threads_per_block         = nominal_threads_per_block / sizeof(int);

  using env_t =
    decltype(cuda::execution::tune(reduce_policy_selector<nominal_threads_per_block, int>{}, scan_policy_selector{}));
  using tuning_t        = cuda::std::execution::__query_result_t<env_t, cuda::execution::__get_tuning_t>;
  using reduce_policy_t = cuda::std::execution::__query_result_t<tuning_t, reduce_policy>;
  using scan_policy_t   = cuda::std::execution::__query_result_t<tuning_t, scan_policy>;

  static_assert(reduce_policy_t{}(cuda::compute_capability{7, 5}).threads_per_block == threads_per_block);
  static_assert(scan_policy_t{}(cuda::compute_capability{7, 5}).threads_per_block == 1);
}

int main(int, char**)
{
  test();

  return 0;
}
