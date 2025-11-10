//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__random_>
#if !_CCCL_COMPILER(NVRTC)
#  include <cstdint>
#  include <random>
#endif // !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  static_assert(noexcept(cuda::std::seed_seq{}));
  cuda::std::seed_seq seq{1, 2, 3, 4, 5};
  cuda::std::uint32_t out[5] = {};
  seq.generate(cuda::std::begin(out), cuda::std::end(out));
  static_assert(cuda::std::is_void_v<decltype(seq.generate(cuda::std::begin(out), cuda::std::end(out)))>);
  return true;
}

#if !_CCCL_COMPILER(NVRTC)
void test_against_std()
{
  cuda::std::size_t n                                     = 100;
  std::vector<std::vector<cuda::std::uint32_t>> sequences = {
    {1, 2, 3, 4, 5},
    {42, 43, 44, 45, 46, 47, 48, 49, 50},
    {123456789, 987654321, 555555555, 333333333, 111111111},
    {cuda::std::numeric_limits<cuda::std::uint32_t>::max()},
    {0}};
  for (const auto& seq_values : sequences)
  {
    cuda::std::seed_seq cuda_seq{seq_values.data(), seq_values.data() + seq_values.size()};
    std::seed_seq std_seq(seq_values.begin(), seq_values.end());

    std::vector<cuda::std::uint32_t> cuda_output(n);
    std::vector<cuda::std::uint32_t> std_output(n);

    cuda_seq.generate(cuda_output.data(), cuda_output.data() + n);
    std_seq.generate(std_output.data(), std_output.data() + n);
    assert(cuda_output == std_output);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();
// Constexpr new/delete works from C++20 onwards
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif
  NV_IF_TARGET(NV_IS_HOST, ({ test_against_std(); }));
  return 0;
}
