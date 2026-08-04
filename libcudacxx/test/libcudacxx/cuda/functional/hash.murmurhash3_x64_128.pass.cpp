//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include "hash_test_helper.h"
#include "test_macros.h"

#if _CCCL_HAS_INT128()

struct test_murmurhash3_x64_128
{
  hash_test<cuda::hash_algorithm::murmurhash3_x64_128> murmurhash3_x64_128_test;

  TEST_FUNC __uint128_t make_result(cuda::std::array<cuda::std::uint64_t, 2> arr)
  {
    return cuda::std::bit_cast<__uint128_t>(arr);
  }

  TEST_FUNC void operator()()
  {
    murmurhash3_x64_128_test(cuda::std::int32_t(0), make_result({14961230494313510588ull, 6383328099726337777ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(9), make_result({1779292183511753683ull, 16298496441448380334ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(42), make_result({2913627637088662735ull, 16344193523890567190ull}), 0);
    murmurhash3_x64_128_test(cuda::std::int32_t(42), make_result({2248879576374326886ull, 18006515275339376488ull}), 42);

    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int32_t, 2>{2, 2}, make_result({12221386834995143465ull, 6690950894782946573ull}), 0);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int32_t, 3>{1, 4, 9},
                             make_result({299140022350411792ull, 9891903873182035274ull}),
                             42);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int32_t, 4>{42, 64, 108, 1024},
                             make_result({4333511168876981289ull, 4659486988434316416ull}),
                             63);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      make_result({3302412811061286680ull, 7070355726356610672ull}),
      1024);

    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int64_t, 2>{2, 2}, make_result({8554944597931919519ull, 14938998000509429729ull}), 0);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int64_t, 3>{1, 4, 9},
                             make_result({13442629947720186435ull, 7061727494178573325ull}),
                             42);
    murmurhash3_x64_128_test(cuda::std::array<cuda::std::int64_t, 4>{42, 64, 108, 1024},
                             make_result({8786399719555989948ull, 14954183901757012458ull}),
                             63);
    murmurhash3_x64_128_test(
      cuda::std::array<cuda::std::int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      make_result({15409921801541329777ull, 10546487400963404004ull}),
      1024);
  }
};

TEST_FUNC void test()
{
  test_murmurhash3_x64_128{}();
}

#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test();
#endif // _CCCL_HAS_INT128()
  return 0;
}
