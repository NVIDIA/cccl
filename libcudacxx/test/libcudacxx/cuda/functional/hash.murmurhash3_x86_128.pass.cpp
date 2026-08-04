//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvcc-12.0

#include <cuda/functional>

#include "hash_test_helper.h"

struct test_murmurhash3_x86_128
{
  hash_test<cuda::hash_algorithm::murmurhash3_x86_128> murmurhash3_x86_128_test;

  TEST_FUNC constexpr cuda::std::array<cuda::std::uint32_t, 4> make_result(cuda::std::array<cuda::std::uint32_t, 4> arr)
  {
    return arr;
  }

  TEST_FUNC void operator()()
  {
    murmurhash3_x86_128_test(
      cuda::std::int32_t(0), make_result({3422973727u, 2656139328u, 2656139328u, 2656139328u}), 0);
    murmurhash3_x86_128_test(cuda::std::int32_t(9), make_result({2808089785u, 314604614u, 314604614u, 314604614u}), 0);
    murmurhash3_x86_128_test(
      cuda::std::int32_t(42), make_result({3611919118u, 1962256489u, 1962256489u, 1962256489u}), 0);
    murmurhash3_x86_128_test(cuda::std::int32_t(42), make_result({3399017053u, 732469929u, 732469929u, 732469929u}), 42);

    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int32_t, 2>{2, 2},
                             make_result({1234494082u, 1431451587u, 431049201u, 431049201u}),
                             0);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int32_t, 3>{1, 4, 9},
                             make_result({2516796247u, 2757675829u, 778406919u, 2453259553u}),
                             42);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int32_t, 4>{42, 64, 108, 1024},
                             make_result({2686265656u, 591236665u, 3797082165u, 2731908938u}),
                             63);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      make_result({3918256832u, 4205523739u, 1707810111u, 1625952473u}),
      1024);

    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int64_t, 2>{2, 2},
                             make_result({3811075945u, 727160712u, 3510740342u, 235225510u}),
                             0);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int64_t, 3>{1, 4, 9},
                             make_result({2817194959u, 206796677u, 3391242768u, 248681098u}),
                             42);
    murmurhash3_x86_128_test(cuda::std::array<cuda::std::int64_t, 4>{42, 64, 108, 1024},
                             make_result({2335912146u, 1566515912u, 760710030u, 452077451u}),
                             63);
    murmurhash3_x86_128_test(
      cuda::std::array<cuda::std::int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      make_result({1101169764u, 1758958147u, 2406511780u, 2903571412u}),
      1024);
  }
};

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
static_assert((cuda::hash<cuda::std::uint32_t, cuda::hash_algorithm::murmurhash3_x86_128>{}(0u)
               == cuda::std::array<cuda::std::uint32_t, 4>{3422973727u, 2656139328u, 2656139328u, 2656139328u}));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

TEST_FUNC void test()
{
  test_murmurhash3_x86_128{}();
  test_sized_keys<cuda::hash_algorithm::murmurhash3_x86_128>();
  test_noncopyable_key<cuda::hash_algorithm::murmurhash3_x86_128>();
  test_empty_span<cuda::hash_algorithm::murmurhash3_x86_128>(cuda::std::array<cuda::std::uint32_t, 4>{});
}

int main(int, char**)
{
  test();
  return 0;
}
