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

struct test_xxhash32
{
  hash_test<cuda::hash_algorithm::xxhash_32> xxhash32_test;

  TEST_FUNC void operator()()
  {
    xxhash32_test(static_cast<char>(0), 3479547966u, 0);
    xxhash32_test(static_cast<char>(42), 3774771295u, 0);
    xxhash32_test(static_cast<char>(0), 2099223482u, 42);
    xxhash32_test(static_cast<cuda::std::int32_t>(0), 148298089u, 0);
    xxhash32_test(static_cast<cuda::std::int32_t>(0), 2132181312u, 42);
    xxhash32_test(static_cast<cuda::std::int32_t>(42), 1161967057u, 0);
    xxhash32_test(static_cast<cuda::std::int32_t>(123456789), 2987034094u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(0), 3736311059u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(0), 1076387279u, 42);
    xxhash32_test(static_cast<cuda::std::int64_t>(42), 2332451213u, 0);
    xxhash32_test(static_cast<cuda::std::int64_t>(123456789), 1561711919u, 0);
#if _CCCL_HAS_INT128()
    xxhash32_test(static_cast<__int128_t>(123456789), 1846633701u, 0);
#endif
    xxhash32_test(large_key<32>(123456789), 3715432378u, 0);
  }
};

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
static_assert(cuda::hash<cuda::std::uint32_t>{}(0u) == 148298089u);
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()

TEST_FUNC void test()
{
  test_xxhash32{}();
  test_sized_keys<cuda::hash_algorithm::xxhash_32>();
  test_noncopyable_key<cuda::hash_algorithm::xxhash_32>();
  test_empty_span<cuda::hash_algorithm::xxhash_32>(0x02cc5d05u);
}

int main(int, char**)
{
  test();
  return 0;
}
