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
#include <cuda/std/cstdint>

#include "hash_test_helper.h"
#include "test_macros.h"

TEST_FUNC void test()
{
  static_assert(cuda::std::is_same_v<cuda::hash<int>, cuda::hash<int, cuda::hash_algorithm::xxhash_64>>);

  hash_test<cuda::hash_algorithm::xxhash_64> xxhash64_test;

  xxhash64_test(static_cast<char>(0), 16804241149081757544ull, 0);
  xxhash64_test(static_cast<char>(42), 765293966243412708ull, 0);
  xxhash64_test(static_cast<char>(0), 9486749600008296231ull, 42);
  xxhash64_test(static_cast<cuda::std::int32_t>(0), 4246796580750024372ull, 0);
  xxhash64_test(static_cast<cuda::std::int32_t>(0), 3614696996920510707ull, 42);
  xxhash64_test(static_cast<cuda::std::int32_t>(42), 15516826743637085169ull, 0);
  xxhash64_test(static_cast<cuda::std::int32_t>(123456789), 9462334144942111946ull, 0);
  xxhash64_test(static_cast<cuda::std::int64_t>(0), 3803688792395291579ull, 0);
  xxhash64_test(static_cast<cuda::std::int64_t>(0), 13194218611613725804ull, 42);
  xxhash64_test(static_cast<cuda::std::int64_t>(42), 13066772586158965587ull, 0);
  xxhash64_test(static_cast<cuda::std::int64_t>(123456789), 14662639848940634189ull, 0);
#if _CCCL_HAS_INT128()
  xxhash64_test(static_cast<__int128_t>(123456789), 7986913354431084250ull, 0);
#endif
  xxhash64_test(large_key<32>(123456789), 2031761887105658523ull, 0);
}

int main(int, char**)
{
  test();
  return 0;
}
