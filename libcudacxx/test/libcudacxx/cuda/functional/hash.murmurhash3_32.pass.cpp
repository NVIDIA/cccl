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

static_assert(cuda::hash_algorithm::murmurhash3_x86_128 != cuda::hash_algorithm::murmurhash3_x64_128);

TEST_FUNC void test()
{
  hash_test<cuda::hash_algorithm::murmurhash3_32> murmurhash3_32_test;

  murmurhash3_32_test(static_cast<char>(0), 1364076727u, 0);
  murmurhash3_32_test(static_cast<char>(42), 338914844u, 0);
  murmurhash3_32_test(static_cast<char>(0), 3712240066u, 42);
  murmurhash3_32_test(static_cast<cuda::std::uint16_t>(42), 2959774506u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int32_t>(0), 593689054u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int32_t>(0), 933211791u, 42);
  murmurhash3_32_test(static_cast<cuda::std::int32_t>(42), 3160117731u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int32_t>(123456789), 3206620847u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int64_t>(0), 1669671676u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int64_t>(0), 2624043101u, 42);
  murmurhash3_32_test(static_cast<cuda::std::int64_t>(42), 1871679806u, 0);
  murmurhash3_32_test(static_cast<cuda::std::int64_t>(123456789), 690028081u, 0);
#if _CCCL_HAS_INT128()
  murmurhash3_32_test(static_cast<__int128_t>(123456789), 2191144977u, 0);
#endif
  murmurhash3_32_test(large_key<32>(123456789), 2555553099u, 0);
}

int main(int, char**)
{
  test();
  return 0;
}
