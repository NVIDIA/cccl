//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>

#if __cpp_lib_format >= 201907L
#  include <format>
#endif // __cpp_lib_format >= 201907L

#include "literal.h"

#if __cpp_lib_format >= 201907L
template <class C>
void test()
{
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_60) == TEST_STRLIT(C, "sm_60"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_61) == TEST_STRLIT(C, "sm_61"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_62) == TEST_STRLIT(C, "sm_62"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_70) == TEST_STRLIT(C, "sm_70"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_75) == TEST_STRLIT(C, "sm_75"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_80) == TEST_STRLIT(C, "sm_80"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_86) == TEST_STRLIT(C, "sm_86"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_87) == TEST_STRLIT(C, "sm_87"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_88) == TEST_STRLIT(C, "sm_88"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_89) == TEST_STRLIT(C, "sm_89"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_90) == TEST_STRLIT(C, "sm_90"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_100) == TEST_STRLIT(C, "sm_100"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_103) == TEST_STRLIT(C, "sm_103"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_110) == TEST_STRLIT(C, "sm_110"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_120) == TEST_STRLIT(C, "sm_120"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_121) == TEST_STRLIT(C, "sm_121"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_90a) == TEST_STRLIT(C, "sm_90a"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_100a) == TEST_STRLIT(C, "sm_100a"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_103a) == TEST_STRLIT(C, "sm_103a"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_110a) == TEST_STRLIT(C, "sm_110a"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_120a) == TEST_STRLIT(C, "sm_120a"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::arch_id::sm_121a) == TEST_STRLIT(C, "sm_121a"));
}

void test()
{
  test<char>();
  test<wchar_t>();
}
#endif // __cpp_lib_format >= 201907L

int main(int, char**)
{
#if __cpp_lib_format >= 201907L
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // __cpp_lib_format >= 201907L
  return 0;
}
