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
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{0}) == TEST_STRLIT(C, "0"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{60}) == TEST_STRLIT(C, "60"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{61}) == TEST_STRLIT(C, "61"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{62}) == TEST_STRLIT(C, "62"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{70}) == TEST_STRLIT(C, "70"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{75}) == TEST_STRLIT(C, "75"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{80}) == TEST_STRLIT(C, "80"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{86}) == TEST_STRLIT(C, "86"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{87}) == TEST_STRLIT(C, "87"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{88}) == TEST_STRLIT(C, "88"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{89}) == TEST_STRLIT(C, "89"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{90}) == TEST_STRLIT(C, "90"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{100}) == TEST_STRLIT(C, "100"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{103}) == TEST_STRLIT(C, "103"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{110}) == TEST_STRLIT(C, "110"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{120}) == TEST_STRLIT(C, "120"));
  assert(std::format(TEST_STRLIT(C, "{}"), cuda::compute_capability{121}) == TEST_STRLIT(C, "121"));
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
