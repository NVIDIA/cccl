//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class I>
// using iter_rvalue_reference;

#include <cuda/std/iterator>

#include "test_macros.h"

static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<int*>, int&&>);
static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<const int*>, const int&&>);

TEST_HOST_DEVICE void test_undefined_internal() {
  struct A {
    TEST_HOST_DEVICE int& operator*() const;
  };
  static_assert(cuda::std::same_as<cuda::std::iter_rvalue_reference_t<A>, int&&>);
}

int main(int, char**)
{
  return 0;
}
