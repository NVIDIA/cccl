//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The iterators of the contiguous containers of the libstdc++ debug mode are contiguous iterators, and to_address does
// not dereference them, because they verify that they are dereferenceable.

// ADDITIONAL_COMPILE_DEFINITIONS: _GLIBCXX_DEBUG

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/memory>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <deque>
#  include <vector>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

#if _CCCL_HAS_HOST_STD_LIB()
static_assert(cuda::std::contiguous_iterator<std::vector<int>::iterator>);
static_assert(cuda::std::contiguous_iterator<std::vector<int>::const_iterator>);
static_assert(!cuda::std::contiguous_iterator<std::deque<int>::iterator>);

void test()
{
  std::vector<int> vec(3);
  assert(cuda::std::to_address(vec.begin()) == vec.data());
  assert(cuda::std::to_address(vec.end()) == vec.data() + vec.size());
  assert(cuda::std::to_address(vec.cend()) == vec.data() + vec.size());
}
#endif // _CCCL_HAS_HOST_STD_LIB()

int main(int, char**)
{
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_HAS_HOST_STD_LIB()
  return 0;
}
