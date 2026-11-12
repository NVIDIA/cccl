//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// __has_contiguous_traversal<Iter>
//
// Contiguous iterators have contiguous traversal. That includes the iterators of contiguous standard library
// containers, which only advertise it through iterator_concept since C++20.

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <array>
#  include <deque>
#  include <list>
#  include <string>
#  include <string_view>
#  include <vector>
#endif // !TEST_COMPILER(NVRTC)

static_assert(cuda::std::__has_contiguous_traversal<int*>);
static_assert(cuda::std::__has_contiguous_traversal<const int*>);
static_assert(cuda::std::__has_contiguous_traversal<contiguous_iterator<int*>>);
static_assert(!cuda::std::__has_contiguous_traversal<random_access_iterator<int*>>);
static_assert(!cuda::std::__has_contiguous_traversal<cpp20_input_iterator<int*>>);
static_assert(!cuda::std::__has_contiguous_traversal<cuda::std::reverse_iterator<int*>>);

// Types that are not iterators
static_assert(!cuda::std::__has_contiguous_traversal<void>);
static_assert(!cuda::std::__has_contiguous_traversal<int>);

#if !TEST_COMPILER(NVRTC)
static_assert(cuda::std::__has_contiguous_traversal<std::array<int, 4>::iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::array<int, 4>::const_iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::vector<int>::iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::vector<int>::const_iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::string::iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::string::const_iterator>);
static_assert(cuda::std::__has_contiguous_traversal<std::string_view::iterator>);

static_assert(!cuda::std::__has_contiguous_traversal<std::vector<int>::reverse_iterator>);
static_assert(!cuda::std::__has_contiguous_traversal<std::vector<bool>::iterator>);
static_assert(!cuda::std::__has_contiguous_traversal<std::deque<int>::iterator>);
static_assert(!cuda::std::__has_contiguous_traversal<std::list<int>::iterator>);
#endif // !TEST_COMPILER(NVRTC)

int main(int, char**)
{
  return 0;
}
