//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// decltype(auto) operator->() const
//   requires see below;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iterator>
__host__ __device__ constexpr void test_access_5_1()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Iterator iter(buffer);
  using Common = cuda::std::common_iterator<Iterator, sentinel_wrapper<Iterator>>;

  Common common(iter);
  decltype(auto) result = common.operator->();
  static_assert(cuda::std::same_as<decltype(result), Iterator>);
  assert(base(result) == buffer);

  Common const ccommon(iter);
  decltype(auto) cresult = ccommon.operator->();
  static_assert(cuda::std::same_as<decltype(cresult), Iterator>);
  assert(base(cresult) == buffer);
};

template <class Iterator>
__host__ __device__ constexpr void test_access_5_2()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Iterator iter(buffer);
  using Common = cuda::std::common_iterator<Iterator, sentinel_type<int*>>;

  Common common(iter);
  decltype(auto) result = common.operator->();
  static_assert(cuda::std::same_as<decltype(result), int*>);
  assert(result == buffer);

  Common const ccommon(iter);
  decltype(auto) cresult = ccommon.operator->();
  static_assert(cuda::std::same_as<decltype(cresult), int*>);
  assert(cresult == buffer);
};

template <class Iterator>
__host__ __device__ constexpr void test_access_5_3()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  Iterator iter(buffer);
  using Common = cuda::std::common_iterator<Iterator, sentinel_type<int*>>;

  Common common(iter);
  auto proxy            = common.operator->();
  decltype(auto) result = proxy.operator->();
  static_assert(cuda::std::same_as<decltype(result), int const*>);
  assert(result != buffer); // we copied to a temporary proxy
  assert(*result == *buffer);

  Common const ccommon(iter);
  auto cproxy            = ccommon.operator->();
  decltype(auto) cresult = cproxy.operator->();
  static_assert(cuda::std::same_as<decltype(cresult), int const*>);
  assert(cresult != buffer); // we copied to a temporary proxy
  assert(*cresult == *buffer);
};

__host__ __device__ constexpr bool test()
{
  // Case 1: http://eel.is/c++draft/iterators.common#common.iter.access-5.1
  {
    test_access_5_1<contiguous_iterator<int*>>();
    test_access_5_1<int*>();
  }

  // Case 2: http://eel.is/c++draft/iterators.common#common.iter.access-5.2
  {
    test_access_5_2<simple_iterator<int*>>();
    test_access_5_2<cpp17_input_iterator<int*>>();
    // cpp20_input_iterator can't be used with common_iterator because it's not copyable
    test_access_5_2<forward_iterator<int*>>();
    test_access_5_2<bidirectional_iterator<int*>>();
    test_access_5_2<random_access_iterator<int*>>();
  }

  // Case 3: http://eel.is/c++draft/iterators.common#common.iter.access-5.3
  {
    test_access_5_3<value_iterator<int*>>();
    test_access_5_3<void_plus_plus_iterator<int*>>();
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  return 0;
}
