//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template<class Iter>
// struct iterator_traits
// {
//   typedef typename Iter::difference_type difference_type;
//   typedef typename Iter::value_type value_type;
//   typedef typename Iter::pointer pointer;
//   typedef typename Iter::reference reference;
//   typedef typename Iter::iterator_category iterator_category;
// };

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif // TEST_COMPILER_MSVC

#if !defined(TEST_COMPILER_NVRTC)
#  include <iterator>
#  include <vector>
#endif // !TEST_COMPILER_NVRTC

struct A
{};

struct test_iterator
{
  typedef int difference_type;
  typedef A value_type;
  typedef A* pointer;
  typedef A& reference;
  typedef cuda::std::forward_iterator_tag iterator_category;
};

#if !defined(TEST_COMPILER_NVRTC)
struct specialized_test_iterator
{};

namespace std
{
template <>
struct iterator_traits<specialized_test_iterator>
{
  using iterator_category = bidirectional_iterator_tag;
  using value_type        = int;
  using difference_type   = unsigned short;
  using pointer           = double*;
  using reference         = long&;
};
} // namespace std

static_assert(!cuda::std::__is_primary_std_template<specialized_test_iterator>::value);
static_assert(cuda::std::__specialized_from_std<specialized_test_iterator>);

#endif // !TEST_COMPILER_NVRTC

int main(int, char**)
{
  {
    typedef cuda::std::iterator_traits<test_iterator> It;
    static_assert((cuda::std::is_same<It::difference_type, int>::value), "");
    static_assert((cuda::std::is_same<It::value_type, A>::value), "");
    static_assert((cuda::std::is_same<It::pointer, A*>::value), "");
    static_assert((cuda::std::is_same<It::reference, A&>::value), "");
    static_assert((cuda::std::is_same<It::iterator_category, cuda::std::forward_iterator_tag>::value), "");
  }

#if !defined(TEST_COMPILER_NVRTC)
  { // std::vector
    typedef cuda::std::iterator_traits<typename std::vector<int>::iterator> It;
    static_assert((cuda::std::is_same<It::difference_type, std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<It::value_type, int>::value), "");
    static_assert((cuda::std::is_same<It::pointer, int*>::value), "");
    static_assert((cuda::std::is_same<It::reference, int&>::value), "");
    static_assert((cuda::std::is_same<It::iterator_category, std::random_access_iterator_tag>::value), "");

    static_assert(cuda::std::__is_cpp17_random_access_iterator<typename std::vector<int>::iterator>::value, "");
  }

  { // specialization of std::iterator_traits
    using It = cuda::std::iterator_traits<specialized_test_iterator>;
    static_assert((cuda::std::is_same<It::difference_type, unsigned short>::value), "");
    static_assert((cuda::std::is_same<It::value_type, int>::value), "");
    static_assert((cuda::std::is_same<It::pointer, double*>::value), "");
    static_assert((cuda::std::is_same<It::reference, long&>::value), "");
    static_assert((cuda::std::is_same<It::iterator_category, cuda::std::bidirectional_iterator_tag>::value), "");
  }
#endif // !TEST_COMPILER_NVRTC

  return 0;
}
