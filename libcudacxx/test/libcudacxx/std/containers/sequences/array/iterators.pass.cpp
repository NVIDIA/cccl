//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// iterator begin() noexcept;                         // constexpr in C++17
// const_iterator begin() const noexcept;             // constexpr in C++17
// iterator end() noexcept;                           // constexpr in C++17
// const_iterator end() const noexcept;               // constexpr in C++17
//
// reverse_iterator rbegin() noexcept;                // constexpr in C++17
// const_reverse_iterator rbegin() const noexcept;    // constexpr in C++17
// reverse_iterator rend() noexcept;                  // constexpr in C++17
// const_reverse_iterator rend() const noexcept;      // constexpr in C++17
//
// const_iterator cbegin() const noexcept;            // constexpr in C++17
// const_iterator cend() const noexcept;              // constexpr in C++17
// const_reverse_iterator crbegin() const noexcept;   // constexpr in C++17
// const_reverse_iterator crend() const noexcept;     // constexpr in C++17

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ TEST_CONSTEXPR NoDefault(int) {}
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void check_noexcept(T& c)
{
  ASSERT_NOEXCEPT(c.begin());
  ASSERT_NOEXCEPT(c.end());
  ASSERT_NOEXCEPT(c.cbegin());
  ASSERT_NOEXCEPT(c.cend());
  ASSERT_NOEXCEPT(c.rbegin());
  ASSERT_NOEXCEPT(c.rend());
  ASSERT_NOEXCEPT(c.crbegin());
  ASSERT_NOEXCEPT(c.crend());

  const T& cc = c;
  unused(cc);
  ASSERT_NOEXCEPT(cc.begin());
  ASSERT_NOEXCEPT(cc.end());
  ASSERT_NOEXCEPT(cc.rbegin());
  ASSERT_NOEXCEPT(cc.rend());
}

// gcc-7 and gcc-8 are really helpfull here
__host__ __device__
#if TEST_STD_VER >= 2014 && (!defined(TEST_COMPILER_GCC) || __GNUC__ > 8)
  TEST_CONSTEXPR_CXX14
#endif // TEST_STD_VER >= 2014 && (!defined(TEST_COMPILER_GCC) || __GNUC__ > 8)
  bool
  tests()
{
  {
    typedef cuda::std::array<int, 5> C;
    C array = {};
    check_noexcept(array);
    typename C::iterator i       = array.begin();
    typename C::const_iterator j = array.cbegin();
    assert(i == j);
  }
  {
    typedef cuda::std::array<int, 0> C;
    C array = {};
    check_noexcept(array);
    typename C::iterator i       = array.begin();
    typename C::const_iterator j = array.cbegin();
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // seems there are different nullptr's
    assert(i == j);
#else // ^^^ !TEST_COMPILER_CUDACC_BELOW_11_3 ^^^ / vvv TEST_COMPILER_CUDACC_BELOW_11_3 vvv
    assert(i == nullptr);
    assert(j == nullptr);
#endif // TEST_COMPILER_CUDACC_BELOW_11_3
  }

  {
    typedef cuda::std::array<int, 0> C;
    C array = {};
    check_noexcept(array);
    typename C::iterator i       = array.begin();
    typename C::const_iterator j = array.cbegin();

    assert(i == array.end());
    assert(j == array.cend());
  }
  {
    typedef cuda::std::array<int, 1> C;
    C array = {1};
    check_noexcept(array);
    typename C::iterator i = array.begin();
    assert(*i == 1);
    assert(&*i == array.data());
    *i = 99;
    assert(array[0] == 99);
  }
  {
    typedef cuda::std::array<int, 2> C;
    C array = {1, 2};
    check_noexcept(array);
    typename C::iterator i = array.begin();
    assert(*i == 1);
    assert(&*i == array.data());
    *i = 99;
    assert(array[0] == 99);
    assert(array[1] == 2);
  }
  {
    typedef cuda::std::array<double, 3> C;
    C array = {1, 2, 3.5};
    check_noexcept(array);
    typename C::iterator i = array.begin();
    assert(*i == 1);
    assert(&*i == array.data());
    *i = 5.5;
    assert(array[0] == 5.5);
    assert(array[1] == 2.0);
  }
  {
    typedef cuda::std::array<NoDefault, 0> C;
    C array                 = {};
    typename C::iterator ib = array.begin();
    typename C::iterator ie = array.end();
    assert(ib == ie);
  }

#if TEST_STD_VER >= 2014
  { // N3644 testing
    {
      typedef cuda::std::array<int, 5> C;
      C::iterator ii1{}, ii2{};
      C::iterator ii4 = ii1;
      C::const_iterator cii{};
      assert(ii1 == ii2);
      assert(ii1 == ii4);
      static_assert(cuda::std::is_same_v<decltype(ii1), int*>, "");
      static_assert(cuda::std::is_same_v<decltype(cii), const int*>, "");
#  if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // old NVCC has issues comparing int*
                                                                               // with const int*
      assert(ii1 == cii);
#  else // ^^^ !TEST_COMPILER_CUDACC_BELOW_11_3 ^^^ / vvv TEST_COMPILER_CUDACC_BELOW_11_3 vvv
      assert(ii1 == nullptr);
      assert(cii == nullptr);
#  endif // TEST_COMPILER_CUDACC_BELOW_11_3

      assert(!(ii1 != ii2));
#  if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // old NVCC has issues comparing int*
                                                                               // with const int*
      assert(!(ii1 != cii));
#  endif // TEST_COMPILER_CUDACC_BELOW_11_3

      C c = {};
      check_noexcept(c);
      assert(c.begin() == cuda::std::begin(c));
      assert(c.cbegin() == cuda::std::cbegin(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(c.rbegin() == cuda::std::rbegin(c));
        assert(c.crbegin() == cuda::std::crbegin(c));
      }
      assert(c.end() == cuda::std::end(c));
      assert(c.cend() == cuda::std::cend(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(c.rend() == cuda::std::rend(c));
        assert(c.crend() == cuda::std::crend(c));
      }

      assert(cuda::std::begin(c) != cuda::std::end(c));
      assert(cuda::std::cbegin(c) != cuda::std::cend(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(cuda::std::rbegin(c) != cuda::std::rend(c));
        assert(cuda::std::crbegin(c) != cuda::std::crend(c));
      }
    }
    {
      typedef cuda::std::array<int, 0> C;
      C::iterator ii1{}, ii2{};
      C::iterator ii4 = ii1;
      C::const_iterator cii{};
      assert(ii1 == ii2);
      assert(ii1 == ii4);

      assert(!(ii1 != ii2));

#  ifndef TEST_COMPILER_CUDACC_BELOW_11_3 // old NVCC has issues comparing int* with const int*
      assert((ii1 == cii));
      assert((cii == ii1));
      assert(!(ii1 != cii));
      assert(!(cii != ii1));
#  else // ^^^ !TEST_COMPILER_CUDACC_BELOW_11_3 ^^^ / vvv TEST_COMPILER_CUDACC_BELOW_11_3 vvv
      assert(ii1 == nullptr);
      assert(cii == nullptr);
#  endif // TEST_COMPILER_CUDACC_BELOW_11_3
         // This breaks NVCCs constexpr evaluator
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(!(ii1 < cii));
        assert(!(cii < ii1));
        assert((ii1 <= cii));
        assert((cii <= ii1));
        assert(!(ii1 > cii));
        assert(!(cii > ii1));
        assert((ii1 >= cii));
        assert((cii >= ii1));
      }
      assert(cii - ii1 == 0);
      assert(ii1 - cii == 0);

      C c = {};
      check_noexcept(c);
      assert(c.begin() == cuda::std::begin(c));
      assert(c.cbegin() == cuda::std::cbegin(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(c.rbegin() == cuda::std::rbegin(c));
        assert(c.crbegin() == cuda::std::crbegin(c));
      }
      assert(c.end() == cuda::std::end(c));
      assert(c.cend() == cuda::std::cend(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(c.rend() == cuda::std::rend(c));
        assert(c.crend() == cuda::std::crend(c));
      }

      assert(cuda::std::begin(c) == cuda::std::end(c));
      assert(cuda::std::cbegin(c) == cuda::std::cend(c));
#  if TEST_STD_VER < 2017
      if (!TEST_IS_CONSTANT_EVALUATED())
#  endif // TEST_STD_VER < 2017
      {
        assert(cuda::std::rbegin(c) == cuda::std::rend(c));
        assert(cuda::std::crbegin(c) == cuda::std::crend(c));
      }
    }
  }
#endif
  return true;
}

int main(int, char**)
{
  tests();
#ifndef TEST_COMPILER_ICC
#  if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) \
    && (!defined(TEST_COMPILER_GCC) || __GNUC__ > 8)
  static_assert(tests(), "");
#  endif // TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#endif // TEST_COMPILER_ICC
  return 0;
}
