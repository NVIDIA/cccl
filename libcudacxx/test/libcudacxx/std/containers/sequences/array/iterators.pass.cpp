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
  __host__ __device__ constexpr NoDefault(int) {}
};

template <class T>
__host__ __device__ constexpr void check_noexcept(T& c)
{
  static_assert(noexcept(c.begin()));
  static_assert(noexcept(c.end()));
  static_assert(noexcept(c.cbegin()));
  static_assert(noexcept(c.cend()));
  static_assert(noexcept(c.rbegin()));
  static_assert(noexcept(c.rend()));
  static_assert(noexcept(c.crbegin()));
  static_assert(noexcept(c.crend()));

  const T& cc = c;
  unused(cc);
  static_assert(noexcept(cc.begin()));
  static_assert(noexcept(cc.end()));
  static_assert(noexcept(cc.rbegin()));
  static_assert(noexcept(cc.rend()));
}

// gcc-7 and gcc-8 are really helpful here
__host__ __device__
#if !TEST_COMPILER(GCC, <, 8)
  constexpr
#endif // !TEST_COMPILER(GCC, <, 8)
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
    assert(i == j);
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
      assert(ii1 == cii);

      assert(!(ii1 != ii2));
      assert(!(ii1 != cii));

      C c = {};
      check_noexcept(c);
      assert(c.begin() == cuda::std::begin(c));
      assert(c.cbegin() == cuda::std::cbegin(c));
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(c.rbegin() == cuda::std::rbegin(c));
        assert(c.crbegin() == cuda::std::crbegin(c));
      }
      assert(c.end() == cuda::std::end(c));
      assert(c.cend() == cuda::std::cend(c));
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(c.rend() == cuda::std::rend(c));
        assert(c.crend() == cuda::std::crend(c));
      }

      assert(cuda::std::begin(c) != cuda::std::end(c));
      assert(cuda::std::cbegin(c) != cuda::std::cend(c));
      if (!TEST_IS_CONSTANT_EVALUATED())
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

      assert((ii1 == cii));
      assert((cii == ii1));
      assert(!(ii1 != cii));
      assert(!(cii != ii1));
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
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(c.rbegin() == cuda::std::rbegin(c));
        assert(c.crbegin() == cuda::std::crbegin(c));
      }
      assert(c.end() == cuda::std::end(c));
      assert(c.cend() == cuda::std::cend(c));
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(c.rend() == cuda::std::rend(c));
        assert(c.crend() == cuda::std::crend(c));
      }

      assert(cuda::std::begin(c) == cuda::std::end(c));
      assert(cuda::std::cbegin(c) == cuda::std::cend(c));
      if (!TEST_IS_CONSTANT_EVALUATED())
      {
        assert(cuda::std::rbegin(c) == cuda::std::rend(c));
        assert(cuda::std::crbegin(c) == cuda::std::crend(c));
      }
    }
  }
  return true;
}

int main(int, char**)
{
  tests();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) && !TEST_COMPILER(GCC, <, 8)
  static_assert(tests(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED && !TEST_COMPILER(GCC, <, 8)
  return 0;
}
