//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class U, class... Args>
//   T& optional<T>::emplace(initializer_list<U> il, Args&&... args);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_VECTOR
#  include <cuda/std/vector>
#endif

#include "test_macros.h"

using cuda::std::optional;

class X
{
  int i_;
  int j_ = 0;
  bool* dtor_called_;

public:
  __host__ __device__ constexpr X(bool& dtor_called)
      : i_(0)
      , dtor_called_(&dtor_called)
  {}
  __host__ __device__ constexpr X(int i, bool& dtor_called)
      : i_(i)
      , dtor_called_(&dtor_called)
  {}
  __host__ __device__ constexpr X(cuda::std::initializer_list<int> il, bool& dtor_called)
      : i_(il.begin()[0])
      , j_(il.begin()[1])
      , dtor_called_(&dtor_called)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~X()
  {
    *dtor_called_ = true;
  }

  __host__ __device__ friend constexpr bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

class Y
{
  int i_;
  int j_ = 0;

public:
  __host__ __device__ constexpr Y()
      : i_(0)
  {}
  __host__ __device__ constexpr Y(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr Y(cuda::std::initializer_list<int> il)
      : i_(il.begin()[0])
      , j_(il.begin()[1])
  {}

  __host__ __device__ friend constexpr bool operator==(const Y& x, const Y& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool check_X()
{
  bool dtor_called = false;
  X x(dtor_called);
  optional<X> opt(x);
  assert(dtor_called == false);
  auto& v = opt.emplace({1, 2}, dtor_called);
  static_assert(cuda::std::is_same_v<X&, decltype(v)>, "");
  assert(dtor_called);
  assert(*opt == X({1, 2}, dtor_called));
  assert(&v == &*opt);
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool check_Y()
{
  optional<Y> opt;
  auto& v = opt.emplace({1, 2});
  static_assert(cuda::std::is_same_v<Y&, decltype(v)>, "");
  assert(static_cast<bool>(opt) == true);
  assert(*opt == Y({1, 2}));
  assert(&v == &*opt);
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
class Z
{
  int i_;
  int j_ = 0;

public:
  STATIC_MEMBER_VAR(dtor_called, bool);
  Z()
      : i_(0)
  {}
  Z(int i)
      : i_(i)
  {}
  Z(cuda::std::initializer_list<int> il)
      : i_(il.begin()[0])
      , j_(il.begin()[1])
  {
    TEST_THROW(6);
  }
  ~Z()
  {
    dtor_called() = true;
  }

  friend bool operator==(const Z& x, const Z& y)
  {
    return x.i_ == y.i_ && x.j_ == y.j_;
  }
};
void test_exceptions()
{
  Z::dtor_called() = false;
  Z z;
  optional<Z> opt(z);
  try
  {
    assert(static_cast<bool>(opt) == true);
    assert(Z::dtor_called() == false);
    auto& v = opt.emplace({1, 2});
    static_assert(cuda::std::is_same_v<Z&, decltype(v)>, "");
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
    assert(static_cast<bool>(opt) == false);
    assert(Z::dtor_called() == true);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    check_X();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(check_X());
#endif
  }
#ifdef _LIBCUDACXX_HAS_VECTOR
  {
    optional<cuda::std::vector<int>> opt;
    auto& v = opt.emplace({1, 2, 3}, cuda::std::allocator<int>());
    static_assert(cuda::std::is_same_v<cuda::std::vector<int>&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(*opt == cuda::std::vector<int>({1, 2, 3}));
    assert(&v == &*opt);
  }
#endif
  {
    check_Y();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(check_Y());
#endif
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
