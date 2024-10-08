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

// template <class... Args> T& optional<T>::emplace(Args&&... args);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class T>
__host__ __device__ constexpr bool test_one_arg()
{
  using Opt = cuda::std::optional<T>;
  {
    Opt opt;
    auto& v = opt.emplace();
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(0));
    assert(&v == &*opt);
  }
  {
    Opt opt;
    auto& v = opt.emplace(1);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(1));
    assert(&v == &*opt);
  }
  {
    Opt opt(2);
    auto& v = opt.emplace();
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(0));
    assert(&v == &*opt);
  }
  {
    Opt opt(2);
    auto& v = opt.emplace(1);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(1));
    assert(&v == &*opt);
  }
  return true;
}

template <class T>
__host__ __device__ constexpr bool test_multi_arg()
{
  test_one_arg<T>();
  using Opt = cuda::std::optional<T>;
  {
    Opt opt;
    auto& v = opt.emplace(101, 41);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(v == T(101, 41));
    assert(*opt == T(101, 41));
  }
  {
    Opt opt;
    auto& v = opt.emplace({1, 2, 3, 4});
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(v == T(4)); // T sets its value to the size of the init list
    assert(*opt == T(4));
  }
  {
    Opt opt;
    auto& v = opt.emplace({1, 2, 3, 4, 5}, 6);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(static_cast<bool>(opt) == true);
    assert(v == T(5)); // T sets its value to the size of the init list
    assert(*opt == T(5)); // T sets its value to the size of the init list
  }
  return true;
}

template <class T>
__host__ __device__ void test_on_test_type()
{
  T::reset();
  optional<T> opt;
  assert(T::alive() == 0);
  {
    T::reset_constructors();
    auto& v = opt.emplace();
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::default_constructed() == 1);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T());
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace();
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::default_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T());
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace(101);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(101));
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace(-10, 99);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(-10, 99));
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace(-10, 99);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(-10, 99));
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace({-10, 99, 42, 1});
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(4)); // size of the initializer list
    assert(&v == &*opt);
  }
  {
    T::reset_constructors();
    auto& v = opt.emplace({-10, 99, 42, 1}, 42);
    static_assert(cuda::std::is_same_v<T&, decltype(v)>, "");
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(4)); // size of the initializer list
    assert(&v == &*opt);
  }
}

__host__ __device__ constexpr bool test_empty_emplace()
{
  optional<const int> opt;
  auto& v = opt.emplace(42);
  static_assert(cuda::std::is_same_v<const int&, decltype(v)>, "");
  assert(*opt == 42);
  assert(v == 42);
  opt.emplace();
  assert(*opt == 0);
  return true;
}
#ifndef TEST_HAS_NO_EXCEPTIONS
struct Y
{
  STATIC_MEMBER_VAR(dtor_called, bool);
  Y() = default;
  Y(int)
  {
    TEST_THROW(6);
  }
  ~Y()
  {
    dtor_called() = true;
  }
};

void test_exceptions()
{
  Y::dtor_called() = true;
  Y y;
  optional<Y> opt(y);
  try
  {
    assert(static_cast<bool>(opt) == true);
    assert(Y::dtor_called() == false);
    auto& v = opt.emplace(1);
    static_assert(cuda::std::is_same_v<Y&, decltype(v)>, "");
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
    assert(static_cast<bool>(opt) == false);
    assert(Y::dtor_called() == true);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  {
    test_on_test_type<TestTypes::TestType>();
    test_on_test_type<ExplicitTestTypes::TestType>();
  }
  {
    using T = int;
    test_one_arg<T>();
    test_one_arg<const T>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_one_arg<T>());
    static_assert(test_one_arg<const T>());
#endif
  }
  {
    using T = ConstexprTestTypes::TestType;
    test_multi_arg<T>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_multi_arg<T>());
#endif
  }
  {
    using T = ExplicitConstexprTestTypes::TestType;
    test_multi_arg<T>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_multi_arg<T>());
#endif
  }
  {
    using T = TrivialTestTypes::TestType;
    test_multi_arg<T>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_multi_arg<T>());
#endif
  }
  {
    using T = ExplicitTrivialTestTypes::TestType;
    test_multi_arg<T>();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_multi_arg<T>());
#endif
  }
  {
    test_empty_emplace();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
    static_assert(test_empty_emplace());
#endif
  }

  return 0;
}
