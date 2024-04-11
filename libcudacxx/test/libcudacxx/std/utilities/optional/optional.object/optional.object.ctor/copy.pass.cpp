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

// constexpr optional(const optional<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class T, class... InitArgs>
__host__ __device__ void test(InitArgs&&... args)
{
  const optional<T> rhs(cuda::std::forward<InitArgs>(args)...);
  bool rhs_engaged = static_cast<bool>(rhs);
  optional<T> lhs  = rhs;
  assert(static_cast<bool>(lhs) == rhs_engaged);
  if (rhs_engaged)
  {
    assert(*lhs == *rhs);
  }
}

template <class T, class... InitArgs>
__host__ __device__ constexpr bool constexpr_test(InitArgs&&... args)
{
  static_assert(cuda::std::is_trivially_copy_constructible_v<T>, ""); // requirement
  const optional<T> rhs(cuda::std::forward<InitArgs>(args)...);
  optional<T> lhs = rhs;
  return (lhs.has_value() == rhs.has_value()) && (lhs.has_value() ? *lhs == *rhs : true);
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Z
{
  Z()
      : count(0)
  {}
  Z(Z const& o)
      : count(o.count + 1)
  {
    if (count == 2)
    {
      TEST_THROW(6);
    }
  }
  int count;
};

void test_throwing_ctor()
{
  const Z z;
  const optional<Z> rhs(z);
  try
  {
    optional<Z> lhs(rhs);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

template <class T, class... InitArgs>
__host__ __device__ void test_ref(InitArgs&&... args)
{
  const optional<T> rhs(cuda::std::forward<InitArgs>(args)...);
  bool rhs_engaged = static_cast<bool>(rhs);
  optional<T> lhs  = rhs;
  assert(static_cast<bool>(lhs) == rhs_engaged);
  if (rhs_engaged)
  {
    assert(&(*lhs) == &(*rhs));
  }
}

__host__ __device__ void test_reference_extension()
{
#if defined(_LIBCPP_VERSION) && 0 // FIXME these extensions are currently disabled.
  using T = TestTypes::TestType;
  T::reset();
  {
    T t;
    T::reset_constructors();
    test_ref<T&>();
    test_ref<T&>(t);
    assert(T::alive() == 1);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
  }
  assert(T::destroyed() == 1);
  assert(T::alive() == 0);
  {
    T t;
    const T& ct = t;
    T::reset_constructors();
    test_ref<T const&>();
    test_ref<T const&>(t);
    test_ref<T const&>(ct);
    assert(T::alive() == 1);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
  }
  assert(T::alive() == 0);
  assert(T::destroyed() == 1);
  {
    static_assert(!cuda::std::is_copy_constructible<cuda::std::optional<T&&>>::value, "");
    static_assert(!cuda::std::is_copy_constructible<cuda::std::optional<T const&&>>::value, "");
  }
#endif
}

int main(int, char**)
{
  test<int>();
  test<int>(3);
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(constexpr_test<int>(), "");
  static_assert(constexpr_test<int>(3), "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

  {
    const optional<const int> o(42);
    optional<const int> o2(o);
    assert(*o2 == 42);
  }
  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs;
    assert(T::alive() == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value() == false);
    assert(T::alive() == 0);
  }
  TestTypes::TestType::reset();
  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs(42);
    assert(T::alive() == 1);
    assert(T::value_constructed() == 1);
    assert(T::copy_constructed() == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value());
    assert(T::copy_constructed() == 1);
    assert(T::alive() == 2);
  }
  TestTypes::TestType::reset();
  {
    using namespace ConstexprTestTypes;
    test<TestType>();
    test<TestType>(42);
  }
  {
    using namespace TrivialTestTypes;
    test<TestType>();
    test<TestType>(42);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    NV_IF_TARGET(NV_IS_HOST, (test_throwing_ctor();))
  }
#endif // !TEST_HAS_NO_EXCEPTIONS
  {
    test_reference_extension();
  }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr cuda::std::optional<int> o1{4};
    constexpr cuda::std::optional<int> o2 = o1;
    static_assert(*o2 == 4, "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

  return 0;
}
