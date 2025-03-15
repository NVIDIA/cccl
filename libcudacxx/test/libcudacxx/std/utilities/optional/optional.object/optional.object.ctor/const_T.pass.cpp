//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/optional>

// constexpr optional(const T& v);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class T>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_convertible_v<const T&, optional<T>> == cuda::std::is_convertible_v<const T&, T>, "");
  {
    cuda::std::remove_reference_t<T> input{42};
    optional<T> opt{input};
    assert(opt.has_value());
    assert(*opt == input);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(cuda::std::addressof(input) == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<double>();
  test<const int>();

  test<ConstexprTestTypes::TestType>();
  test<ExplicitConstexprTestTypes::TestType>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<const int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

#if TEST_HAS_EXCEPTIONS()
struct Z
{
  Z(int) {}
  Z(const Z&)
  {
    TEST_THROW(6);
  }
};

void test_exceptions()
{
  typedef Z T;
  try
  {
    const T t(3);
    optional<T> opt(t);
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test(), "");

  {
    using T = TestTypes::TestType;
    T::reset();
    const T t(3);
    optional<T> opt = t;
    assert(T::alive() == 2);
    assert(T::copy_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    using T = ExplicitTestTypes::TestType;
    static_assert(!cuda::std::is_convertible<T const&, optional<T>>::value, "");
    T::reset();
    const T t(3);
    optional<T> opt(t);
    assert(T::alive() == 2);
    assert(T::copy_constructed() == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
