//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr optional<T>& operator=(const optional<T>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

#if TEST_HAS_EXCEPTIONS()
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool)

  X() = default;
  X(const X&)
  {
    if (throw_now())
    {
      TEST_THROW(6);
    }
  }
  X& operator=(X const&) = default;
};

void test_exceptions()
{
  optional<X> opt{};
  optional<X> opt2(X{});
  assert(static_cast<bool>(opt2) == true);
  try
  {
    X::throw_now() = true;
    opt            = opt2;
    assert(false);
  }
  catch (int i)
  {
    assert(i == 6);
    assert(static_cast<bool>(opt) == false);
  }
}
#endif // TEST_HAS_EXCEPTIONS()

template <class T>
__host__ __device__ constexpr void test()
{
  cuda::std::remove_reference_t<T> val{42};
  cuda::std::remove_reference_t<T> other_val{1337};

  static_assert(cuda::std::is_nothrow_copy_assignable<optional<T>>::value, "");
  // empty copy assigned to empty
  {
    optional<T> opt{};
    const optional<T> input{};
    opt = input;
    assert(!opt.has_value());
    assert(!input.has_value());
  }
  // empty copy assigned to non-empty
  {
    optional<T> opt{val};
    const optional<T> input{};
    opt = input;
    assert(!opt.has_value());
    assert(!input.has_value());
  }
  // non-empty copy assigned to empty
  {
    optional<T> opt{};
    const optional<T> input{val};
    opt = input;
    assert(opt.has_value());
    assert(input.has_value());
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(input.operator->() == opt.operator->());
    }
  }
  // non-empty copy assigned to empty
  {
    optional<T> opt{other_val};
    const optional<T> input{val};
    opt = input;
    assert(opt.has_value());
    assert(input.has_value());
    assert(*opt == val);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      assert(input.operator->() == opt.operator->());
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  test<TrivialTestTypes::TestType>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> opt(3);
    const optional<T> opt2{};
    assert(T::alive() == 1);
    opt = opt2;
    assert(T::alive() == 0);
    assert(!opt2.has_value());
    assert(!opt.has_value());
  }

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
