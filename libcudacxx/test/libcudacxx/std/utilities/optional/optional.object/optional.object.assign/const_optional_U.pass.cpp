//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// From LWG2451:
// template<class U>
//   optional<T>& operator=(const optional<U>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

struct Y1
{
  Y1() = default;
  __host__ __device__ Y1(const int&) {}
  Y1& operator=(const Y1&) = delete;
};

struct Y2
{
  Y2()           = default;
  Y2(const int&) = delete;
  __host__ __device__ Y2& operator=(const int&)
  {
    return *this;
  }
};

struct B
{
  int val_;

  __host__ __device__ constexpr bool operator==(const int& other) const noexcept
  {
    return other == val_;
  }
};
class D : public B
{};

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class T>
struct ConvertibleToReference
{
  T val_;

  __host__ __device__ constexpr operator const T&() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ConvertibleToReference& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};

template <class T>
struct ConvertibleToValue
{
  T val_;

  __host__ __device__ constexpr operator T() const noexcept
  {
    return val_;
  }

  __host__ __device__ friend constexpr bool operator==(const int& lhs, const ConvertibleToValue& rhs) noexcept
  {
    return lhs == rhs.val_;
  }
};
#endif // CCCL_ENABLE_OPTIONAL_REF

template <class T>
struct AssignableFrom
{
  STATIC_MEMBER_VAR(type_constructed, int)
  STATIC_MEMBER_VAR(type_assigned, int)
  STATIC_MEMBER_VAR(int_constructed, int)
  STATIC_MEMBER_VAR(int_assigned, int)

  __host__ __device__ static void reset()
  {
    type_constructed() = int_constructed() = 0;
    type_assigned() = int_assigned() = 0;
  }

  AssignableFrom() = default;

  __host__ __device__ explicit AssignableFrom(T)
  {
    ++type_constructed();
  }
  __host__ __device__ AssignableFrom& operator=(T)
  {
    ++type_assigned();
    return *this;
  }

  __host__ __device__ AssignableFrom(int)
  {
    ++int_constructed();
  }
  __host__ __device__ AssignableFrom& operator=(int)
  {
    ++int_assigned();
    return *this;
  }

private:
  AssignableFrom(AssignableFrom const&)            = delete;
  AssignableFrom& operator=(AssignableFrom const&) = delete;
};

__host__ __device__ void test_with_test_type()
{
  using T = TestTypes::TestType;
  T::reset();
  { // non-empty to empty
    T::reset_constructors();
    optional<T> opt{};
    const optional<int> other(42);
    opt = other;
    assert(T::alive() == 1);
    assert(T::constructed() == 1);
    assert(T::value_constructed() == 1);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == true);
    assert(*other == 42);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(42));
  }
  assert(T::alive() == 0);
  { // non-empty to non-empty
    optional<T> opt(101);
    const optional<int> other(42);
    T::reset_constructors();
    opt = other;
    assert(T::alive() == 1);
    assert(T::constructed() == 0);
    assert(T::assigned() == 1);
    assert(T::value_assigned() == 1);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == true);
    assert(*other == 42);
    assert(static_cast<bool>(opt) == true);
    assert(*opt == T(42));
  }
  assert(T::alive() == 0);
  { // empty to non-empty
    optional<T> opt(101);
    const optional<int> other;
    T::reset_constructors();
    opt = other;
    assert(T::alive() == 0);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 1);
    assert(static_cast<bool>(other) == false);
    assert(static_cast<bool>(opt) == false);
  }
  assert(T::alive() == 0);
  { // empty to empty
    optional<T> opt{};
    const optional<int> other;
    T::reset_constructors();
    opt = other;
    assert(T::alive() == 0);
    assert(T::constructed() == 0);
    assert(T::assigned() == 0);
    assert(T::destroyed() == 0);
    assert(static_cast<bool>(other) == false);
    assert(static_cast<bool>(opt) == false);
  }
  assert(T::alive() == 0);
}

__host__ __device__ __noinline__ void test_ambiguous_assign()
{
  using OptInt = cuda::std::optional<int>;
  {
    using T = AssignableFrom<OptInt const&>;
    const OptInt a(42);
    T::reset();
    {
      cuda::std::optional<T> t;
      t = a;
      assert(T::type_constructed() == 1);
      assert(T::type_assigned() == 0);
      assert(T::int_constructed() == 0);
      assert(T::int_assigned() == 0);
    }
    T::reset();
    {
      cuda::std::optional<T> t(42);
      t = a;
      assert(T::type_constructed() == 0);
      assert(T::type_assigned() == 1);
      assert(T::int_constructed() == 1);
      assert(T::int_assigned() == 0);
    }
    T::reset();
    {
      cuda::std::optional<T> t(42);
      t = cuda::std::move(a);
      assert(T::type_constructed() == 0);
      assert(T::type_assigned() == 1);
      assert(T::int_constructed() == 1);
      assert(T::int_assigned() == 0);
    }
  }
  {
    using T = AssignableFrom<OptInt&>;
    OptInt a(42);
    T::reset();
    {
      cuda::std::optional<T> t;
      t = a;
      assert(T::type_constructed() == 1);
      assert(T::type_assigned() == 0);
      assert(T::int_constructed() == 0);
      assert(T::int_assigned() == 0);
    }
    {
      using Opt = cuda::std::optional<T>;
      static_assert(!cuda::std::is_assignable_v<Opt&, OptInt const&>, "");
    }
  }
}

template <class T, class U>
__host__ __device__ constexpr bool test()
{
  { // empty assigned to empty
    optional<T> opt{};
    const optional<U> input{};
    opt = input;
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // non-empty assigned to empty
    cuda::std::remove_reference_t<U> val{42};
    optional<T> opt{};
    const optional<U> input{val};
    opt = input;
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == 42);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      // optional<U> does not necessarily hold a reference so we cannot use addressof(val)
      assert(cuda::std::addressof(static_cast<T>(*input)) == opt.operator->());
    }
  }
  { // empty assigned to non-empty
    cuda::std::remove_reference_t<T> val{42};
    optional<T> opt{val};
    const optional<U> input{};
    opt = input;
    assert(!input.has_value());
    assert(!opt.has_value());
  }
  { // non-empty assigned to non-empty
    cuda::std::remove_reference_t<U> val{42};
    cuda::std::remove_reference_t<T> other_val{1337};
    optional<T> opt{other_val};
    const optional<U> input{val};
    opt = input;
    assert(input.has_value());
    assert(opt.has_value());
    assert(*opt == 42);
    if constexpr (cuda::std::is_reference_v<T>)
    {
      // optional<U> does not necessarily hold a reference so we cannot use addressof(val)
      assert(cuda::std::addressof(static_cast<T>(*input)) == opt.operator->());
    }
  }

  return true;
}

__host__ __device__ constexpr bool test()
{
  test<int, short>();

#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<B&, D&>();

  test<const int&, ConvertibleToReference<int>&>();
  test<const int&, const ConvertibleToReference<int>&>();

  test<int, int&>();
  test<int, const int&>();

  test<int, ConvertibleToReference<int>&>();
  test<int, const ConvertibleToReference<int>&>();

  test<int, ConvertibleToValue<int>&>();
  test<int, const ConvertibleToValue<int>&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

#if TEST_HAS_EXCEPTIONS()
struct X
{
  STATIC_MEMBER_VAR(throw_now, bool)

  X() = default;
  X(int)
  {
    if (throw_now())
    {
      TEST_THROW(6);
    }
  }
};

void throws_exception()
{
  optional<X> opt{};
  optional<int> opt2(42);
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

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  test_with_test_type();
  test_ambiguous_assign();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (throws_exception();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
