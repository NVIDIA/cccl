//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <cuda/std/expected>

// template<class F> constexpr auto transform(F&&) &;
// template<class F> constexpr auto transform(F&&) &&;
// template<class F> constexpr auto transform(F&&) const&;
// template<class F> constexpr auto transform(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>

#include "../../types.h"
#include "test_macros.h"

struct LVal
{
  __host__ __device__ constexpr int operator()()
  {
    return 1;
  }
};

struct RefQual
{
  __host__ __device__ constexpr int operator()() &
  {
    return 1;
  }
  int operator()() const&  = delete;
  int operator()() &&      = delete;
  int operator()() const&& = delete;
};

struct CRefQual
{
  int operator()() & = delete;
  __host__ __device__ constexpr int operator()() const&
  {
    return 1;
  }
  int operator()() &&      = delete;
  int operator()() const&& = delete;
};

struct RVRefQual
{
  int operator()() &      = delete;
  int operator()() const& = delete;
  __host__ __device__ constexpr int operator()() &&
  {
    return 1;
  }
  int operator()() const&& = delete;
};

struct RVCRefQual
{
  int operator()() &      = delete;
  int operator()() const& = delete;
  int operator()() &&     = delete;
  __host__ __device__ constexpr int operator()() const&&
  {
    return 1;
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<int, TestError> expected_value{cuda::std::in_place, 1};
  const cuda::std::expected<int, TestError> previous_error{cuda::std::unexpect, 42};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.transform(LVal{}) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 42};
      assert(i.transform(LVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      RefQual l{};
      assert(i.transform(l) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 42};
      RefQual l{};
      assert(i.transform(l) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test const& overload
  {
    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const CRefQual l{};
      assert(i.transform(l) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 42};
      const CRefQual l{};
      assert(i.transform(l) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test && overload
  {
    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.transform(RVRefQual{}) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 42};
      assert(i.transform(RVRefQual{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), cuda::std::expected<int, TestError>);
    }
  }

  // Test const&& overload
  {
    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const RVCRefQual l{};
      assert(i.transform(cuda::std::move(l)) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.transform(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 42};
      const RVCRefQual l{};
      assert(i.transform(cuda::std::move(l)) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.transform(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }
  }
}

struct NonConst
{
  __host__ __device__ constexpr int non_const()
  {
    return 1;
  }
};

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct nvrtc_workaround
{
  NonConst t{};

  template <class T = int>
  __host__ __device__ constexpr int operator()()
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<void, TestError> expect{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.transform(l);
  cuda::std::move(expect).transform(l);
}

struct NoCopy
{
  NoCopy() = default;
  __host__ __device__ constexpr NoCopy(const NoCopy&);
  __host__ __device__ constexpr int operator()()
  {
    return 1;
  }
};

// We need an indirection so the assert does not break the compilation
template <class T>
struct AlwaysFalse
{
  __host__ __device__ constexpr AlwaysFalse()
  {
    assert(false);
  }
};

struct NeverCalled
{
  template <class T = int>
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() const
  {
    return AlwaysFalse<T>{}, cuda::std::expected<int, TestError>{42};
  }
};

__host__ __device__ constexpr bool test()
{
  test_sfinae();
  test_val_types();

  cuda::std::expected<void, TestError> expect{cuda::std::unexpect, 42};
  const auto& cexpect = expect;

  expect.transform(NeverCalled{});
  cuda::std::move(expect).transform(NeverCalled{});
  cexpect.transform(NeverCalled{});
  cuda::std::move(cexpect).transform(NeverCalled{});
  return true;
}

int main(int, char**)
{
  test();
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test(), "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  return 0;
}
