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

// template<class F> constexpr auto and_then(F&&) &;
// template<class F> constexpr auto and_then(F&&) &&;
// template<class F> constexpr auto and_then(F&&) const&;
// template<class F> constexpr auto and_then(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/expected>

#include "../../types.h"
#include "test_macros.h"

struct LVal
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()()
  {
    return 1;
  }
};

struct NOLVal
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()()
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct RefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() &
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()() const&  = delete;
  cuda::std::expected<int, TestError> operator()() &&      = delete;
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct CRefQual
{
  cuda::std::expected<int, TestError> operator()() & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() const&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()() &&      = delete;
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct RVRefQual
{
  cuda::std::expected<int, TestError> operator()() &      = delete;
  cuda::std::expected<int, TestError> operator()() const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() &&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct RVCRefQual
{
  cuda::std::expected<int, TestError> operator()() &      = delete;
  cuda::std::expected<int, TestError> operator()() const& = delete;
  cuda::std::expected<int, TestError> operator()() &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() const&&
  {
    return 1;
  }
};

struct NORefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() &
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()() const&  = delete;
  cuda::std::expected<int, TestError> operator()() &&      = delete;
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct NOCRefQual
{
  cuda::std::expected<int, TestError> operator()() & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() const&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()() &&      = delete;
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct NORVRefQual
{
  cuda::std::expected<int, TestError> operator()() &      = delete;
  cuda::std::expected<int, TestError> operator()() const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() &&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()() const&& = delete;
};

struct NORVCRefQual
{
  cuda::std::expected<int, TestError> operator()() &      = delete;
  cuda::std::expected<int, TestError> operator()() const& = delete;
  cuda::std::expected<int, TestError> operator()() &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()() const&&
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<int, TestError> expected_value{cuda::std::unexpect, 42};
  const cuda::std::expected<int, TestError> previous_error{cuda::std::unexpect, 1337};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.and_then(LVal{}) == 1);
      assert(i.and_then(NOLVal{}) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.and_then(LVal{}) == previous_error);
      assert(i.and_then(NOLVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      RefQual l{};
      assert(i.and_then(l) == 1);
      NORefQual nl{};
      assert(i.and_then(nl) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      RefQual l{};
      assert(i.and_then(l) == previous_error);
      NORefQual nl{};
      assert(i.and_then(nl) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test const& overload
  {
    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const CRefQual l{};
      assert(i.and_then(l) == 1);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const CRefQual l{};
      assert(i.and_then(l) == previous_error);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test && overload
  {
    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.and_then(RVRefQual{}) == 1);
      assert(i.and_then(NORVRefQual{}) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.and_then(RVRefQual{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).and_then(RVRefQual{}) == previous_error);
      assert(cuda::std::move(i).and_then(NORVRefQual{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(RVRefQual{})), cuda::std::expected<int, TestError>);
    }
  }

  // Test const&& overload
  {
    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const RVCRefQual l{};
      assert(i.and_then(cuda::std::move(l)) == 1);
      const NORVCRefQual nl{};
      assert(i.and_then(cuda::std::move(nl)) == expected_value);
      ASSERT_SAME_TYPE(decltype(i.and_then(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const RVCRefQual l{};
      assert(i.and_then(cuda::std::move(l)) == previous_error);
      const NORVCRefQual nl{};
      assert(i.and_then(cuda::std::move(nl)) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }
  }
}

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct NonConst
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> non_const()
  {
    return 1;
  }
};

struct nvrtc_workaround
{
  NonConst t{};

  template <class T = int>
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()()
  {
    return t.non_const();
  };
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<void, TestError> expect{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.and_then(l);
  cuda::std::move(expect).and_then(l);
}

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
  test_val_types();
  test_sfinae();

  cuda::std::expected<void, TestError> expect{cuda::std::unexpect, 42};
  const auto& cexpect = expect;

  expect.and_then(NeverCalled{});
  cuda::std::move(expect).and_then(NeverCalled{});
  cexpect.and_then(NeverCalled{});
  cuda::std::move(cexpect).and_then(NeverCalled{});

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
