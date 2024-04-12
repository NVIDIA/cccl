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

// template<class F> constexpr auto or_else(F&&) &;
// template<class F> constexpr auto or_else(F&&) &&;
// template<class F> constexpr auto or_else(F&&) const&;
// template<class F> constexpr auto or_else(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/expected>

#include "../../types.h"
#include "test_macros.h"

struct LVal
{
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(const TestError&)  = delete;
  cuda::std::expected<void, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct CLVal
{
  cuda::std::expected<void, TestError> operator()(TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(const TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct RVal
{
  cuda::std::expected<void, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct CRVal
{
  cuda::std::expected<void, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&) = delete;
  cuda::std::expected<void, TestError> operator()(TestError&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(const TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct RefQual
{
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) &
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) const&  = delete;
  cuda::std::expected<void, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct CRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) & = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) const&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct RVRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) &&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct RVCRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const& = delete;
  cuda::std::expected<void, TestError> operator()(TestError) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) const&&
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NOLVal
{
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(const TestError&)  = delete;
  cuda::std::expected<void, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct NOCLVal
{
  cuda::std::expected<void, TestError> operator()(TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(const TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct NORVal
{
  cuda::std::expected<void, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(const TestError&&) = delete;
};

struct NOCRVal
{
  cuda::std::expected<void, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<void, TestError> operator()(const TestError&) = delete;
  cuda::std::expected<void, TestError> operator()(TestError&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(const TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NORefQual
{
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) &
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) const&  = delete;
  cuda::std::expected<void, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct NOCRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) & = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) const&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct NORVRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) &&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<void, TestError> operator()(TestError) const&& = delete;
};

struct NORVCRefQual
{
  cuda::std::expected<void, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<void, TestError> operator()(TestError) const& = delete;
  cuda::std::expected<void, TestError> operator()(TestError) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(TestError) const&&
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NonConst
{
  __host__ __device__ constexpr cuda::std::expected<void, TestError> non_const()
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<void, TestError> previous_value{cuda::std::in_place};
  const cuda::std::expected<void, TestError> expected_error{cuda::std::unexpect, 42};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.or_else(LVal{}) == previous_value);
      assert(i.or_else(NOLVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(LVal{})), cuda::std::expected<void, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.or_else(LVal{}) == expected_error);
      assert(i.or_else(NOLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(LVal{})), cuda::std::expected<void, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      RefQual l{};
      assert(i.or_else(l) == previous_value);
      NORefQual nl{};
      assert(i.or_else(nl) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<void, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      RefQual l{};
      assert(i.or_else(l) == expected_error);
      NORefQual nl{};
      assert(i.or_else(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<void, TestError>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.or_else(CLVal{}) == previous_value);
      assert(i.or_else(NOCLVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(CLVal{})), cuda::std::expected<void, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.or_else(CLVal{}) == expected_error);
      assert(i.or_else(NOCLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(CLVal{})), cuda::std::expected<void, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const CRefQual l{};
      assert(i.or_else(l) == previous_value);
      const NOCRefQual nl{};
      assert(i.or_else(nl) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<void, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const CRefQual l{};
      assert(i.or_else(l) == expected_error);
      const NOCRefQual nl{};
      assert(i.or_else(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<void, TestError>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(cuda::std::move(i).or_else(RVal{}) == previous_value);
      assert(cuda::std::move(i).or_else(NORVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<void, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(RVal{}) == expected_error);
      assert(cuda::std::move(i).or_else(NORVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<void, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.or_else(RVRefQual{}) == previous_value);
      assert(i.or_else(NORVRefQual{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(RVRefQual{})), cuda::std::expected<void, TestError>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(RVal{}) == expected_error);
      assert(cuda::std::move(i).or_else(NORVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<void, TestError>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(cuda::std::move(i).or_else(CRVal{}) == previous_value);
      assert(cuda::std::move(i).or_else(NOCRVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(CRVal{})), cuda::std::expected<void, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(CRVal{}) == expected_error);
      assert(cuda::std::move(i).or_else(NOCRVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(CRVal{})), cuda::std::expected<void, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const RVCRefQual l{};
      assert(i.or_else(cuda::std::move(l)) == previous_value);
      const NORVCRefQual nl{};
      assert(i.or_else(cuda::std::move(nl)) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(cuda::std::move(l))), cuda::std::expected<void, TestError>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const RVCRefQual l{};
      assert(i.or_else(cuda::std::move(l)) == expected_error);
      const NORVCRefQual nl{};
      assert(i.or_else(cuda::std::move(nl)) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(cuda::std::move(l))), cuda::std::expected<void, TestError>);
    }
  }
}

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct nvrtc_workaround
{
  template <typename T>
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<void, NonConst> expect{cuda::std::in_place};
  ;
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.or_else(l);
  cuda::std::move(expect).or_else(l);
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
  template <class T>
  __host__ __device__ constexpr cuda::std::expected<void, TestError> operator()(T) const
  {
    return AlwaysFalse<T>{}, cuda::std::expected<void, TestError>{cuda::std::in_place};
  }
};

__host__ __device__ constexpr bool test()
{
  test_val_types();
  test_sfinae();

  cuda::std::expected<void, TestError> expect{cuda::std::in_place};
  const auto& cexpect = expect;

  expect.or_else(NeverCalled{});
  cuda::std::move(expect).or_else(NeverCalled{});
  cexpect.or_else(NeverCalled{});
  cuda::std::move(cexpect).or_else(NeverCalled{});

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
