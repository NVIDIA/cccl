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
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(const TestError&)  = delete;
  cuda::std::expected<int, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct CLVal
{
  cuda::std::expected<int, TestError> operator()(TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const TestError&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct RVal
{
  cuda::std::expected<int, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError&&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct CRVal
{
  cuda::std::expected<int, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&) = delete;
  cuda::std::expected<int, TestError> operator()(TestError&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const TestError&&)
  {
    return 1;
  }
};

struct RefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) &
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(TestError) const&  = delete;
  cuda::std::expected<int, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct CRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) const&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct RVRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) &&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct RVCRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const& = delete;
  cuda::std::expected<int, TestError> operator()(TestError) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) const&&
  {
    return 1;
  }
};

struct NOLVal
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(const TestError&)  = delete;
  cuda::std::expected<int, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct NOCLVal
{
  cuda::std::expected<int, TestError> operator()(TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const TestError&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(TestError&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct NORVal
{
  cuda::std::expected<int, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(const TestError&&) = delete;
};

struct NOCRVal
{
  cuda::std::expected<int, TestError> operator()(TestError&)       = delete;
  cuda::std::expected<int, TestError> operator()(const TestError&) = delete;
  cuda::std::expected<int, TestError> operator()(TestError&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const TestError&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NORefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) &
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(TestError) const&  = delete;
  cuda::std::expected<int, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct NOCRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) const&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(TestError) &&      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct NORVRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) &&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(TestError) const&& = delete;
};

struct NORVCRefQual
{
  cuda::std::expected<int, TestError> operator()(TestError) &      = delete;
  cuda::std::expected<int, TestError> operator()(TestError) const& = delete;
  cuda::std::expected<int, TestError> operator()(TestError) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(TestError) const&&
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NonConst
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> non_const()
  {
    return 1;
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<int, TestError> previous_value{cuda::std::in_place, 42};
  const cuda::std::expected<int, TestError> expected_error{cuda::std::unexpect, 42};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{42};
      assert(i.or_else(LVal{}) == previous_value);
      assert(i.or_else(NOLVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(LVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(i.or_else(LVal{}) == 1);
      assert(i.or_else(NOLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(LVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{42};
      RefQual l{};
      assert(i.or_else(l) == previous_value);
      NORefQual nl{};
      assert(i.or_else(nl) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      RefQual l{};
      assert(i.or_else(l) == 1);
      NORefQual nl{};
      assert(i.or_else(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{42};
      assert(i.or_else(CLVal{}) == previous_value);
      assert(i.or_else(NOCLVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(CLVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(i.or_else(CLVal{}) == 1);
      assert(i.or_else(NOCLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(CLVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{42};
      const CRefQual l{};
      assert(i.or_else(l) == previous_value);
      const NOCRefQual nl{};
      assert(i.or_else(nl) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      const CRefQual l{};
      assert(i.or_else(l) == 1);
      const NOCRefQual nl{};
      assert(i.or_else(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{42};
      assert(cuda::std::move(i).or_else(RVal{}) == previous_value);
      assert(cuda::std::move(i).or_else(NORVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(RVal{}) == 1);
      assert(cuda::std::move(i).or_else(NORVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{42};
      assert(i.or_else(RVRefQual{}) == previous_value);
      assert(i.or_else(NORVRefQual{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(RVRefQual{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(RVal{}) == 1);
      assert(cuda::std::move(i).or_else(NORVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(RVal{})), cuda::std::expected<int, TestError>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{42};
      assert(cuda::std::move(i).or_else(CRVal{}) == previous_value);
      assert(cuda::std::move(i).or_else(NOCRVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(CRVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).or_else(CRVal{}) == 1);
      assert(cuda::std::move(i).or_else(NOCRVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).or_else(CRVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{42};
      const RVCRefQual l{};
      assert(i.or_else(cuda::std::move(l)) == previous_value);
      const NORVCRefQual nl{};
      assert(i.or_else(cuda::std::move(nl)) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.or_else(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      const RVCRefQual l{};
      assert(i.or_else(cuda::std::move(l)) == 1);
      const NORVCRefQual nl{};
      assert(i.or_else(cuda::std::move(nl)) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.or_else(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }
  }
}

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct nvrtc_workaround
{
  template <typename T>
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<int, NonConst> expect{cuda::std::in_place, 42};
  ;
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.or_else(l);
  cuda::std::move(expect).or_else(l);
}

struct NoCopy
{
  NoCopy()                                            = default;
  __host__ __device__ constexpr NoCopy(const NoCopy&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const NoCopy&&)
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
  template <class T>
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(T) const
  {
    return AlwaysFalse<T>{}, cuda::std::expected<int, TestError>{42};
  }
};

__host__ __device__ constexpr bool test()
{
  test_val_types();
  test_sfinae();

  cuda::std::expected<int, TestError> expect{cuda::std::in_place, 42};
  const auto& cexpect = expect;

  expect.or_else(NeverCalled{});
  cuda::std::move(expect).or_else(NeverCalled{});
  cexpect.or_else(NeverCalled{});
  cuda::std::move(cexpect).or_else(NeverCalled{});

  cuda::std::expected<NoCopy, TestError> nc{cuda::std::in_place};
  const auto& cnc = nc;
  cuda::std::move(cnc).and_then(NoCopy{});
  cuda::std::move(nc).and_then(NoCopy{});

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
