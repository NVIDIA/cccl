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
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(const int&)  = delete;
  cuda::std::expected<int, TestError> operator()(int&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct CLVal
{
  cuda::std::expected<int, TestError> operator()(int&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const int&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(int&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct RVal
{
  cuda::std::expected<int, TestError> operator()(int&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int&&)
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct CRVal
{
  cuda::std::expected<int, TestError> operator()(int&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&) = delete;
  cuda::std::expected<int, TestError> operator()(int&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const int&&)
  {
    return 1;
  }
};

struct RefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) &
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(int) const&  = delete;
  cuda::std::expected<int, TestError> operator()(int) &&      = delete;
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct CRefQual
{
  cuda::std::expected<int, TestError> operator()(int) & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) const&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(int) &&      = delete;
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct RVRefQual
{
  cuda::std::expected<int, TestError> operator()(int) &      = delete;
  cuda::std::expected<int, TestError> operator()(int) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) &&
  {
    return 1;
  }
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct RVCRefQual
{
  cuda::std::expected<int, TestError> operator()(int) &      = delete;
  cuda::std::expected<int, TestError> operator()(int) const& = delete;
  cuda::std::expected<int, TestError> operator()(int) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) const&&
  {
    return 1;
  }
};

struct NOLVal
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(const int&)  = delete;
  cuda::std::expected<int, TestError> operator()(int&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct NOCLVal
{
  cuda::std::expected<int, TestError> operator()(int&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const int&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(int&&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct NORVal
{
  cuda::std::expected<int, TestError> operator()(int&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&) = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(const int&&) = delete;
};

struct NOCRVal
{
  cuda::std::expected<int, TestError> operator()(int&)       = delete;
  cuda::std::expected<int, TestError> operator()(const int&) = delete;
  cuda::std::expected<int, TestError> operator()(int&&)      = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(const int&&)
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

struct NORefQual
{
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) &
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(int) const&  = delete;
  cuda::std::expected<int, TestError> operator()(int) &&      = delete;
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct NOCRefQual
{
  cuda::std::expected<int, TestError> operator()(int) & = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) const&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(int) &&      = delete;
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct NORVRefQual
{
  cuda::std::expected<int, TestError> operator()(int) &      = delete;
  cuda::std::expected<int, TestError> operator()(int) const& = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) &&
  {
    return cuda::std::unexpected<TestError>{42};
  }
  cuda::std::expected<int, TestError> operator()(int) const&& = delete;
};

struct NORVCRefQual
{
  cuda::std::expected<int, TestError> operator()(int) &      = delete;
  cuda::std::expected<int, TestError> operator()(int) const& = delete;
  cuda::std::expected<int, TestError> operator()(int) &&     = delete;
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(int) const&&
  {
    return cuda::std::unexpected<TestError>{42};
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<int, TestError> expected_error{cuda::std::unexpect, 42};
  const cuda::std::expected<int, TestError> previous_error{cuda::std::unexpect, 1337};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(i.and_then(LVal{}) == 1);
      assert(i.and_then(NOLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(i.and_then(LVal{}) == previous_error);
      assert(i.and_then(NOLVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      RefQual l{};
      assert(i.and_then(l) == 1);
      NORefQual nl{};
      assert(i.and_then(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      RefQual l{};
      assert(i.and_then(l) == previous_error);
      NORefQual nl{};
      assert(i.and_then(nl) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      assert(i.and_then(CLVal{}) == 1);
      assert(i.and_then(NOCLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(CLVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(i.and_then(CLVal{}) == previous_error);
      assert(i.and_then(NOCLVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(CLVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      const CRefQual l{};
      assert(i.and_then(l) == 1);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      const CRefQual l{};
      assert(i.and_then(l) == previous_error);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == previous_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(cuda::std::move(i).and_then(RVal{}) == 1);
      assert(cuda::std::move(i).and_then(NORVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(RVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).and_then(RVal{}) == previous_error);
      assert(cuda::std::move(i).and_then(NORVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(RVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(i.and_then(RVRefQual{}) == 1);
      assert(i.and_then(NORVRefQual{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(RVRefQual{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).and_then(RVal{}) == previous_error);
      assert(cuda::std::move(i).and_then(NORVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(RVal{})), cuda::std::expected<int, TestError>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      assert(cuda::std::move(i).and_then(CRVal{}) == 1);
      assert(cuda::std::move(i).and_then(NOCRVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(CRVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).and_then(CRVal{}) == previous_error);
      assert(cuda::std::move(i).and_then(NOCRVal{}) == previous_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(CRVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      const RVCRefQual l{};
      assert(i.and_then(cuda::std::move(l)) == 1);
      const NORVCRefQual nl{};
      assert(i.and_then(cuda::std::move(nl)) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.and_then(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 1337};
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
  template <typename T>
  __host__ __device__ constexpr cuda::std::expected<int, TestError> operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<NonConst, TestError> expect{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.and_then(l);
  cuda::std::move(expect).and_then(l);
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

  cuda::std::expected<int, TestError> expect{cuda::std::unexpect, 42};
  const auto& cexpect = expect;

  expect.and_then(NeverCalled{});
  cuda::std::move(expect).and_then(NeverCalled{});
  cexpect.and_then(NeverCalled{});
  cuda::std::move(cexpect).and_then(NeverCalled{});

  cuda::std::expected<NoCopy, TestError> nc{cuda::std::unexpect, 42};
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
