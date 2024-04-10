//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvcc-11.1

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
  __host__ __device__ constexpr int operator()(int&)
  {
    return 1;
  }
  int operator()(const int&)  = delete;
  int operator()(int&&)       = delete;
  int operator()(const int&&) = delete;
};

struct CLVal
{
  int operator()(int&) = delete;
  __host__ __device__ constexpr int operator()(const int&)
  {
    return 1;
  }
  int operator()(int&&)       = delete;
  int operator()(const int&&) = delete;
};

struct RVal
{
  int operator()(int&)       = delete;
  int operator()(const int&) = delete;
  __host__ __device__ constexpr int operator()(int&&)
  {
    return 1;
  }
  int operator()(const int&&) = delete;
};

struct CRVal
{
  int operator()(int&)       = delete;
  int operator()(const int&) = delete;
  int operator()(int&&)      = delete;
  __host__ __device__ constexpr int operator()(const int&&)
  {
    return 1;
  }
};

struct RefQual
{
  __host__ __device__ constexpr int operator()(int) &
  {
    return 1;
  }
  int operator()(int) const&  = delete;
  int operator()(int) &&      = delete;
  int operator()(int) const&& = delete;
};

struct CRefQual
{
  int operator()(int) & = delete;
  __host__ __device__ constexpr int operator()(int) const&
  {
    return 1;
  }
  int operator()(int) &&      = delete;
  int operator()(int) const&& = delete;
};

struct RVRefQual
{
  int operator()(int) &      = delete;
  int operator()(int) const& = delete;
  __host__ __device__ constexpr int operator()(int) &&
  {
    return 1;
  }
  int operator()(int) const&& = delete;
};

struct RVCRefQual
{
  int operator()(int) &      = delete;
  int operator()(int) const& = delete;
  int operator()(int) &&     = delete;
  __host__ __device__ constexpr int operator()(int) const&&
  {
    return 1;
  }
};

__host__ __device__ constexpr void test_val_types()
{
  const cuda::std::expected<int, TestError> expected_error{cuda::std::unexpect, 42};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(i.transform(LVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      assert(i.transform(LVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      RefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      RefQual l{};
      assert(i.transform(l) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      assert(i.transform(CLVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(CLVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      assert(i.transform(CLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(CLVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      const CRefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      const CRefQual l{};
      assert(i.transform(l) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::expected<int, TestError>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(cuda::std::move(i).transform(RVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(RVal{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      assert(cuda::std::move(i).transform(RVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(RVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::expected<int, TestError> i{0};
      assert(i.transform(RVRefQual{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), cuda::std::expected<int, TestError>);
    }

    {
      cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      assert(i.transform(RVRefQual{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), cuda::std::expected<int, TestError>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      assert(cuda::std::move(i).transform(CRVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(CRVal{})), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      assert(cuda::std::move(i).transform(CRVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(CRVal{})), cuda::std::expected<int, TestError>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::expected<int, TestError> i{0};
      const RVCRefQual l{};
      assert(i.transform(cuda::std::move(l)) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }

    {
      const cuda::std::expected<int, TestError> i{cuda::std::unexpect, 42};
      const RVCRefQual l{};
      assert(i.transform(cuda::std::move(l)) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform(cuda::std::move(l))), cuda::std::expected<int, TestError>);
    }
  }
}

#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 8 // GCC7 and GCC8 seem to be too eager to instantiate the world
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
  template <typename T>
  __host__ __device__ constexpr int operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ constexpr void test_sfinae()
{
  cuda::std::expected<NonConst, TestError> expect{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.transform(l);
  cuda::std::move(expect).transform(l);
}
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 8

struct NoCopy
{
  NoCopy()                                            = default;
  __host__ __device__ constexpr NoCopy(const NoCopy&) = delete;
  __host__ __device__ constexpr int operator()(const NoCopy&&)
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
#if !defined(TEST_COMPILER_GCC) || __GNUC__ > 8 // GCC7 and GCC8 seem to be too eager to instantiate the world
  test_sfinae();
#endif // !defined(TEST_COMPILER_GCC) || __GNUC__ > 8

  cuda::std::expected<int, TestError> expect{cuda::std::unexpect, 42};
  const auto& cexpect = expect;

  expect.transform(NeverCalled{});
  cuda::std::move(expect).transform(NeverCalled{});
  cexpect.transform(NeverCalled{});
  cuda::std::move(cexpect).transform(NeverCalled{});

  cuda::std::expected<NoCopy, TestError> nc{cuda::std::unexpect, 42};
  const auto& cnc = nc;
  cuda::std::move(nc).transform(NoCopy{});
  cuda::std::move(cnc).transform(NoCopy{});
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
