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

// template<class F> constexpr auto transform(F&&) &;
// template<class F> constexpr auto transform(F&&) &&;
// template<class F> constexpr auto transform(F&&) const&;
// template<class F> constexpr auto transform(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

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

#if TEST_STD_VER >= 2017
struct NoCopy
{
  NoCopy() = default;
  __host__ __device__ NoCopy(const NoCopy&)
  {
    assert(false);
  }
  __host__ __device__ int operator()(const NoCopy&&)
  {
    return 1;
  }
};

struct NoMove
{
  NoMove()         = default;
  NoMove(NoMove&&) = delete;
  __host__ __device__ NoMove operator()(const NoCopy&&)
  {
    return NoMove{};
  }
};
#endif

__host__ __device__ constexpr void test_val_types()
{
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(i.transform(LVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(LVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      RefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::optional<int>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      assert(i.transform(CLVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(CLVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      const CRefQual l{};
      assert(i.transform(l) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(l)), cuda::std::optional<int>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(cuda::std::move(i).transform(RVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(RVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(i.transform(RVRefQual{}) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(RVRefQual{})), cuda::std::optional<int>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      assert(cuda::std::move(i).transform(CRVal{}) == 1);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform(CRVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      const RVCRefQual l{};
      assert(i.transform(cuda::std::move(l)) == 1);
      ASSERT_SAME_TYPE(decltype(i.transform(cuda::std::move(l))), cuda::std::optional<int>);
    }
  }
}

struct NonConst
{
  __host__ __device__ int non_const()
  {
    return 1;
  }
};

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct nvrtc_workaround
{
  template <typename T>
  __host__ __device__ int operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ TEST_CONSTEXPR_CXX17 void test_sfinae()
{
  cuda::std::optional<NonConst> opt{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  opt.transform(l);
  cuda::std::move(opt).transform(l);
}

__host__ __device__ TEST_CONSTEXPR_CXX17 bool test()
{
  test_sfinae();
  test_val_types();
  cuda::std::optional<int> opt;
  const auto& copt = opt;

  const auto never_called = [](int) {
    assert(false);
    return 0;
  };

  opt.transform(never_called);
  cuda::std::move(opt).transform(never_called);
  copt.transform(never_called);
  cuda::std::move(copt).transform(never_called);

  // the code below depends on guaranteed copy/move elision
#if TEST_STD_VER >= 2017 && (!defined(TEST_COMPILER_MSVC) || TEST_STD_VER >= 2020)
  cuda::std::optional<NoCopy> nc;
  const auto& cnc = nc;
  cuda::std::move(nc).transform(NoCopy{});
  cuda::std::move(cnc).transform(NoCopy{});

  cuda::std::move(nc).transform(NoMove{});
  cuda::std::move(cnc).transform(NoMove{});
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test());
#endif
  return 0;
}
