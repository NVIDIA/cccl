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

// template<class F> constexpr auto and_then(F&&) &;
// template<class F> constexpr auto and_then(F&&) &&;
// template<class F> constexpr auto and_then(F&&) const&;
// template<class F> constexpr auto and_then(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "test_macros.h"

struct LVal
{
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int&)
  {
    return 1;
  }
  cuda::std::optional<int> operator()(const int&)  = delete;
  cuda::std::optional<int> operator()(int&&)       = delete;
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct CLVal
{
  cuda::std::optional<int> operator()(int&) = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(const int&)
  {
    return 1;
  }
  cuda::std::optional<int> operator()(int&&)       = delete;
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct RVal
{
  cuda::std::optional<int> operator()(int&)       = delete;
  cuda::std::optional<int> operator()(const int&) = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int&&)
  {
    return 1;
  }
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct CRVal
{
  cuda::std::optional<int> operator()(int&)       = delete;
  cuda::std::optional<int> operator()(const int&) = delete;
  cuda::std::optional<int> operator()(int&&)      = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(const int&&)
  {
    return 1;
  }
};

struct RefQual
{
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) &
  {
    return 1;
  }
  cuda::std::optional<int> operator()(int) const&  = delete;
  cuda::std::optional<int> operator()(int) &&      = delete;
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct CRefQual
{
  cuda::std::optional<int> operator()(int) & = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) const&
  {
    return 1;
  }
  cuda::std::optional<int> operator()(int) &&      = delete;
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct RVRefQual
{
  cuda::std::optional<int> operator()(int) &      = delete;
  cuda::std::optional<int> operator()(int) const& = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) &&
  {
    return 1;
  }
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct RVCRefQual
{
  cuda::std::optional<int> operator()(int) &      = delete;
  cuda::std::optional<int> operator()(int) const& = delete;
  cuda::std::optional<int> operator()(int) &&     = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) const&&
  {
    return 1;
  }
};

struct NOLVal
{
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int&)
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(const int&)  = delete;
  cuda::std::optional<int> operator()(int&&)       = delete;
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct NOCLVal
{
  cuda::std::optional<int> operator()(int&) = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(const int&)
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(int&&)       = delete;
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct NORVal
{
  cuda::std::optional<int> operator()(int&)       = delete;
  cuda::std::optional<int> operator()(const int&) = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int&&)
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(const int&&) = delete;
};

struct NOCRVal
{
  cuda::std::optional<int> operator()(int&)       = delete;
  cuda::std::optional<int> operator()(const int&) = delete;
  cuda::std::optional<int> operator()(int&&)      = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(const int&&)
  {
    return cuda::std::nullopt;
  }
};

struct NORefQual
{
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) &
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(int) const&  = delete;
  cuda::std::optional<int> operator()(int) &&      = delete;
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct NOCRefQual
{
  cuda::std::optional<int> operator()(int) & = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) const&
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(int) &&      = delete;
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct NORVRefQual
{
  cuda::std::optional<int> operator()(int) &      = delete;
  cuda::std::optional<int> operator()(int) const& = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) &&
  {
    return cuda::std::nullopt;
  }
  cuda::std::optional<int> operator()(int) const&& = delete;
};

struct NORVCRefQual
{
  cuda::std::optional<int> operator()(int) &      = delete;
  cuda::std::optional<int> operator()(int) const& = delete;
  cuda::std::optional<int> operator()(int) &&     = delete;
  __host__ __device__ constexpr cuda::std::optional<int> operator()(int) const&&
  {
    return cuda::std::nullopt;
  }
};

struct NoCopy
{
  NoCopy() = default;
  __host__ __device__ NoCopy(const NoCopy&)
  {
    assert(false);
  }
  __host__ __device__ cuda::std::optional<int> operator()(const NoCopy&&)
  {
    return 1;
  }
};

struct NonConst
{
  __host__ __device__ cuda::std::optional<int> non_const()
  {
    return 1;
  }
};

__host__ __device__ constexpr void test_val_types()
{
  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(i.and_then(LVal{}) == 1);
      assert(i.and_then(NOLVal{}) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(LVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      RefQual l{};
      assert(i.and_then(l) == 1);
      NORefQual nl{};
      assert(i.and_then(nl) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::optional<int>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      assert(i.and_then(CLVal{}) == 1);
      assert(i.and_then(NOCLVal{}) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(CLVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      const CRefQual l{};
      assert(i.and_then(l) == 1);
      const NOCRefQual nl{};
      assert(i.and_then(nl) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(l)), cuda::std::optional<int>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(cuda::std::move(i).and_then(RVal{}) == 1);
      assert(cuda::std::move(i).and_then(NORVal{}) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(RVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      cuda::std::optional<int> i{0};
      assert(i.and_then(RVRefQual{}) == 1);
      assert(i.and_then(NORVRefQual{}) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(RVRefQual{})), cuda::std::optional<int>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      assert(cuda::std::move(i).and_then(CRVal{}) == 1);
      assert(cuda::std::move(i).and_then(NOCRVal{}) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).and_then(CRVal{})), cuda::std::optional<int>);
    }

    // With & qualifier on F's operator()
    {
      const cuda::std::optional<int> i{0};
      const RVCRefQual l{};
      assert(i.and_then(cuda::std::move(l)) == 1);
      const NORVCRefQual nl{};
      assert(i.and_then(cuda::std::move(nl)) == cuda::std::nullopt);
      ASSERT_SAME_TYPE(decltype(i.and_then(cuda::std::move(l))), cuda::std::optional<int>);
    }
  }
}

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas
// don't). This is an expanded lambda from the original test.
struct nvrtc_workaround
{
  template <typename T>
  __host__ __device__ cuda::std::optional<int> operator()(T&& t)
  {
    return t.non_const();
  }
};

// check that the lambda body is not instantiated during overload resolution
__host__ __device__ TEST_CONSTEXPR_CXX17 void test_sfinae()
{
  cuda::std::optional<NonConst> opt{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  opt.and_then(l);
  cuda::std::move(opt).and_then(l);
}

__host__ __device__ TEST_CONSTEXPR_CXX17 bool test()
{
  test_val_types();
  cuda::std::optional<int> opt{};
  const auto& copt = opt;

  const auto never_called = [](int) {
    assert(false);
    return cuda::std::optional<int>{};
  };

  opt.and_then(never_called);
  cuda::std::move(opt).and_then(never_called);
  copt.and_then(never_called);
  cuda::std::move(copt).and_then(never_called);

  cuda::std::optional<NoCopy> nc;
  const auto& cnc = nc;
  cuda::std::move(cnc).and_then(NoCopy{});
  cuda::std::move(nc).and_then(NoCopy{});

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
