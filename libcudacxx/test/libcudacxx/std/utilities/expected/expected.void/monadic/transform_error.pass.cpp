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

// template<class F> constexpr auto transform_error(F&&) &;
// template<class F> constexpr auto transform_error(F&&) &&;
// template<class F> constexpr auto transform_error(F&&) const&;
// template<class F> constexpr auto transform_error(F&&) const&&;

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>

#include "../../types.h"
#include "test_macros.h"

struct LVal {
  TEST_HOST_DEVICE
  constexpr int operator()(TestError&) { return 42; }
  int operator()(const TestError&) = delete;
  int operator()(TestError&&) = delete;
  int operator()(const TestError&&) = delete;
};

struct CLVal {
  int operator()(TestError&) = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(const TestError&) { return 42; }
  int operator()(TestError&&) = delete;
  int operator()(const TestError&&) = delete;
};

struct RVal {
  int operator()(TestError&) = delete;
  int operator()(const TestError&) = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(TestError&&) { return 42; }
  int operator()(const TestError&&) = delete;
};

struct CRVal {
  int operator()(TestError&) = delete;
  int operator()(const TestError&) = delete;
  int operator()(TestError&&) = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(const TestError&&) { return 42; }
};

struct RefQual {
  TEST_HOST_DEVICE
  constexpr int operator()(TestError) & { return 42; }
  int operator()(TestError) const& = delete;
  int operator()(TestError) && = delete;
  int operator()(TestError) const&& = delete;
};

struct CRefQual {
  int operator()(TestError) & = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(TestError) const& { return 42; }
  int operator()(TestError) && = delete;
  int operator()(TestError) const&& = delete;
};

struct RVRefQual {
  int operator()(TestError) & = delete;
  int operator()(TestError) const& = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(TestError) && { return 42; }
  int operator()(TestError) const&& = delete;
};

struct RVCRefQual {
  int operator()(TestError) & = delete;
  int operator()(TestError) const& = delete;
  int operator()(TestError) && = delete;
  TEST_HOST_DEVICE
  constexpr int operator()(TestError) const&& { return 42; }
};

TEST_HOST_DEVICE
constexpr void test_val_types() {
  const cuda::std::expected<void, TestError> previous_value{cuda::std::in_place};
  const cuda::std::expected<void, TestError> expected_error{cuda::std::unexpect, 42};

  // Test & overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.transform_error(LVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(LVal{})), cuda::std::expected<void, int>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.transform_error(LVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(LVal{})), cuda::std::expected<void, int>);
    }

    //With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      RefQual l{};
      assert(i.transform_error(l) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(l)), cuda::std::expected<void, int>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      RefQual l{};
      assert(i.transform_error(l) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(l)), cuda::std::expected<void, int>);
    }
  }

  // Test const& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.transform_error(CLVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(CLVal{})), cuda::std::expected<void, int>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.transform_error(CLVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(CLVal{})), cuda::std::expected<void, int>);
    }

    //With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const CRefQual l{};
      assert(i.transform_error(l) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(l)), cuda::std::expected<void, int>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const CRefQual l{};
      assert(i.transform_error(l) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(l)), cuda::std::expected<void, int>);
    }
  }

  // Test && overload
  {
    // Without & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(cuda::std::move(i).transform_error(RVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform_error(RVal{})), cuda::std::expected<void, int>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).transform_error(RVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform_error(RVal{})), cuda::std::expected<void, int>);
    }

    //With & qualifier on F's operator()
    {
      cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(i.transform_error(RVRefQual{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(RVRefQual{})), cuda::std::expected<void, int>);
    }

    {
      cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(i.transform_error(RVRefQual{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(RVRefQual{})), cuda::std::expected<void, int>);
    }
  }

  // Test const&& overload
  {
    // Without & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      assert(cuda::std::move(i).transform_error(CRVal{}) == previous_value);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform_error(CRVal{})), cuda::std::expected<void, int>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      assert(cuda::std::move(i).transform_error(CRVal{}) == expected_error);
      ASSERT_SAME_TYPE(decltype(cuda::std::move(i).transform_error(CRVal{})), cuda::std::expected<void, int>);
    }

    //With & qualifier on F's operator()
    {
      const cuda::std::expected<void, TestError> i{cuda::std::in_place};
      const RVCRefQual l{};
      assert(i.transform_error(cuda::std::move(l)) == previous_value);
      ASSERT_SAME_TYPE(decltype(i.transform_error(cuda::std::move(l))), cuda::std::expected<void, int>);
    }

    {
      const cuda::std::expected<void, TestError> i{cuda::std::unexpect, 1337};
      const RVCRefQual l{};
      assert(i.transform_error(cuda::std::move(l)) == expected_error);
      ASSERT_SAME_TYPE(decltype(i.transform_error(cuda::std::move(l))), cuda::std::expected<void, int>);
    }
  }
}

struct NonConst {
  TEST_HOST_DEVICE
  constexpr int non_const() { return 1; }
};

// For a generic lambda, nvrtc appears to not know what to do and claims it needs an annotation (when normal lambdas don't).
// This is an expanded lambda from the original test.
struct nvrtc_workaround {
    template<typename T>
    TEST_HOST_DEVICE
    constexpr int operator()(T && t) {
        return t.non_const();
    }
};

// check that the lambda body is not instantiated during overload resolution
TEST_HOST_DEVICE
constexpr void test_sfinae() {
  cuda::std::expected<int, NonConst> expect{};
  auto l = nvrtc_workaround(); // [](auto&& x) { return x.non_const(); };
  expect.transform_error(l);
  cuda::std::move(expect).transform_error(l);
}

struct NoCopy {
  NoCopy() = default;
  TEST_HOST_DEVICE
  constexpr NoCopy(const NoCopy&);
  TEST_HOST_DEVICE
  constexpr int operator()(const NoCopy&&) { return 42; }
};

// We need an indirection so the assert does not break the compilation
template<class T>
struct AlwaysFalse {
  TEST_HOST_DEVICE
  constexpr AlwaysFalse() { assert(false); }
};

struct NeverCalled {
  template<class T>
  TEST_HOST_DEVICE
  constexpr int operator()(T) const { return AlwaysFalse<T>{}, 42; }
};

TEST_HOST_DEVICE
constexpr bool test() {
  test_sfinae();
  test_val_types();

  cuda::std::expected<void, TestError> expect{cuda::std::in_place};
  const auto& cexpect = expect;

  expect.transform_error(NeverCalled{});
  cuda::std::move(expect).transform_error(NeverCalled{});
  cexpect.transform_error(NeverCalled{});
  cuda::std::move(cexpect).transform_error(NeverCalled{});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");
  return 0;
}
