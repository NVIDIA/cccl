//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct minimal_comparable_value
{
  int value;
};

TEST_FUNC constexpr bool operator<(minimal_comparable_value lhs, minimal_comparable_value rhs)
{
  return lhs.value < rhs.value;
}

TEST_FUNC constexpr bool operator==(minimal_comparable_value lhs, minimal_comparable_value rhs)
{
  return lhs.value == rhs.value;
}

namespace cuda::std
{
template <>
class numeric_limits<minimal_comparable_value>
{
public:
  static constexpr bool is_specialized = true;

  TEST_FUNC static constexpr minimal_comparable_value lowest() noexcept
  {
    return {0};
  }

  TEST_FUNC static constexpr minimal_comparable_value max() noexcept
  {
    return {100};
  }
};
} // namespace cuda::std

TEST_FUNC constexpr bool test()
{
  // --- static_bounds ---

  // Basic static bounds
  {
    constexpr auto b = cuda::args::static_bounds<1, 4096>{};
    static_assert(b.lower() == 1);
    static_assert(b.upper() == 4096);
  }

  // Exact static bounds
  {
    constexpr auto b = cuda::args::static_bounds<42, 42>{};
    static_assert(b.lower() == 42);
    static_assert(b.upper() == 42);
  }

  // Long type deduced from NTTPs
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::args::static_bounds<0L, 1000L>::lower()), long>);
  }

#if TEST_HAS_CLASS_NTTP
  // Static bounds preserve their original NTTP types
  {
    constexpr auto b = cuda::args::bounds<1.0f, 8.0f>();
    static_assert(b.lower() == 1.0f);
    static_assert(b.upper() == 8);
    static_assert(cuda::std::is_same_v<decltype(b.lower()), float>);
    static_assert(cuda::std::is_same_v<decltype(b.upper()), float>);
  }
#endif // TEST_HAS_CLASS_NTTP

  // --- runtime_bounds ---

  // Basic runtime bounds
  {
    auto b = cuda::args::runtime_bounds{10, 100};
    assert(b.lower() == 10);
    assert(b.upper() == 100);
    static_assert(cuda::std::is_same_v<decltype(b.lower()), int>);
  }

  // Default runtime bounds span the element type's numeric_limits range
  {
    constexpr cuda::args::runtime_bounds<int> b{};
    static_assert(b.lower() == cuda::std::numeric_limits<int>::lowest());
    static_assert(b.upper() == (cuda::std::numeric_limits<int>::max)());
  }

  // --- argument_bounds factory functions ---

  // Static via factory
  {
    constexpr auto b = cuda::args::bounds<1, 8>();
    static_assert(b.lower() == 1);
    static_assert(b.upper() == 8);
    static_assert(cuda::args::__is_static_bounds_cv_v<decltype(b)>);
    static_assert(!cuda::args::__is_runtime_bounds_cv_v<decltype(b)>);
    static_assert(cuda::args::__is_bounds_v<decltype(b)>);
  }

  // Runtime via factory
  {
    auto b = cuda::args::bounds(10, 100);
    assert(b.lower() == 10);
    assert(b.upper() == 100);
    static_assert(!cuda::args::__is_static_bounds_cv_v<decltype(b)>);
    static_assert(cuda::args::__is_runtime_bounds_cv_v<decltype(b)>);
    static_assert(cuda::args::__is_bounds_v<decltype(b)>);
  }

  // Runtime bounds only require operator< and operator==.
  {
    constexpr auto b = cuda::args::bounds(minimal_comparable_value{10}, minimal_comparable_value{20});
    static_assert(b.lower() == minimal_comparable_value{10});
    static_assert(b.upper() == minimal_comparable_value{20});
  }

  // Static and runtime bounds intersection
  {
    static_assert(cuda::args::__has_bounds_intersection<int, cuda::args::static_bounds<1, 100>>(
      cuda::args::runtime_bounds<int>{50, 200}));
    static_assert(!cuda::args::__has_bounds_intersection<int, cuda::args::static_bounds<100, 200>>(
      cuda::args::runtime_bounds<int>{0, 50}));
  }

  // Runtime bounds validation with no static bounds only requires operator< and operator==.
  {
    minimal_comparable_value values[] = {{10}, {20}};
    [[maybe_unused]] auto arg         = cuda::args::deferred_sequence{
      cuda::std::span<minimal_comparable_value>{values, 2},
      cuda::args::bounds(minimal_comparable_value{5}, minimal_comparable_value{50})};
  }

  // Unsigned no-bounds arguments must not instantiate a pointless `value < 0` comparison.
  {
    unsigned int value        = 0;
    [[maybe_unused]] auto arg = cuda::args::deferred{&value};
  }

#if TEST_HAS_CLASS_NTTP
  // Static/runtime bounds intersection only requires operator< and operator==.
  {
    using static_bounds_t = cuda::args::static_bounds<minimal_comparable_value{10}, minimal_comparable_value{50}>;

    constexpr auto runtime_bounds = cuda::args::bounds(minimal_comparable_value{20}, minimal_comparable_value{40});
    static_assert(cuda::args::__has_bounds_intersection<minimal_comparable_value, static_bounds_t>(runtime_bounds));
    static_assert(!cuda::args::__has_bounds_intersection<minimal_comparable_value, static_bounds_t>(
      cuda::args::bounds(minimal_comparable_value{60}, minimal_comparable_value{70})));

    cuda::args::__validate_static_element_bounds<minimal_comparable_value, static_bounds_t>(
      minimal_comparable_value{30});
    cuda::args::__validate_runtime_element_bounds(minimal_comparable_value{30}, runtime_bounds);

    minimal_comparable_value values[] = {{20}, {30}};
    [[maybe_unused]] auto arg         = cuda::args::__immediate_sequence{
      cuda::std::span<minimal_comparable_value>{values, 2}, static_bounds_t{}, runtime_bounds};
  }
#endif // TEST_HAS_CLASS_NTTP

  // Non-bounds type
  {
    static_assert(!cuda::args::__is_bounds_v<int>);
  }

  // Bounds types accepted by argument wrapper template parameters
  {
    static_assert(cuda::args::__valid_static_bounds_v<int, cuda::args::no_bounds>);
    static_assert(cuda::args::__valid_static_bounds_v<int, cuda::args::static_bounds<1, 8>>);
    static_assert(!cuda::args::__valid_static_bounds_v<int, cuda::args::runtime_bounds<int>>);
    static_assert(!cuda::args::__valid_static_bounds_v<int, int>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
