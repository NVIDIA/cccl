//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Include this first
#include <cuda/experimental/execution.cuh>

// Then include the test helpers
#include "testing.cuh" // IWYU pragma: keep

// NOLINTBEGIN(misc-unused-using-decls)
using cuda::experimental::execution::completion_signatures;
using cuda::experimental::execution::set_error;
using cuda::experimental::execution::set_error_t;
using cuda::experimental::execution::set_stopped;
using cuda::experimental::execution::set_stopped_t;
using cuda::experimental::execution::set_value;
using cuda::experimental::execution::set_value_t;
// NOLINTEND(misc-unused-using-decls)

namespace
{
C2H_TEST("", "[utilities][completion_signatures]")
{
  STATIC_REQUIRE(completion_signatures{} == completion_signatures{});
  STATIC_REQUIRE_FALSE(completion_signatures{} != completion_signatures{});
}

// Additional tests for completion_signatures
C2H_TEST("completion_signatures_basic", "[utilities][completion_signatures]")
{
  constexpr auto cs_empty   = completion_signatures<>{};
  constexpr auto cs_value   = completion_signatures<set_value_t(int)>{};
  constexpr auto cs_error   = completion_signatures<set_error_t(float)>{};
  constexpr auto cs_stopped = completion_signatures<set_stopped_t()>{};
  constexpr auto cs_all     = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};

  // Test size
  STATIC_REQUIRE(cs_empty.size() == 0);
  STATIC_REQUIRE(cs_value.size() == 1);
  STATIC_REQUIRE(cs_all.size() == 3);

  // Test contains
  STATIC_REQUIRE(cs_value.contains(static_cast<set_value_t (*)(int)>(nullptr)));
  STATIC_REQUIRE_FALSE(cs_value.contains(static_cast<set_error_t (*)(float)>(nullptr)));
  STATIC_REQUIRE(cs_all.contains(static_cast<set_stopped_t (*)()>(nullptr)));

  // Test count
  STATIC_REQUIRE(cs_all.count(set_value) == 1);
  STATIC_REQUIRE(cs_all.count(set_error) == 1);
  STATIC_REQUIRE(cs_all.count(set_stopped) == 1);

  // Test operator==
  STATIC_REQUIRE(cs_value == cs_value);
  STATIC_REQUIRE_FALSE(cs_value == cs_error);
  STATIC_REQUIRE(cs_empty == cs_empty);
  STATIC_REQUIRE(cs_all == cs_all);
  STATIC_REQUIRE(completion_signatures<set_value_t(int), set_error_t(float)>{}
                 == completion_signatures<set_error_t(float), set_value_t(int)>{});

  // Test operator!=
  STATIC_REQUIRE(cs_value != cs_error);
  STATIC_REQUIRE_FALSE(cs_all != cs_all);

  // Test operator+
  STATIC_REQUIRE((cs_value + cs_error) == completion_signatures<set_value_t(int), set_error_t(float)>{});
  STATIC_REQUIRE((cs_empty + cs_value) == cs_value);
  STATIC_REQUIRE((cs_value + cs_empty) == cs_value);

  // Test operator-
  STATIC_REQUIRE((cs_all - cs_value) == completion_signatures<set_error_t(float), set_stopped_t()>{});
  STATIC_REQUIRE((cs_all - cs_error) == completion_signatures<set_value_t(int), set_stopped_t()>{});
  STATIC_REQUIRE((cs_all - cs_stopped) == completion_signatures<set_value_t(int), set_error_t(float)>{});
  STATIC_REQUIRE((cs_all - cs_all) == completion_signatures<>{});
  STATIC_REQUIRE((cs_value - cs_error) == cs_value);
}

// Test select
C2H_TEST("completion_signatures_select", "[utilities][completion_signatures]")
{
  constexpr auto cs = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};

  // select(set_value) should return only set_value_t(int)
  constexpr auto v = cs.select(set_value);
  STATIC_REQUIRE(v.size() == 1);
  STATIC_REQUIRE(v.contains<set_value_t(int)>());

  // select(set_error) should return only set_error_t(float)
  constexpr auto e = cs.select(set_error);
  STATIC_REQUIRE(e.size() == 1);
  STATIC_REQUIRE(e.contains<set_error_t(float)>());

  // select(set_stopped) should return only set_stopped_t()
  constexpr auto s = cs.select(set_stopped);
  STATIC_REQUIRE(s.size() == 1);
  STATIC_REQUIRE(s.contains<set_stopped_t()>());
}

// Test filter
struct filter_value_only
{
  template <class Sig>
  constexpr bool operator()(Sig*) const noexcept
  {
    return cuda::experimental::execution::__signature_disposition<Sig>
        == cuda::experimental::execution::__disposition_t::__value;
  }
};

C2H_TEST("completion_signatures_filter", "[utilities][completion_signatures]")
{
  constexpr auto cs       = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
  constexpr auto filtered = cs.filter(filter_value_only{});
  STATIC_REQUIRE(filtered.size() == 1);
  STATIC_REQUIRE(filtered.contains<set_value_t(int)>());
}

// Test apply
struct count_signatures
{
  template <class... Sigs>
  constexpr int operator()(Sigs*...) const noexcept
  {
    return sizeof...(Sigs);
  }
};

C2H_TEST("completion_signatures_apply", "[utilities][completion_signatures]")
{
  constexpr auto cs   = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
  constexpr int count = cs.apply(count_signatures{});
  STATIC_REQUIRE(count == 3);
}
} // namespace
