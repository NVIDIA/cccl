//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// SIMD host/device functions are not supported in tile mode.

// [simd.permute.static]: mapper constraints, named-lvalue invocation, and overload selection.

#include <cuda/simd>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../simd_test_utils.h"
#include "test_macros.h"

using Index = cuda::std::ptrdiff_t;

struct lvalue_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index i) &
  {
    return i;
  }
};

struct qualified_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index) &
  {
    return 1;
  }
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index) const&
  {
    return 0;
  }
};

struct mixed_result_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index) const
  {
    return 0;
  }
  TEST_HOST_DEVICE_FUNC constexpr double operator()(Index i, Index size) const
  {
    return static_cast<double>(size - 1 - i);
  }
};

struct scalar_argument_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index i) const
  {
    return i;
  }
  template <typename I>
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(I) const
  {
    return simd::zero_element;
  }
};

struct unsigned_result_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr unsigned operator()(Index i) const
  {
    return static_cast<unsigned>(i);
  }
};

struct non_integral_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr double operator()(Index) const
  {
    return 0.0;
  }
};

struct reference_result_mapper
{
  int value = 0;
  TEST_HOST_DEVICE_FUNC constexpr int& operator()(Index) &
  {
    return value;
  }
};

struct rvalue_mapper
{
  TEST_HOST_DEVICE_FUNC constexpr Index operator()(Index) &&
  {
    return 0;
  }
};

template <typename V, typename Map, typename = void>
inline constexpr bool has_static_permute = false;

template <typename V, typename Map>
inline constexpr bool has_static_permute<
  V,
  Map,
  cuda::std::void_t<decltype(simd::permute(cuda::std::declval<const V&>(), cuda::std::declval<Map>()))>> = true;

template <typename Map>
TEST_HOST_DEVICE_FUNC constexpr void test_mapper(Map&& map, int stride, int offset)
{
  using V = simd::vec<int, 4>;
  using M = simd::mask<int, 4>;

  const V values(iota_generator<int>{});
  const M mask(is_even{});
  // Check cv-qualification and value categories for named-lvalue mapper invocation.
  // NOLINTBEGIN(bugprone-use-after-move)
  const auto selected_values = simd::permute(values, cuda::std::forward<Map>(map));
  const auto selected_mask   = simd::permute(mask, cuda::std::forward<Map>(map));
  // NOLINTEND(bugprone-use-after-move)
  for (int i = 0; i < 4; ++i)
  {
    const int index = (i * stride + offset) % 4;
    assert(selected_values[i] == values[index]);
    assert(selected_mask[i] == mask[index]);
  }
}

template <typename V>
TEST_HOST_DEVICE_FUNC constexpr void test_constraints()
{
  static_assert(has_static_permute<V, lvalue_mapper>);
  static_assert(has_static_permute<V, lvalue_mapper&>);
  static_assert(!has_static_permute<V, rvalue_mapper>);
  static_assert(!has_static_permute<V, reference_result_mapper>);
  static_assert(!has_static_permute<V, non_integral_mapper>);
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_constraints<simd::vec<int, 4>>();
  test_constraints<simd::mask<int, 4>>();

  test_mapper(lvalue_mapper{}, 1, 0); // linear mapping
  test_mapper(scalar_argument_mapper{}, 1, 0); // linear mapping
  test_mapper(unsigned_result_mapper{}, 1, 0); // linear mapping
  test_mapper(mixed_result_mapper{}, 3, 3); // must match double overload

  // GCC 7 rejects default-initialized empty objects in constexpr functions; value-initialize the mappers.
  qualified_mapper mutable_map{}; // NOLINT(misc-const-correctness)
  const qualified_mapper const_map{};
  test_mapper(mutable_map, 0, 1); // return 1 (operator()&) for mutable lvalue
  test_mapper(const_map, 0, 0); // return 0 (operator() const&) for const lvalue
  test_mapper(qualified_mapper{}, 0, 1); // return 1 (operator()&) for mutable rvalue
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
