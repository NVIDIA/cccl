//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_TRANSFORM_TRANSFORM_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_TRANSFORM_TRANSFORM_COMMON_CUH

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <vector>

#include <c2h/catch2_test_helper.h>

namespace transform_test_util
{
// Doubling is not the identity, so an implementation that only copied the input to the output
// would fail. `custom_value` is accumulateable, so this works for every type under test.
struct custom_double
{
  template <class T>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr T operator()(const T& value) const
  {
    return value + value;
  }
};

// A second operator, so that nothing along the way assumes the operator is the one above.
struct custom_triple
{
  template <class T>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr T operator()(const T& value) const
  {
    return value + value + value;
  }
};

using custom_value = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
using value_types  = c2h::type_list<cuda::std::int32_t, float, custom_value>;
using operators    = c2h::type_list<custom_double, custom_triple>;

template <typename T>
T make_value(int i)
{
  return static_cast<T>(i);
}

template <>
inline custom_value make_value<>(int i)
{
  custom_value ret{};

  ret.key = static_cast<std::size_t>(i);
  ret.val = static_cast<std::size_t>(i);
  return ret;
};

// `transform` is rank-local: rank `r` reads only the elements it owns, so its reference is its own
// input with the operator applied element by element.
template <class T, class Op>
[[nodiscard]] std::vector<T> expected_for_rank(const std::vector<T>& values, Op op)
{
  std::vector<T> ret;

  ret.reserve(values.size());
  for (const auto& value : values)
  {
    ret.push_back(op(value));
  }
  return ret;
}
} // namespace transform_test_util

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_TRANSFORM_TRANSFORM_COMMON_CUH
