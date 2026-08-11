//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_COMMON_CUH

#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <nccl_test_common.h>

#include <c2h/catch2_test_helper.h>

struct custom_plus
{
  template <class T>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr T operator()(const T& lhs, const T& rhs) const
  {
    return lhs + rhs;
  }
};

using custom_value = c2h::custom_type_t<c2h::accumulateable_t, c2h::less_comparable_t, c2h::equal_comparable_t>;
using value_types  = c2h::type_list<cuda::std::int32_t, float, custom_value>;
using operators    = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, custom_plus>;

static_assert(cudax::nccl_transportable<custom_value>);

template <typename T>
[[nodiscard]] inline T make_value(int i)
{
  return static_cast<T>(i);
}

template <>
[[nodiscard]] inline custom_value make_value<>(int i)
{
  custom_value ret{};

  ret.key = static_cast<std::size_t>(i);
  ret.val = static_cast<std::size_t>(i);
  return ret;
}

// Must cover every operator in operators
template <class T, class Op>
[[nodiscard]] T get_identity()
{
  if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>> || cuda::std::is_same_v<Op, custom_plus>)
  {
    return make_value<T>(0);
  }
  else if constexpr (cuda::std::is_same_v<Op, cuda::maximum<>>)
  {
    return cuda::std::numeric_limits<T>::lowest();
  }
  else
  {
    static_assert(cuda::std::__always_false_v<T, Op>, "Add handling");
  }
}

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_REDUCE_REDUCE_COMMON_CUH
