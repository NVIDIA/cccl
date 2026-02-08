// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <ostream>

/***********************************************************************************************************************
 * Sensible Distribution Intervals for Test Data
 **********************************************************************************************************************/

namespace detail
{
template <typename T, typename Operator, cuda::std::ptrdiff_t MaxReductionLength, typename = void>
struct dist_interval
{
  static constexpr T min()
  {
    return cuda::std::numeric_limits<T>::lowest();
  }
  static constexpr T max()
  {
    return cuda::std::numeric_limits<T>::max();
  }
};

template <typename T, cuda::std::ptrdiff_t MaxReductionLength>
struct dist_interval<
  T,
  cuda::std::plus<>,
  MaxReductionLength,
  cuda::std::enable_if_t<cuda::std::__cccl_is_signed_integer_v<T> || cuda::std::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf destroying pseudo-associativity
  static constexpr T min()
  {
    return static_cast<T>(cuda::std::numeric_limits<T>::lowest() / MaxReductionLength);
  }
  static constexpr T max()
  {
    return static_cast<T>(cuda::std::numeric_limits<T>::max() / MaxReductionLength);
  }
};

template <typename T, cuda::std::ptrdiff_t MaxReductionLength>
struct dist_interval<
  T,
  cuda::std::multiplies<>,
  MaxReductionLength,
  cuda::std::enable_if_t<cuda::std::__cccl_is_signed_integer_v<T> || cuda::std::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf destroying pseudo-associativity
  // Use floating point arithmetic to avoid unnecessarily small interval.
  static constexpr T min()
  {
    const double log2_abs_min = cuda::std::log2(cuda::std::fabs(cuda::std::numeric_limits<T>::lowest()));
    return static_cast<T>(-cuda::std::exp2(log2_abs_min / MaxReductionLength));
  }
  static constexpr T max()
  {
    const double log2_max = cuda::std::log2(cuda::std::numeric_limits<T>::max());
    return static_cast<T>(cuda::std::exp2(log2_max / MaxReductionLength));
  }
};
} // namespace detail

template <typename Input,
          typename Operator,
          cuda::std::ptrdiff_t MaxRedductionLength,
          typename Accum  = cuda::std::__accumulator_t<Operator, Input>,
          typename Output = Accum>
struct dist_interval
{
  // Values in the interval need to be representable in Input and if either Output or Accum are signed integers we want
  // to avoid UB.
  // If Accum is FP, we also want to avoid overflow b/c it breaks down pseudo-associativity.
  static constexpr Input min()
  {
    auto res = cuda::std::numeric_limits<Input>::lowest();
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Output>)
    {
      res =
        cuda::std::max(res, static_cast<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::min()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum> || cuda::std::is_floating_point_v<Accum>)
    {
      res = cuda::std::max(res, static_cast<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::min()));
    }
    return res;
  }
  static constexpr Input max()
  {
    auto res = cuda::std::numeric_limits<Input>::max();
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Output>)
    {
      res =
        cuda::std::min(res, static_cast<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::max()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum> || cuda::std::is_floating_point_v<Accum>)
    {
      res = cuda::std::min(res, static_cast<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::max()));
    }
    return res;
  }
};

/***********************************************************************************************************************
 * For testing for invalid values being passed to the binary operator
 **********************************************************************************************************************/

struct segment
{
  using offset_t = int32_t;
  // Make sure that default constructed segments can not be merged
  offset_t begin = cuda::std::numeric_limits<offset_t>::min();
  offset_t end   = cuda::std::numeric_limits<offset_t>::max();

  __host__ __device__ friend bool operator==(segment left, segment right)
  {
    return left.begin == right.begin && left.end == right.end;
  }

  // Needed for final comparison with reference
  friend std::ostream& operator<<(std::ostream& os, const segment& seg)
  {
    return os << "[ " << seg.begin << ", " << seg.end << " )";
  }
};

// Needed for data input using fancy iterators
struct tuple_to_segment_op
{
  __host__ __device__ segment operator()(cuda::std::tuple<segment::offset_t, segment::offset_t> interval)
  {
    const auto [begin, end] = interval;
    return {begin, end};
  }
};

// Actual scan operator doing the core test when run on device
struct merge_segments_op
{
  bool* error_flag_ptr;

  __device__ void check_inputs(segment left, segment right)
  {
    if (left.end != right.begin || left == right)
    {
      *error_flag_ptr = true;
    }
  }

  __host__ __device__ segment operator()(segment left, segment right)
  {
    NV_IF_TARGET(NV_IS_DEVICE, check_inputs(left, right););
    return {left.begin, right.end};
  }
};
