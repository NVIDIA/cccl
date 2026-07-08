// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/numeric>
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

// Narrow a bound computed in a wider Accum/Output domain down to the Input domain. A plain static_cast would wrap
// around when the value is outside Input's range (e.g. static_cast<int16_t>(INT_MAX) == -1), which can invert the
// resulting interval (min > max). Saturating instead clamps to Input's range, preserving the intersection.
template <typename To, typename From>
constexpr To clamp_to(From value)
{
  if constexpr (cuda::std::__cccl_is_integer_v<To> && cuda::std::__cccl_is_integer_v<From>)
  {
    return cuda::std::saturating_cast<To>(value);
  }
  else if constexpr (cuda::std::__cccl_is_integer_v<To> && cuda::std::is_floating_point_v<From>)
  {
    // Floating -> integer narrowing: a plain static_cast is UB when the value is outside To's range, so clamp first.
    if (value <= static_cast<From>(cuda::std::numeric_limits<To>::lowest()))
    {
      return cuda::std::numeric_limits<To>::lowest();
    }
    if (value >= static_cast<From>(cuda::std::numeric_limits<To>::max()))
    {
      return cuda::std::numeric_limits<To>::max();
    }
    return static_cast<To>(value);
  }
  else
  {
    return static_cast<To>(value);
  }
}
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
      res = cuda::std::max(
        res, detail::clamp_to<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::min()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum> || cuda::std::is_floating_point_v<Accum>)
    {
      res = cuda::std::max(res,
                           detail::clamp_to<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::min()));
    }
    return res;
  }
  static constexpr Input max()
  {
    auto res = cuda::std::numeric_limits<Input>::max();
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Output>)
    {
      res = cuda::std::min(
        res, detail::clamp_to<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::max()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum> || cuda::std::is_floating_point_v<Accum>)
    {
      res = cuda::std::min(res,
                           detail::clamp_to<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::max()));
    }
    return res;
  }
};

// Regression guard: these (Input, Op, num_items, Output) combinations used to compute inverted intervals (min > max)
// because a wider Accum/Output bound was narrowed to Input with a wrapping static_cast (e.g.
// static_cast<int16_t>(INT_MAX) == -1). An inverted interval makes Catch2's random(min, max) hang. These cover the
// integer-narrowing cases; they avoid log2/exp2 so they are always constant-evaluable across host compilers.
template <typename Input, typename Op, cuda::std::ptrdiff_t MaxReductionLength, typename Output>
using test_dist_interval = dist_interval<Input, Op, MaxReductionLength, cuda::std::__accumulator_t<Op, Input>, Output>;

// Primary reported case (int16_t input, wider int accumulator): assert the exact overflow-safe bounds, not just
// ordering, so a semantic regression (not only an inversion) is caught too.
static_assert(test_dist_interval<cuda::std::int16_t, cuda::std::plus<>, 16, cuda::std::int16_t>::min() == -2048);
static_assert(test_dist_interval<cuda::std::int16_t, cuda::std::plus<>, 16, cuda::std::int16_t>::max() == 2047);
// Further combinations that previously inverted (bit_and reaches the primary full-range specialization; the signed
// int8_t/int32_t pair is exercised by catch2_test_thread_scan_inclusive_partial.cu).
static_assert(test_dist_interval<cuda::std::int16_t, cuda::std::bit_and<>, 16, cuda::std::int16_t>::min()
              <= test_dist_interval<cuda::std::int16_t, cuda::std::bit_and<>, 16, cuda::std::int16_t>::max());
static_assert(test_dist_interval<cuda::std::int8_t, cuda::std::plus<>, 3, cuda::std::int32_t>::min()
              <= test_dist_interval<cuda::std::int8_t, cuda::std::plus<>, 3, cuda::std::int32_t>::max());

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
