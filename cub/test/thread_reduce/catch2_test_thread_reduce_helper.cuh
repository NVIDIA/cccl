#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <ostream>

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

// Replace with identity_v once #4312 lands
template <typename T, typename Operator, typename = void>
struct cub_operator_to_identity;

template <typename T>
struct cub_operator_to_identity<T, _CUDA_VSTD::plus<>>
{
  static constexpr T value()
  {
    return T{};
  }
};

template <typename T>
struct cub_operator_to_identity<T, _CUDA_VSTD::multiplies<>>
{
  static constexpr T value()
  {
    return T{1};
  }
};

template <typename T>
struct cub_operator_to_identity<T, _CUDA_VSTD::bit_and<>>
{
  static constexpr T value()
  {
    return static_cast<T>(~T{0});
  }
};

template <typename T>
struct cub_operator_to_identity<T, _CUDA_VSTD::bit_or<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, _CUDA_VSTD::bit_xor<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::minimum<>>
{
  static constexpr T value()
  {
    return _CUDA_VSTD::numeric_limits<T>::max();
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::maximum<>>
{
  static constexpr T value()
  {
    return _CUDA_VSTD::numeric_limits<T>::lowest();
  }
};

/***********************************************************************************************************************
 * Sensible Distribution Intervals for Test Data
 **********************************************************************************************************************/

namespace detail
{

template <typename T, typename Operator, _CUDA_VSTD::ptrdiff_t MaxReductionLength, typename = void>
struct dist_interval
{
  static constexpr T min()
  {
    return _CUDA_VSTD::numeric_limits<T>::lowest();
  }
  static constexpr T max()
  {
    return _CUDA_VSTD::numeric_limits<T>::max();
  }
};

template <typename T, _CUDA_VSTD::ptrdiff_t MaxReductionLength>
struct dist_interval<
  T,
  _CUDA_VSTD::plus<>,
  MaxReductionLength,
  _CUDA_VSTD::enable_if_t<_CUDA_VSTD::__cccl_is_signed_integer_v<T> || _CUDA_VSTD::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf destroying pseudo-associativity
  static constexpr T min()
  {
    return static_cast<T>(_CUDA_VSTD::numeric_limits<T>::lowest() / MaxReductionLength);
  }
  static constexpr T max()
  {
    return static_cast<T>(_CUDA_VSTD::numeric_limits<T>::max() / MaxReductionLength);
  }
};

template <typename T, _CUDA_VSTD::ptrdiff_t MaxReductionLength>
struct dist_interval<
  T,
  _CUDA_VSTD::multiplies<>,
  MaxReductionLength,
  _CUDA_VSTD::enable_if_t<_CUDA_VSTD::__cccl_is_signed_integer_v<T> || _CUDA_VSTD::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf destroying pseudo-associativity
  // Use floating point arithmetic to avoid unnecessarily small interval.
  static constexpr T min()
  {
    const double log2_abs_min = _CUDA_VSTD::log2(_CUDA_VSTD::fabs(_CUDA_VSTD::numeric_limits<T>::lowest()));
    return static_cast<T>(-_CUDA_VSTD::exp2(log2_abs_min / MaxReductionLength));
  }
  static constexpr T max()
  {
    const double log2_max = _CUDA_VSTD::log2(_CUDA_VSTD::numeric_limits<T>::max());
    return static_cast<T>(_CUDA_VSTD::exp2(log2_max / MaxReductionLength));
  }
};

} // namespace detail

template <typename Input,
          typename Operator,
          _CUDA_VSTD::ptrdiff_t MaxRedductionLength,
          typename Accum  = _CUDA_VSTD::__accumulator_t<Operator, Input>,
          typename Output = Accum>
struct dist_interval
{
  // Values in the interval need to be representable in Input and if either Output or Accum are signed integers we want
  // to avoid UB.
  // If Accum is FP, we also want to avoid overflow b/c it breaks down pseudo-associativity.
  static constexpr Input min()
  {
    auto res = _CUDA_VSTD::numeric_limits<Input>::lowest();
    if constexpr (_CUDA_VSTD::__cccl_is_signed_integer_v<Output>)
    {
      res =
        _CUDA_VSTD::max(res, static_cast<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::min()));
    }
    if constexpr (_CUDA_VSTD::__cccl_is_signed_integer_v<Accum> || _CUDA_VSTD::is_floating_point_v<Accum>)
    {
      res =
        _CUDA_VSTD::max(res, static_cast<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::min()));
    }
    return res;
  }
  static constexpr Input max()
  {
    auto res = _CUDA_VSTD::numeric_limits<Input>::max();
    if constexpr (_CUDA_VSTD::__cccl_is_signed_integer_v<Output>)
    {
      res =
        _CUDA_VSTD::min(res, static_cast<Input>(detail::dist_interval<Output, Operator, MaxRedductionLength>::max()));
    }
    if constexpr (_CUDA_VSTD::__cccl_is_signed_integer_v<Accum> || _CUDA_VSTD::is_floating_point_v<Accum>)
    {
      res =
        _CUDA_VSTD::min(res, static_cast<Input>(detail::dist_interval<Accum, Operator, MaxRedductionLength>::max()));
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
  offset_t begin = _CUDA_VSTD::numeric_limits<offset_t>::min();
  offset_t end   = _CUDA_VSTD::numeric_limits<offset_t>::max();

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
  __host__ __device__ segment operator()(_CUDA_VSTD::tuple<segment::offset_t, segment::offset_t> interval)
  {
    const auto [begin, end] = interval;
    return {begin, end};
  }
};

// Actual scan operator doing the core test when run on device
struct merge_segments_op
{
  __host__ merge_segments_op(bool* error_flag_ptr)
      : error_flag_ptr_{error_flag_ptr}
  {}

  __device__ void check_inputs(segment left, segment right)
  {
    if (left.end != right.begin || left == right)
    {
      *error_flag_ptr_ = true;
    }
  }

  __host__ __device__ segment operator()(segment left, segment right)
  {
    NV_IF_TARGET(NV_IS_DEVICE, check_inputs(left, right););
    return {left.begin, right.end};
  }

  bool* error_flag_ptr_;
};
