/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER
#  include <thrust/system/cuda/config.h>

#  include <thrust/distance.h>
#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/pair.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/zip_function.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class InputIt1, class InputIt2, class BinaryPred>
pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPred binary_pred);

template <class Derived, class InputIt1, class InputIt2>
pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2);
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/system/cuda/detail/find.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{
template <class ValueType, class InputIt1, class InputIt2, class BinaryOp>
struct transform_pair_of_input_iterators_t
{
  using self_t            = transform_pair_of_input_iterators_t;
  using difference_type   = typename iterator_traits<InputIt1>::difference_type;
  using value_type        = ValueType;
  using pointer           = void;
  using reference         = value_type;
  using iterator_category = std::random_access_iterator_tag;

  InputIt1 input1;
  InputIt2 input2;
  mutable BinaryOp op;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  transform_pair_of_input_iterators_t(InputIt1 input1_, InputIt2 input2_, BinaryOp op_)
      : input1(input1_)
      , input2(input2_)
      , op(op_)
  {}

  transform_pair_of_input_iterators_t(const self_t&) = default;

  // BinaryOp might not be copy assignable, such as when it is a lambda.
  // Define an explicit copy assignment operator that doesn't try to assign it.
  _CCCL_HOST_DEVICE self_t& operator=(const self_t& o)
  {
    input1 = o.input1;
    input2 = o.input2;
    return *this;
  }

  /// Postfix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator++(int)
  {
    self_t retval = *this;
    ++input1;
    ++input2;
    return retval;
  }

  /// Prefix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator++()
  {
    ++input1;
    ++input2;
    return *this;
  }

  /// Indirection
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*() const
  {
    return op(*input1, *input2);
  }
  /// Indirection
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*()
  {
    return op(*input1, *input2);
  }

  /// Addition
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator+(difference_type n) const
  {
    return self_t(input1 + n, input2 + n, op);
  }

  /// Addition assignment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t& operator+=(difference_type n)
  {
    input1 += n;
    input2 += n;
    return *this;
  }

  /// Subtraction
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator-(difference_type n) const
  {
    return self_t(input1 - n, input2 - n, op);
  }

  /// Subtraction assignment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t& operator-=(difference_type n)
  {
    input1 -= n;
    input2 -= n;
    return *this;
  }

  /// Distance
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_t other) const
  {
    return input1 - other.input1;
  }

  /// Array subscript
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator[](difference_type n) const
  {
    return op(input1[n], input2[n]);
  }

  /// Equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_t& rhs) const
  {
    return (input1 == rhs.input1) && (input2 == rhs.input2);
  }

  /// Not equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_t& rhs) const
  {
    return (input1 != rhs.input1) || (input2 != rhs.input2);
  }

}; // struct transform_pair_of_input_iterators_t
} // namespace detail

template <class Derived, class InputIt1, class InputIt2, class BinaryPred>
pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPred binary_pred)
{
  const auto transform_first =
    detail::transform_pair_of_input_iterators_t<bool, InputIt1, InputIt2, BinaryPred>(first1, first2, binary_pred);
  const auto result = cuda_cub::find_if_not(
    policy, transform_first, transform_first + thrust::distance(first1, last1), ::cuda::std::__identity{});
  return thrust::make_pair(first1 + thrust::distance(transform_first, result),
                           first2 + thrust::distance(transform_first, result));

  // FIXME(bgruber): the following code should be equivalent and not require a dedicated iterator. However, it
  // additionally requires the value_type to constructible/destructible on the device, which should be fixed at some
  // point. See also: https://github.com/NVIDIA/cccl/issues/3591
#  if 0
  const auto n            = thrust::distance(first1, last1);
  const auto first        = make_zip_iterator(first1, first2);
  const auto last         = make_zip_iterator(last1, first2 + n);
  const auto mismatch_pos = cuda_cub::find_if_not(policy, first, last, make_zip_function(binary_pred));
  const auto dist         = thrust::distance(first, mismatch_pos);
  return thrust::make_pair(first1 + dist, first2 + dist);
#  endif
}

template <class Derived, class InputIt1, class InputIt2>
pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
  using InputType1 = typename thrust::iterator_value<InputIt1>::type;
  return cuda_cub::mismatch(policy, first1, last1, first2, equal_to<InputType1>());
}

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
