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

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/system/cuda/config.h>

#  include <thrust/distance.h>
#  include <thrust/iterator/counting_iterator.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/cuda/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

// XXX forward declare to circumvent circular dependency
template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class T>
InputIt _CCCL_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value);

}; // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/cuda/detail/reduce.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

namespace __find_if
{

template <typename TupleType>
struct functor
{
  THRUST_DEVICE_FUNCTION TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (thrust::get<0>(lhs) && thrust::get<0>(rhs))
    {
      return TupleType(true, (::cuda::std::min)(thrust::get<1>(lhs), thrust::get<1>(rhs)));
    }
    else if (thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <class ValueType, class InputIt, class UnaryOp>
struct transform_input_iterator_t
{
  using self_t            = transform_input_iterator_t;
  using difference_type   = thrust::detail::it_difference_t<InputIt>;
  using value_type        = ValueType;
  using pointer           = void;
  using reference         = value_type;
  using iterator_category = ::cuda::std::random_access_iterator_tag;

  InputIt input;
  mutable UnaryOp op;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE transform_input_iterator_t(InputIt input, UnaryOp op)
      : input(input)
      , op(op)
  {}

  transform_input_iterator_t(const self_t&) = default;

  // UnaryOp might not be copy assignable, such as when it is a lambda.  Define
  // an explicit copy assignment operator that doesn't try to assign it.
  _CCCL_HOST_DEVICE self_t& operator=(const self_t& o)
  {
    input = o.input;
    return *this;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator++(int)
  {
    self_t retval = *this;
    ++input;
    return retval;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator++()
  {
    ++input;
    return *this;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*() const
  {
    thrust::detail::it_value_t<InputIt> x = *input;
    return op(x);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator*()
  {
    thrust::detail::it_value_t<InputIt> x = *input;
    return op(x);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator+(difference_type n) const
  {
    return self_t(input + n, op);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t& operator+=(difference_type n)
  {
    input += n;
    return *this;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t operator-(difference_type n) const
  {
    return self_t(input - n, op);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_t& operator-=(difference_type n)
  {
    input -= n;
    return *this;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_t other) const
  {
    return input - other.input;
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference operator[](difference_type n) const
  {
    return op(input[n]);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_t& rhs) const
  {
    return (input == rhs.input);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_t& rhs) const
  {
    return (input != rhs.input);
  }
};
} // namespace __find_if

template <class Derived, class InputIt, class Size, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_n(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  using result_type = typename thrust::tuple<bool, Size>;

  // empty sequence
  if (num_items == 0)
  {
    return first;
  }

  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found
  //
  // XXX compose find_if from a look-back prefix scan algorithm
  //     and abort kernel when the first element is found

  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const Size interval_threshold = 1 << 20;
  const Size interval_size      = (::cuda::std::min)(interval_threshold, num_items);

  // FIXME(bgruber): we should also be able to use transform_iterator here, but it makes nvc++ hang. See:
  // https://github.com/NVIDIA/cccl/issues/3594. The problem does not occur with nvcc, so we could not add a test :/
  using XfrmIterator = __find_if::transform_input_iterator_t<bool, InputIt, Predicate>;
  // using XfrmIterator  = transform_iterator<Predicate, InputIt>;
  using IteratorTuple = thrust::tuple<XfrmIterator, counting_iterator<Size>>;
  using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

  IteratorTuple iter_tuple = thrust::make_tuple(XfrmIterator(first, predicate), counting_iterator<Size>(0));

  ZipIterator begin = thrust::make_zip_iterator(iter_tuple);
  ZipIterator end   = begin + num_items;

  for (ZipIterator interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    ZipIterator interval_end = interval_begin + interval_size;
    if (end < interval_end)
    {
      interval_end = end;
    } // end if

    result_type result = reduce(
      policy, interval_begin, interval_end, result_type(false, interval_end - begin), __find_if::functor<result_type>());

    // see if we found something
    if (thrust::get<0>(result))
    {
      return first + thrust::get<1>(result);
    }
  }

  // nothing was found if we reach here...
  return first + num_items;
}

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return cuda_cub::find_if_n(policy, first, ::cuda::std::distance(first, last), predicate);
}

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return cuda_cub::find_if(policy, first, last, ::cuda::std::not_fn(predicate));
}

template <class Derived, class InputIt, class T>
InputIt _CCCL_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value)
{
  using thrust::placeholders::_1;

  return cuda_cub::find_if(policy, first, last, _1 == value);
}

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
