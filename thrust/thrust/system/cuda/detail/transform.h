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

#  include <cub/device/device_transform.cuh>

#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/zip_function.h>

#  include <cuda/__functional/address_stability.h>
#  include <cuda/std/__algorithm/transform.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace __transform
{
_CCCL_EXEC_CHECK_DISABLE
template <class Derived,
          class Offset,
          class... InputIts,
          class OutputIt,
          class TransformOp,
          class Predicate = cub::detail::transform::always_true_predicate>
OutputIt THRUST_FUNCTION cub_transform_many(
  execution_policy<Derived>& policy,
  ::cuda::std::tuple<InputIts...> firsts,
  OutputIt result,
  Offset num_items,
  TransformOp transform_op,
  Predicate pred = {})
{
  if (num_items == 0)
  {
    return result;
  }

  constexpr auto stable_address =
    (::cuda::proclaims_copyable_arguments<Predicate>::value && ::cuda::proclaims_copyable_arguments<TransformOp>::value)
      ? cub::detail::transform::requires_stable_address::no
      : cub::detail::transform::requires_stable_address::yes;

  cudaError_t status;
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    (cub::detail::transform::dispatch_t<stable_address,
                                        decltype(num_items_fixed),
                                        ::cuda::std::tuple<InputIts...>,
                                        OutputIt,
                                        Predicate,
                                        TransformOp>::dispatch),
    num_items,
    (firsts, result, num_items_fixed, pred, transform_op, cuda_cub::stream(policy)));
  throw_on_error(status, "transform: failed inside CUB");

  status = cuda_cub::synchronize_optional(policy);
  throw_on_error(status, "transform: failed to synchronize");

  return result + num_items;
}

// unwrap zip_iterator and zip_function into their underlying iterators so cub::DeviceTransform can optimize them
template <class Derived, class Offset, class... InputIts, class OutputIt, class TransformOp>
OutputIt THRUST_FUNCTION cub_transform_many(
  execution_policy<Derived>& policy,
  ::cuda::std::tuple<zip_iterator<::cuda::std::tuple<InputIts...>>> firsts,
  OutputIt result,
  Offset num_items,
  zip_function<TransformOp> transform_op)
{
  return cub_transform_many(
    policy, get<0>(firsts).get_iterator_tuple(), result, num_items, transform_op.underlying_function());
}

template <typename F>
struct raw_reference_cast_args
{
  mutable F f; // mutable to support non-const F::operator()

  template <typename... Ts>
  THRUST_FUNCTION decltype(auto) operator()(Ts&&... args) const
  {
    return f(raw_reference_cast(::cuda::std::forward<Ts>(args))...);
  }
};

template <typename F>
struct forward_head
{
  mutable F f; // mutable to support non-const F::operator()

  template <typename Head, typename... Tail>
  THRUST_FUNCTION decltype(auto) operator()(Head&& head, Tail&&...) const
  {
    return f(::cuda::std::forward<Head>(head));
  }
};

template <typename F>
struct forward_tail
{
  mutable F f; // mutable to support non-const F::operator()

  template <typename Head, typename... Tail>
  THRUST_FUNCTION decltype(auto) operator()(Head&&, Tail&&... tail) const
  {
    return f(::cuda::std::forward<Tail>(tail)...);
  }
};
} // namespace __transform

//  one input data stream

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_FUNCTION OutputIt
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy, ::cuda::std::make_tuple(first), result, ::cuda::std::distance(first, last), transform_op);),
    (return ::cuda::std::transform(
              first, last, result, __transform::raw_reference_cast_args<TransformOp>{transform_op});));
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp>
THRUST_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt first,
  ::cuda::std::iter_difference_t<InputIt> num_items,
  OutputIt result,
  TransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(policy, ::cuda::std::make_tuple(first), result, num_items, transform_op);),
    (return ::cuda::std::transform(
              first, first + num_items, result, __transform::raw_reference_cast_args<TransformOp>{transform_op});));
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp, typename Predicate>
THRUST_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(first),
              result,
              ::cuda::std::distance(first, last),
              transform_op,
              predicate);),
    (while (first != last) {
      if (predicate(raw_reference_cast(*first)))
      {
        *result = transform_op(raw_reference_cast(*first));
      }
      ++first;
      ++result;
    } return result;));
}

template <typename Derived, typename InputIt, typename OutputIt, typename TransformOp, typename Predicate>
THRUST_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  ::cuda::std::iter_difference_t<InputIt> num_items,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH((return __transform::cub_transform_many(
                                policy, ::cuda::std::make_tuple(first), result, num_items, transform_op, predicate);),
                      (for (decltype(num_items) i = 0; i < num_items; i++) {
                        if (predicate(raw_reference_cast(*first)))
                        {
                          *result = transform_op(raw_reference_cast(*first));
                        }
                        ++first;
                        ++result;
                      } return result;));
}

//  one input data stream + stencil

template <class Derived, class InputIt, class OutputIt, class StencilInputIt, class TransformOp, class Predicate>
THRUST_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(stencil, first),
              result,
              ::cuda::std::distance(first, last),
              __transform::forward_tail<TransformOp>{transform_op},
              __transform::forward_head<Predicate>{predicate});),
    (while (first != last) {
      if (predicate(raw_reference_cast(*stencil)))
      {
        *result = transform_op(raw_reference_cast(*first));
      }
      ++first;
      ++result;
    } return result;));
}

template <typename Derived,
          typename InputIt,
          typename StencilInputIt,
          typename OutputIt,
          typename TransformOp,
          typename Predicate>
THRUST_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt first,
  ::cuda::std::iter_difference_t<InputIt> num_items,
  StencilInputIt stencil,
  OutputIt result,
  TransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(stencil, first),
              result,
              num_items,
              __transform::forward_tail<TransformOp>{transform_op},
              __transform::forward_head<Predicate>{predicate});),
    (for (decltype(num_items) i = 0; i < num_items; i++) {
      if (predicate(raw_reference_cast(*stencil)))
      {
        *result = transform_op(raw_reference_cast(*first));
      }
      ++first;
      ++result;
    } return result;));
}

// two input data streams

template <typename Derived, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryTransformOp>
THRUST_FUNCTION OutputIt transform(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(first1, first2),
              result,
              ::cuda::std::distance(first1, last1),
              transform_op);),
    (return ::cuda::std::transform(
              first1, last1, first2, result, __transform::raw_reference_cast_args<BinaryTransformOp>{transform_op});));
}

template <typename Derived, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryTransformOp>
THRUST_FUNCTION OutputIt transform_n(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  ::cuda::std::iter_difference_t<InputIt1> num_items,
  InputIt2 first2,
  OutputIt result,
  BinaryTransformOp transform_op)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy, ::cuda::std::make_tuple(first1, first2), result, num_items, transform_op);),
    (return ::cuda::std::transform(first1,
                                   first1 + num_items,
                                   first2,
                                   result,
                                   __transform::raw_reference_cast_args<BinaryTransformOp>{transform_op});));
}

// two input data streams + stencil

template <typename Derived,
          typename InputIt1,
          typename InputIt2,
          typename StencilInputIt,
          typename OutputIt,
          typename BinaryTransformOp,
          typename Predicate>
THRUST_FUNCTION OutputIt transform_if(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  StencilInputIt stencil,
  OutputIt result,
  BinaryTransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(stencil, first1, first2),
              result,
              ::cuda::std::distance(first1, last1),
              __transform::forward_tail<BinaryTransformOp>{transform_op},
              __transform::forward_head<Predicate>{predicate});),
    (while (first1 != last1) {
      if (predicate(raw_reference_cast(*stencil)))
      {
        *result = transform_op(raw_reference_cast(*first1), raw_reference_cast(*first2));
      }
      ++first1;
      ++first2;
      ++result;
    } return result;));
}

template <typename Derived,
          typename InputIt1,
          typename InputIt2,
          typename StencilInputIt,
          typename OutputIt,
          typename BinaryTransformOp,
          typename Predicate>
THRUST_FUNCTION OutputIt transform_if_n(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  ::cuda::std::iter_difference_t<InputIt1> num_items,
  InputIt2 first2,
  StencilInputIt stencil,
  OutputIt result,
  BinaryTransformOp transform_op,
  Predicate predicate)
{
  THRUST_CDP_DISPATCH(
    (return __transform::cub_transform_many(
              policy,
              ::cuda::std::make_tuple(stencil, first1, first2),
              result,
              num_items,
              __transform::forward_tail<BinaryTransformOp>{transform_op},
              __transform::forward_head<Predicate>{predicate});),
    (for (decltype(num_items) i = 0; i < num_items; i++) {
      if (predicate(raw_reference_cast(*stencil)))
      {
        *result = transform_op(raw_reference_cast(*first1), raw_reference_cast(*first2));
      }
      ++first1;
      ++first2;
      ++result;
    } return result;));
}

} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
