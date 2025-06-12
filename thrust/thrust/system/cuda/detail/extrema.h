/*******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/system/cuda/config.h>

#include <cub/util_type.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/pair.h>
#include <thrust/system/cuda/detail/cdp_dispatch.h>
#include <thrust/system/cuda/detail/reduce.h>

#include <cuda/std/__algorithm_>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __extrema
{
template <class Predicate>
struct arg_min_f
{
  Predicate predicate;

  template <typename Value, typename Index>
  _CCCL_DEVICE auto operator()(tuple<Value, Index> const& lhs, tuple<Value, Index> const& rhs) -> tuple<Value, Index>
  {
    const auto& [lhs_value, lhs_index] = lhs;
    const auto& [rhs_value, rhs_index] = rhs;

    // compare the values
    if (predicate(lhs_value, rhs_value))
    {
      return lhs;
    }
    else if (predicate(rhs_value, lhs_value))
    {
      return rhs;
    }

    // values are equivalent, prefer smaller index
    if (lhs_index < rhs_index)
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <class Predicate>
struct arg_max_f
{
  Predicate predicate;

  template <typename Value, typename Index>
  _CCCL_DEVICE auto operator()(tuple<Value, Index> const& lhs, tuple<Value, Index> const& rhs) -> tuple<Value, Index>
  {
    const auto& [lhs_value, lhs_index] = lhs;
    const auto& [rhs_value, rhs_index] = rhs;

    // compare values first
    if (predicate(lhs_value, rhs_value))
    {
      return rhs;
    }
    else if (predicate(rhs_value, lhs_value))
    {
      return lhs;
    }

    // values are equivalent, prefer smaller index
    if (lhs_index < rhs_index)
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};

template <class Predicate>
struct arg_minmax_f
{
  Predicate predicate;

  template <typename ValueMin, typename ValueMax, typename IndexMin, typename IndexMax>
  _CCCL_DEVICE auto operator()(tuple<ValueMin, ValueMax, IndexMin, IndexMax> const& lhs,
                               tuple<ValueMin, ValueMax, IndexMin, IndexMax> const& rhs)
    -> tuple<ValueMin, ValueMax, IndexMin, IndexMax>
  {
    const auto& [lhs_value_min, lhs_value_max, lhs_index_min, lhs_index_max] = lhs;
    const auto& [rhs_value_min, rhs_value_max, rhs_index_min, rhs_index_max] = rhs;

    const auto [value_min, index_min] =
      arg_min_f<Predicate>{predicate}(tuple{lhs_value_min, lhs_index_min}, tuple{rhs_value_min, rhs_index_min});
    const auto [value_max, index_max] =
      arg_max_f<Predicate>{predicate}(tuple{lhs_value_max, lhs_index_max}, tuple{rhs_value_max, rhs_index_max});

    return tuple{value_min, value_max, index_min, index_max};
  }
};

template <class Derived, class ItemsIt, class ArgFunctor>
ItemsIt THRUST_RUNTIME_FUNCTION
element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, ArgFunctor arg_func)
{
  if (first == last)
  {
    return last;
  }

  using value_t  = thrust::detail::it_value_t<ItemsIt>;
  using offset_t = thrust::detail::it_difference_t<ItemsIt>;
  using tuple_t  = tuple<value_t, offset_t>;

  const auto num_items = static_cast<offset_t>(::cuda::std::distance(first, last));
  const auto zip_first = make_zip_iterator(first, counting_iterator<offset_t>{0});

  // TODO(bgruber): the previous thrust implementation avoided creating an initial value. Should we bring this back?
  // There is no reduction in CUB without init.
  const auto offset = thrust::cuda_cub::detail::reduce_n_impl(
    policy,
    zip_first,
    num_items,
    tuple_t{cub::FutureValue(first), offset_t{0}},
    arg_func,
    [](execution_policy<Derived>& policy, const tuple_t* result_ptr) {
      // TODO(bgruber): I only want to download the offset (element 1) and not the value (element 0), but how can I
      // legally form a pointer to that tuple element?
      // return get_value(policy, &thrust::get<1>(*result_ptr));
      return thrust::get<1>(get_value(policy, result_ptr));
    });

  return first + offset;
}
} // namespace __extrema

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<>>
ItemsIt _CCCL_HOST_DEVICE
min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  // FIXME(bgruber): we should not need to produce an initial value
  THRUST_CDP_DISPATCH((return __extrema::element(policy, first, last, __extrema::arg_min_f<BinaryPred>{binary_pred});),
                      (return thrust::min_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<>>
ItemsIt _CCCL_HOST_DEVICE
max_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  // FIXME(bgruber): we should not need to produce an initial value
  THRUST_CDP_DISPATCH((return __extrema::element(policy, first, last, __extrema::arg_max_f<BinaryPred>{binary_pred});),
                      (return thrust::max_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<>>
pair<ItemsIt, ItemsIt> _CCCL_HOST_DEVICE
minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  if (first == last)
  {
    return make_pair(last, last);
  }

  THRUST_CDP_DISPATCH(
    (using offset_t       = thrust::detail::it_difference_t<ItemsIt>;
     const auto num_items = static_cast<offset_t>(::cuda::std::distance(first, last));

     using counting_it    = counting_iterator<offset_t>;
     const auto zip_first = make_zip_iterator(first, first, counting_it{0}, counting_it{0});

     // TODO(bgruber): the previous thrust implementation avoided creating an initial value. Should we bring this back?
     // There is no reduction in CUB without init.
     using value_t   = thrust::detail::it_value_t<ItemsIt>;
     using tuple_t   = tuple<value_t, value_t, offset_t, offset_t>;
     const auto func = __extrema::arg_minmax_f<BinaryPred>{binary_pred};
     return thrust::cuda_cub::detail::reduce_n_impl(
       policy,
       zip_first,
       num_items,
       tuple_t{cub::FutureValue(first), cub::FutureValue(first), offset_t{0}, offset_t{0}},
       func,
       [&](execution_policy<Derived>& policy, const tuple_t* result_ptr) -> pair<ItemsIt, ItemsIt> {
         // TODO(bgruber): I only want to download the offsets (element 2/3) and not the values (element 0/1), but how
         // can I legally form a pointer to those tuple elements?
         const auto result = get_value(policy, result_ptr);
         return {first + get<2>(result), first + get<3>(result)};
       });),
    // CDP Sequential impl:
    (return thrust::minmax_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
}

} // namespace cuda_cub
THRUST_NAMESPACE_END
