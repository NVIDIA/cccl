// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <thrust/system/cuda/config.h>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/extrema.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/discard_iterator.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __extrema
{
template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt CUB_RUNTIME_FUNCTION
cub_min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  cudaStream_t stream      = cuda_cub::stream(policy);
  using offset_t           = thrust::detail::it_difference_t<ItemsIt>;
  const offset_t num_items = ::cuda::std::distance(first, last);

  if (num_items == 0)
  {
    return last;
  }

  ::cuda::std::size_t tmp_size = 0;
  auto error                   = cub::DeviceReduce::ArgMin(
    nullptr,
    tmp_size,
    first,
    ::cuda::discard_iterator{},
    static_cast<offset_t*>(nullptr),
    num_items,
    binary_pred,
    stream);
  throw_on_error(error, "min_element failed to allocate temporary storages");

  // We allocate both the temporary storage needed for the algorithm, and a `size_type` to store the result.
  thrust::detail::temporary_array<char, Derived> tmp(policy, sizeof(offset_t) + tmp_size);
  offset_t* index_ptr = thrust::detail::aligned_reinterpret_cast<offset_t*>(tmp.data().get());
  auto tmp_ptr        = static_cast<void*>(tmp.data().get() + sizeof(offset_t));

  error = cub::DeviceReduce::ArgMin(
    tmp_ptr, tmp_size, first, ::cuda::discard_iterator{}, index_ptr, num_items, binary_pred, stream);
  cuda_cub::throw_on_error(error, "min_element failed to launch cub::DeviceReduce::ArgMin");

  cuda_cub::throw_on_error(cuda_cub::synchronize(policy), "min_element failed to synchronize");

  return first + get_value(policy, index_ptr);
}

template <typename OffsetT, typename T>
struct minmax_accum_t
{
  ::cuda::std::pair<OffsetT, T> min, max;
};

template <typename OffsetT, typename T>
struct minmax_load_transformation
{
  // convert from zip_iterator
  template <typename TRef>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(::cuda::std::tuple<OffsetT, TRef> input) const
    -> minmax_accum_t<OffsetT, T>
  {
    auto p = ::cuda::std::pair<OffsetT, T>{::cuda::std::get<0>(input), ::cuda::std::get<1>(input)};
    return {p, p};
  }
};

template <typename OffsetT>
struct output_t
{
  OffsetT min_offset;
  OffsetT max_offset;

  output_t() = default;

  // convert from accumulator type (during assignment at the end of the kernel)
  template <typename T>
  _CCCL_API _CCCL_FORCEINLINE output_t(minmax_accum_t<OffsetT, T> result)
      : min_offset(result.min.first)
      , max_offset(result.max.first)
  {}
};

template <typename OffsetT, typename T, typename ValueLessThen = ::cuda::std::less<>>
struct minmax_reduce_op : ValueLessThen
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
  operator()(const minmax_accum_t<OffsetT, T>& a, const minmax_accum_t<OffsetT, T>& b) const
    -> minmax_accum_t<OffsetT, T>
  {
    const auto& less = static_cast<const ValueLessThen&>(*this);
    const auto min   = cub::detail::arg_less<ValueLessThen>{less}(a.min, b.min);
    const auto max   = cub::detail::arg_less<cub::detail::swap_args<ValueLessThen>>{less}(a.max, b.max);
    return {min, max};
  }

  // needed for __accumulator_t, never called at runtime
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(
    const cub::detail::reduce::empty_problem_init_t<output_t<OffsetT>>&, const minmax_accum_t<OffsetT, T>&) const
    -> minmax_accum_t<OffsetT, T>;
};

template <class Derived, class ItemsIt, class BinaryPred>
::cuda::std::pair<ItemsIt, ItemsIt> CUB_RUNTIME_FUNCTION
cub_minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  cudaStream_t stream      = cuda_cub::stream(policy);
  using offset_t           = thrust::detail::it_difference_t<ItemsIt>;
  const offset_t num_items = ::cuda::std::distance(first, last);

  if (num_items == 0)
  {
    return {first, first};
  }

  using input_t = thrust::detail::it_value_t<ItemsIt>;
  auto indexed_first =
    ::cuda::make_zip_iterator(::cuda::counting_iterator<offset_t>(0), thrust::try_unwrap_contiguous_iterator(first));
  auto reduction_op = minmax_reduce_op<offset_t, input_t, BinaryPred>{binary_pred};
  auto transform_op = minmax_load_transformation<offset_t, input_t>{};
  using output_t    = output_t<offset_t>;
  const auto init   = cub::detail::reduce::empty_problem_init_t<output_t>{};

  size_t tmp_size = 0;
  auto error      = cub::DeviceReduce::TransformReduce(
    nullptr,
    tmp_size,
    indexed_first,
    static_cast<output_t*>(nullptr),
    num_items,
    reduction_op,
    transform_op,
    init,
    stream);
  throw_on_error(error, "minmax_element failed to allocate temporary storages");

  // We allocate both the temporary storage needed for the algorithm, and a `size_type` to store the result.
  thrust::detail::temporary_array<char, Derived> tmp(policy, sizeof(output_t) + tmp_size);
  output_t* out_ptr = thrust::detail::aligned_reinterpret_cast<output_t*>(tmp.data().get());
  void* tmp_ptr     = static_cast<void*>(tmp.data().get() + sizeof(output_t));

  error = cub::DeviceReduce::TransformReduce(
    tmp_ptr, tmp_size, indexed_first, out_ptr, num_items, reduction_op, transform_op, init, stream);
  cuda_cub::throw_on_error(error, "minmax_element failed to launch cub::DeviceReduce::ArgMin");

  cuda_cub::throw_on_error(cuda_cub::synchronize(policy), "min_element failed to synchronize");

  const auto [min_offset, max_offset] = get_value(policy, out_ptr);
  return {first + min_offset, first + max_offset};
}
} // namespace __extrema

/// min element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
ItemsIt _CCCL_HOST_DEVICE
min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(({ return __extrema::cub_min_element(policy, first, last, binary_pred); }),
                      ({ return thrust::min_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}

/// max element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
ItemsIt _CCCL_HOST_DEVICE
max_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(
    ({ return __extrema::cub_min_element(policy, first, last, cub::detail::swap_args{binary_pred}); }),
    ({ return thrust::max_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}

/// minmax element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
::cuda::std::pair<ItemsIt, ItemsIt> _CCCL_HOST_DEVICE
minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(({ return __extrema::cub_minmax_element(policy, first, last, binary_pred); }),
                      ({ return thrust::minmax_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
