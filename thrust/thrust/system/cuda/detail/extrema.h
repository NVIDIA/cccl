/*******************************************************************************
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

#if _CCCL_CUDA_COMPILATION()

#  include <thrust/system/cuda/config.h>

#  include <cub/util_math.cuh>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/extrema.h>
#  include <thrust/iterator/counting_iterator.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/reduce.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __extrema
{
template <class InputType, class IndexType, class Predicate>
struct arg_min_f
{
  Predicate predicate;
  using pair_type = tuple<InputType, IndexType>;

  _CCCL_HOST_DEVICE arg_min_f(Predicate p)
      : predicate(p)
  {}

  pair_type _CCCL_DEVICE operator()(pair_type const& lhs, pair_type const& rhs)
  {
    InputType const& rhs_value = get<0>(rhs);
    InputType const& lhs_value = get<0>(lhs);
    IndexType const& rhs_key   = get<1>(rhs);
    IndexType const& lhs_key   = get<1>(lhs);

    // check values first
    if (predicate(lhs_value, rhs_value))
    {
      return lhs;
    }
    else if (predicate(rhs_value, lhs_value))
    {
      return rhs;
    }

    // values are equivalent, prefer smaller index
    if (lhs_key < rhs_key)
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
}; // struct arg_min_f

template <class InputType, class IndexType, class Predicate>
struct arg_max_f
{
  Predicate predicate;
  using pair_type = tuple<InputType, IndexType>;

  _CCCL_HOST_DEVICE arg_max_f(Predicate p)
      : predicate(p)
  {}

  pair_type _CCCL_DEVICE operator()(pair_type const& lhs, pair_type const& rhs)
  {
    InputType const& rhs_value = get<0>(rhs);
    InputType const& lhs_value = get<0>(lhs);
    IndexType const& rhs_key   = get<1>(rhs);
    IndexType const& lhs_key   = get<1>(lhs);

    // check values first
    if (predicate(lhs_value, rhs_value))
    {
      return rhs;
    }
    else if (predicate(rhs_value, lhs_value))
    {
      return lhs;
    }

    // values are equivalent, prefer smaller index
    if (lhs_key < rhs_key)
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
}; // struct arg_max_f

template <class InputType, class IndexType, class Predicate>
struct arg_minmax_f
{
  Predicate predicate;

  using pair_type      = tuple<InputType, IndexType>;
  using two_pairs_type = tuple<pair_type, pair_type>;

  using arg_min_t = arg_min_f<InputType, IndexType, Predicate>;
  using arg_max_t = arg_max_f<InputType, IndexType, Predicate>;

  _CCCL_HOST_DEVICE arg_minmax_f(Predicate p)
      : predicate(p)
  {}

  two_pairs_type _CCCL_DEVICE operator()(two_pairs_type const& lhs, two_pairs_type const& rhs)
  {
    pair_type const& rhs_min = get<0>(rhs);
    pair_type const& lhs_min = get<0>(lhs);
    pair_type const& rhs_max = get<1>(rhs);
    pair_type const& lhs_max = get<1>(lhs);

    auto result = thrust::make_tuple(arg_min_t(predicate)(lhs_min, rhs_min), arg_max_t(predicate)(lhs_max, rhs_max));

    return result;
  }

  struct duplicate_tuple
  {
    _CCCL_DEVICE two_pairs_type operator()(pair_type const& t)
    {
      return thrust::make_tuple(t, t);
    }
  };
}; // struct arg_minmax_f

template <class T, class InputIt, class OutputIt, class Size, class ReductionOp>
cudaError_t THRUST_RUNTIME_FUNCTION doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIt input_it,
  Size num_items,
  ReductionOp reduction_op,
  OutputIt output_it,
  cudaStream_t stream)
{
  using core::detail::AgentLauncher;
  using core::detail::AgentPlan;
  using core::detail::cuda_optional;
  using core::detail::get_agent_plan;

  using UnsignedSize = typename detail::make_unsigned_special<Size>::type;

  if (num_items == 0)
  {
    return cudaErrorNotSupported;
  }

  using reduce_agent = AgentLauncher<__reduce::ReduceAgent<InputIt, OutputIt, T, Size, ReductionOp>>;

  typename reduce_agent::Plan reduce_plan = reduce_agent::get_plan(stream);

  cudaError_t status = cudaSuccess;

  if (num_items <= reduce_plan.items_per_tile)
  {
    size_t vshmem_size = core::detail::vshmem_size(reduce_plan.shared_memory_size, 1);

    // small, single tile size
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = max<size_t>(1, vshmem_size);
      return status;
    }
    char* vshmem_ptr = vshmem_size > 0 ? (char*) d_temp_storage : nullptr;

    reduce_agent ra(reduce_plan, num_items, stream, vshmem_ptr, "reduce_agent: single_tile only");
    ra.launch(input_it, output_it, num_items, reduction_op);
    _CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
  }
  else
  {
    // regular size
    cuda_optional<int> sm_count = core::detail::get_sm_count();
    _CUDA_CUB_RET_IF_FAIL(sm_count.status());

    // reduction will not use more cta counts than requested
    cuda_optional<int> max_blocks_per_sm = reduce_agent::template get_max_blocks_per_sm<
      InputIt,
      OutputIt,
      Size,
      cub::GridEvenShare<Size>,
      cub::GridQueue<UnsignedSize>,
      ReductionOp>(reduce_plan);
    _CUDA_CUB_RET_IF_FAIL(max_blocks_per_sm.status());

    int reduce_device_occupancy = (int) max_blocks_per_sm * sm_count;

    int sm_oversubscription = 5;
    int max_blocks          = reduce_device_occupancy * sm_oversubscription;

    cub::GridEvenShare<Size> even_share;
    even_share.DispatchInit(num_items, max_blocks, reduce_plan.items_per_tile);

    // we will launch at most "max_blocks" blocks in a grid
    // so preallocate virtual shared memory storage for this if required
    //
    size_t vshmem_size = core::detail::vshmem_size(reduce_plan.shared_memory_size, max_blocks);

    // Temporary storage allocation requirements
    void* allocations[3]       = {nullptr, nullptr, nullptr};
    size_t allocation_sizes[3] = {
      max_blocks * sizeof(T), // bytes needed for privatized block reductions
      cub::GridQueue<UnsignedSize>::AllocationSize(), // bytes needed for grid queue descriptor0
      vshmem_size // size of virtualized shared memory storage
    };
    status = cub::detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    _CUDA_CUB_RET_IF_FAIL(status);
    if (d_temp_storage == nullptr)
    {
      return status;
    }

    T* d_block_reductions = (T*) allocations[0];
    cub::GridQueue<UnsignedSize> queue(allocations[1]);
    char* vshmem_ptr = vshmem_size > 0 ? (char*) allocations[2] : nullptr;

    // Get grid size for device_reduce_sweep_kernel
    int reduce_grid_size = 0;
    if (reduce_plan.grid_mapping == cub::GRID_MAPPING_RAKE)
    {
      // Work is distributed evenly
      reduce_grid_size = even_share.grid_size;
    }
    else if (reduce_plan.grid_mapping == cub::GRID_MAPPING_DYNAMIC)
    {
      // Work is distributed dynamically
      size_t num_tiles = ::cuda::ceil_div(num_items, reduce_plan.items_per_tile);

      // if not enough to fill the device with threadblocks
      // then fill the device with threadblocks
      reduce_grid_size = static_cast<int>((min) (num_tiles, static_cast<size_t>(reduce_device_occupancy)));

      using drain_agent    = AgentLauncher<__reduce::DrainAgent<Size>>;
      AgentPlan drain_plan = drain_agent::get_plan();
      drain_plan.grid_size = 1;
      drain_agent da(drain_plan, stream, "__reduce::drain_agent");
      da.launch(queue, num_items);
      _CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
    }
    else
    {
      _CUDA_CUB_RET_IF_FAIL(cudaErrorNotSupported);
    }

    reduce_plan.grid_size = reduce_grid_size;
    reduce_agent ra(reduce_plan, stream, vshmem_ptr, "reduce_agent: regular size reduce");
    ra.launch(input_it, d_block_reductions, num_items, even_share, queue, reduction_op);
    _CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

    using reduce_agent_single = AgentLauncher<__reduce::ReduceAgent<T*, OutputIt, T, Size, ReductionOp>>;

    reduce_plan.grid_size = 1;
    reduce_agent_single ra1(reduce_plan, stream, vshmem_ptr, "reduce_agent: single tile reduce");

    ra1.launch(d_block_reductions, output_it, reduce_grid_size, reduction_op);
    _CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());
  }

  return status;
} // func doit_step

// this is an init-less reduce, needed for min/max-element functionality
// this will avoid copying the first value from device->host
template <typename Derived, typename InputIt, typename Size, typename BinaryOp, typename T>
THRUST_RUNTIME_FUNCTION T
extrema(execution_policy<Derived>& policy, InputIt first, Size num_items, BinaryOp binary_op, T*)
{
  size_t temp_storage_bytes = 0;
  cudaStream_t stream       = cuda_cub::stream(policy);

  cudaError_t status;
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    doit_step<T>,
    num_items,
    (nullptr, temp_storage_bytes, first, num_items_fixed, binary_op, static_cast<T*>(nullptr), stream));
  cuda_cub::throw_on_error(status, "extrema failed on 1st step");

  size_t allocation_sizes[2] = {sizeof(T*), temp_storage_bytes};
  void* allocations[2]       = {nullptr, nullptr};

  size_t storage_size = 0;
  status              = core::detail::alias_storage(nullptr, storage_size, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "extrema failed on 1st alias storage");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  status = core::detail::alias_storage(ptr, storage_size, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "extrema failed on 2nd alias storage");

  T* d_result = thrust::detail::aligned_reinterpret_cast<T*>(allocations[0]);

  THRUST_INDEX_TYPE_DISPATCH(
    status,
    doit_step<T>,
    num_items,
    (allocations[1], temp_storage_bytes, first, num_items_fixed, binary_op, d_result, stream));
  cuda_cub::throw_on_error(status, "extrema failed on 2nd step");

  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "extrema failed to synchronize");

  T result = cuda_cub::get_value(policy, d_result);

  return result;
}

template <template <class, class, class> class ArgFunctor, class Derived, class ItemsIt, class BinaryPred>
ItemsIt THRUST_RUNTIME_FUNCTION
element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  if (first == last)
  {
    return last;
  }

  using InputType = thrust::detail::it_value_t<ItemsIt>;
  using IndexType = thrust::detail::it_difference_t<ItemsIt>;

  IndexType num_items = static_cast<IndexType>(::cuda::std::distance(first, last));

  using iterator_tuple = tuple<ItemsIt, counting_iterator<IndexType>>;
  using zip_iterator   = zip_iterator<iterator_tuple>;

  iterator_tuple iter_tuple = thrust::make_tuple(first, counting_iterator<IndexType>(0));

  using arg_min_t = ArgFunctor<InputType, IndexType, BinaryPred>;
  using T         = tuple<InputType, IndexType>;

  zip_iterator begin = make_zip_iterator(iter_tuple);

  T result = extrema(policy, begin, num_items, arg_min_t(binary_pred), (T*) (nullptr));
  return first + thrust::get<1>(result);
}
} // namespace __extrema

/// min element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt _CCCL_HOST_DEVICE
min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  THRUST_CDP_DISPATCH((last = __extrema::element<__extrema::arg_min_f>(policy, first, last, binary_pred);),
                      (last = thrust::min_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
  return last;
}

template <class Derived, class ItemsIt>
ItemsIt _CCCL_HOST_DEVICE min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using value_type = thrust::detail::it_value_t<ItemsIt>;
  return cuda_cub::min_element(policy, first, last, ::cuda::std::less<value_type>());
}

/// max element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt _CCCL_HOST_DEVICE
max_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  THRUST_CDP_DISPATCH((last = __extrema::element<__extrema::arg_max_f>(policy, first, last, binary_pred);),
                      (last = thrust::max_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
  return last;
}

template <class Derived, class ItemsIt>
ItemsIt _CCCL_HOST_DEVICE max_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using value_type = thrust::detail::it_value_t<ItemsIt>;
  return cuda_cub::max_element(policy, first, last, ::cuda::std::less<value_type>());
}

/// minmax element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred>
::cuda::std::pair<ItemsIt, ItemsIt> _CCCL_HOST_DEVICE
minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  auto ret = ::cuda::std::make_pair(last, last);
  if (first == last)
  {
    return ret;
  }

  THRUST_CDP_DISPATCH(
    (using InputType = thrust::detail::it_value_t<ItemsIt>; using IndexType = thrust::detail::it_difference_t<ItemsIt>;

     const auto num_items = static_cast<IndexType>(::cuda::std::distance(first, last));

     using iterator_tuple = tuple<ItemsIt, counting_iterator<IndexType>>;
     using zip_iterator   = zip_iterator<iterator_tuple>;

     iterator_tuple iter_tuple = thrust::make_tuple(first, counting_iterator<IndexType>(0));

     using arg_minmax_t   = __extrema::arg_minmax_f<InputType, IndexType, BinaryPred>;
     using two_pairs_type = typename arg_minmax_t::two_pairs_type;
     using duplicate_t    = typename arg_minmax_t::duplicate_tuple;
     using transform_t    = transform_iterator<duplicate_t, zip_iterator, two_pairs_type, two_pairs_type>;

     zip_iterator begin    = make_zip_iterator(iter_tuple);
     two_pairs_type result = __extrema::extrema(
       policy, transform_t(begin, duplicate_t()), num_items, arg_minmax_t(binary_pred), (two_pairs_type*) (nullptr));
     ret = ::cuda::std::make_pair(first + get<1>(get<0>(result)), first + get<1>(get<1>(result)));),
    // CDP Sequential impl:
    (ret = thrust::minmax_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);));
  return ret;
}

template <class Derived, class ItemsIt>
::cuda::std::pair<ItemsIt, ItemsIt> _CCCL_HOST_DEVICE
minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using value_type = thrust::detail::it_value_t<ItemsIt>;
  return cuda_cub::minmax_element(policy, first, last, ::cuda::std::less<value_type>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
