/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/advance.h>
#include <thrust/detail/function.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/pragma_omp.h>
#include <thrust/system/omp/detail/scan.h>

#include <cuda/std/__functional/invoke.h>

#include <omp.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  using namespace thrust::detail;

  using ValueType = thrust::detail::it_value_t<InputIterator>;
  using Size      = thrust::detail::it_difference_t<InputIterator>;

  const Size n = ::cuda::std::distance(first, last);

  if (n == 0)
  {
    return result;
  }

  thrust::detail::wrapped_function<BinaryFunction, ValueType> wrapped_binary_op{binary_op};

  int num_threads = 1;
  THRUST_PRAGMA_OMP(parallel)
  {
    THRUST_PRAGMA_OMP(single)
    {
      num_threads = omp_get_num_threads();
    }
  }

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1 & 2: Scan each block and compute block sum
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid         = omp_get_thread_num();
    Size block_size = (n + num_threads - 1) / num_threads;
    Size start      = tid * block_size;
    Size end        = (start + block_size < n) ? start + block_size : n;

    if (start < n)
    {
      ValueType sum = first[start];
      result[start] = sum;

      for (Size i = start + 1; i < end; ++i)
      {
        sum       = wrapped_binary_op(sum, first[i]);
        result[i] = sum;
      }

      block_sums[tid] = sum;
    }
  }

  // Step 3: Sequential scan of block sums
  for (int i = 1; i < num_threads; ++i)
  {
    block_sums[i] = wrapped_binary_op(block_sums[i - 1], block_sums[i]);
  }

  // Step 4: Add scanned block sums to each block (except first)
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid = omp_get_thread_num();

    if (tid > 0)
    {
      Size block_size = (n + num_threads - 1) / num_threads;
      Size start      = tid * block_size;
      Size end        = (start + block_size < n) ? start + block_size : n;

      ValueType offset = block_sums[tid - 1];

      for (Size i = start; i < end; ++i)
      {
        result[i] = wrapped_binary_op(offset, result[i]);
      }
    }
  }

  ::cuda::std::advance(result, n);
  return result;
}

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  using namespace thrust::detail;

  using ValueType =
    typename ::cuda::std::__accumulator_t<BinaryFunction, thrust::detail::it_value_t<InputIterator>, InitialValueType>;
  using Size = thrust::detail::it_difference_t<InputIterator>;

  const Size n = ::cuda::std::distance(first, last);

  if (n == 0)
  {
    return result;
  }

  thrust::detail::wrapped_function<BinaryFunction, ValueType> wrapped_binary_op{binary_op};

  int num_threads = 1;
  THRUST_PRAGMA_OMP(parallel)
  {
    THRUST_PRAGMA_OMP(single)
    {
      num_threads = omp_get_num_threads();
    }
  }

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1 & 2: Scan each block with init applied to first element
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid         = omp_get_thread_num();
    Size block_size = (n + num_threads - 1) / num_threads;
    Size start      = tid * block_size;
    Size end        = (start + block_size < n) ? start + block_size : n;

    if (start < n)
    {
      ValueType sum;
      if (tid == 0)
      {
        sum           = wrapped_binary_op(init, first[start]);
        result[start] = sum;
      }
      else
      {
        sum           = first[start];
        result[start] = sum;
      }

      for (Size i = start + 1; i < end; ++i)
      {
        sum       = wrapped_binary_op(sum, first[i]);
        result[i] = sum;
      }

      block_sums[tid] = sum;
    }
  }

  // Step 3: Sequential scan of block sums (with init for first block)
  if (num_threads > 0)
  {
    for (int i = 1; i < num_threads; ++i)
    {
      block_sums[i] = wrapped_binary_op(block_sums[i - 1], block_sums[i]);
    }
  }

  // Step 4: Add scanned block sums to each block (except first)
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid = omp_get_thread_num();

    if (tid > 0)
    {
      Size block_size = (n + num_threads - 1) / num_threads;
      Size start      = tid * block_size;
      Size end        = (start + block_size < n) ? start + block_size : n;

      ValueType offset = block_sums[tid - 1];

      for (Size i = start; i < end; ++i)
      {
        result[i] = wrapped_binary_op(offset, result[i]);
      }
    }
  }

  ::cuda::std::advance(result, n);
  return result;
}

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator exclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  InitialValueType init,
  BinaryFunction binary_op)
{
  using namespace thrust::detail;

  using ValueType = InitialValueType;
  using Size      = thrust::detail::it_difference_t<InputIterator>;

  const Size n = ::cuda::std::distance(first, last);

  if (n == 0)
  {
    return result;
  }

  thrust::detail::wrapped_function<BinaryFunction, ValueType> wrapped_binary_op{binary_op};

  int num_threads = 1;
  THRUST_PRAGMA_OMP(parallel)
  {
    THRUST_PRAGMA_OMP(single)
    {
      num_threads = omp_get_num_threads();
    }
  }

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1 & 2: Exclusive scan each block and compute block sum
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid         = omp_get_thread_num();
    Size block_size = (n + num_threads - 1) / num_threads;
    Size start      = tid * block_size;
    Size end        = (start + block_size < n) ? start + block_size : n;

    if (start < n)
    {
      ValueType sum = init;

      for (Size i = start; i < end; ++i)
      {
        ValueType temp = first[i];
        result[i]      = sum;
        sum            = wrapped_binary_op(sum, temp);
      }

      block_sums[tid] = sum;
    }
  }

  // Step 3: Sequential scan of block sums
  for (int i = 1; i < num_threads; ++i)
  {
    block_sums[i] = wrapped_binary_op(block_sums[i - 1], block_sums[i]);
  }

  // Step 4: Add scanned block sums to each block (except first)
  THRUST_PRAGMA_OMP(parallel)
  {
    int tid = omp_get_thread_num();

    if (tid > 0)
    {
      Size block_size = (n + num_threads - 1) / num_threads;
      Size start      = tid * block_size;
      Size end        = (start + block_size < n) ? start + block_size : n;

      ValueType offset = block_sums[tid - 1];

      for (Size i = start; i < end; ++i)
      {
        result[i] = wrapped_binary_op(offset, result[i]);
      }
    }
  }

  ::cuda::std::advance(result, n);
  return result;
}

} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END
