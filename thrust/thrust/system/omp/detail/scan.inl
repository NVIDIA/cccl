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
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__numeric/inclusive_scan.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/cmath>

#include <omp.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{

// Threshold below which serial scan is faster than parallel
// Benchmarking shows parallel overhead dominates for small arrays
inline constexpr size_t parallel_scan_threshold = 1024;

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

  // Use serial scan for small arrays where parallel overhead dominates
  if (static_cast<size_t>(n) < parallel_scan_threshold)
  {
    if (n > 0)
    {
      ValueType sum = first[0];
      result[0]     = sum;
      for (Size i = 1; i < n; ++i)
      {
        sum       = wrapped_binary_op(sum, first[i]);
        result[i] = sum;
      }
    }
    ::cuda::std::advance(result, n);
    return result;
  }

  const int num_threads = omp_get_max_threads();

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1: Reduce each block (N reads)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      // Use cuda::std::reduce to compute block sum
      block_sums[tid] = ::cuda::std::reduce(first + start, first + end, ValueType{}, binary_op);
    }
  }

  // Step 2: Scan block sums using cuda::std::exclusive_scan
  if (num_threads > 1)
  {
    ::cuda::std::exclusive_scan(
      block_sums.begin(), block_sums.begin() + num_threads, block_sums.begin(), ValueType{}, binary_op);
  }

  // Step 3: Scan each block with offset (N reads/writes)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      const ValueType offset = block_sums[tid];

      // Use cuda::std::inclusive_scan with init = offset
      ::cuda::std::inclusive_scan(first + start, first + end, result + start, binary_op, offset);
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

  // Use serial scan for small arrays where parallel overhead dominates
  if (static_cast<size_t>(n) < parallel_scan_threshold)
  {
    if (n > 0)
    {
      ValueType sum = wrapped_binary_op(init, first[0]);
      result[0]     = sum;
      for (Size i = 1; i < n; ++i)
      {
        sum       = wrapped_binary_op(sum, first[i]);
        result[i] = sum;
      }
    }
    ::cuda::std::advance(result, n);
    return result;
  }

  const int num_threads = omp_get_max_threads();

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1: Reduce each block (N reads)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      if (tid == 0)
      {
        // First block: reduce with init
        block_sums[tid] = ::cuda::std::reduce(first + start, first + end, init, binary_op);
      }
      else
      {
        // Other blocks: regular reduce
        block_sums[tid] = ::cuda::std::reduce(first + start, first + end, ValueType{}, binary_op);
      }
    }
  }

  // Step 2: Scan block sums using cuda::std::exclusive_scan
  if (num_threads > 1)
  {
    ::cuda::std::exclusive_scan(
      block_sums.begin(), block_sums.begin() + num_threads, block_sums.begin(), ValueType{}, binary_op);
  }

  // Step 3: Scan each block with offset (N reads/writes)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      const ValueType offset = block_sums[tid];

      // Use cuda::std::inclusive_scan with offset
      ::cuda::std::inclusive_scan(first + start, first + end, result + start, binary_op, offset);
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

  // Use serial scan for small arrays where parallel overhead dominates
  if (static_cast<size_t>(n) < parallel_scan_threshold)
  {
    ValueType sum = init;
    for (Size i = 0; i < n; ++i)
    {
      ValueType temp = first[i];
      result[i]      = sum;
      sum            = wrapped_binary_op(sum, temp);
    }
    ::cuda::std::advance(result, n);
    return result;
  }

  const int num_threads = omp_get_max_threads();

  thrust::detail::temporary_array<ValueType, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1: Reduce each block (N reads)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      // Reduce each block with init
      block_sums[tid] = ::cuda::std::reduce(first + start, first + end, init, binary_op);
    }
  }

  // Step 2: Scan block sums using cuda::std::exclusive_scan
  if (num_threads > 1)
  {
    ::cuda::std::exclusive_scan(
      block_sums.begin(), block_sums.begin() + num_threads, block_sums.begin(), ValueType{}, binary_op);
  }

  // Step 3: Exclusive scan each block with offset (N reads/writes)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      const ValueType offset = block_sums[tid];

      // Use cuda::std::exclusive_scan with offset
      ::cuda::std::exclusive_scan(first + start, first + end, result + start, offset, binary_op);
    }
  }

  ::cuda::std::advance(result, n);
  return result;
}

} // namespace system::omp::detail
THRUST_NAMESPACE_END
