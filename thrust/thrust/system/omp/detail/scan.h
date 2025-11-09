// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// OMP parallel scan implementation
#include <thrust/detail/function.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/omp/detail/pragma_omp.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__numeric/inclusive_scan.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>

#include <omp.h>

THRUST_NAMESPACE_BEGIN
namespace system::omp::detail
{
struct __no_init_tag
{};

// Threshold below which serial scan is faster than parallel
// Benchmarking shows parallel overhead dominates for small arrays
inline constexpr size_t parallel_scan_threshold = 1024;

template <bool IsInclusive,
          typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename InitialValueType,
          typename BinaryFunction>
OutputIterator scan_impl(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  [[maybe_unused]] InitialValueType init,
  BinaryFunction binary_op)
{
  using namespace thrust::detail;

  static constexpr bool has_init = !::cuda::std::is_same_v<InitialValueType, __no_init_tag>;
  // Use logic from https://wg21.link/P0571
  using accum_t = ::cuda::std::conditional_t<has_init, InitialValueType, it_value_t<InputIterator>>;
  using Size    = it_difference_t<InputIterator>;

  const Size n = ::cuda::std::distance(first, last);

  if (n == 0)
  {
    return result;
  }

  auto wrapped_binary_op = wrapped_function<BinaryFunction, accum_t>{binary_op};

  const int num_threads = omp_get_max_threads();

  // Use serial scan for small arrays where parallel overhead dominates
  if (static_cast<size_t>(n) < parallel_scan_threshold || num_threads <= 1)
  {
    if constexpr (IsInclusive)
    {
      if constexpr (has_init)
      {
        return ::cuda::std::inclusive_scan(first, last, result, wrapped_binary_op, init);
      }
      else
      {
        return ::cuda::std::inclusive_scan(first, last, result, wrapped_binary_op);
      }
    }
    else
    {
      return ::cuda::std::exclusive_scan(first, last, result, init, wrapped_binary_op);
    }
  }

  temporary_array<accum_t, DerivedPolicy> block_sums(exec, num_threads);

  // Step 1: Reduce each block (N reads)
  THRUST_PRAGMA_OMP(parallel num_threads(num_threads))
  {
    const int tid         = omp_get_thread_num();
    const Size block_size = ::cuda::ceil_div(n, num_threads);
    const Size start      = tid * block_size;
    const Size end        = ::cuda::std::min(start + block_size, n);

    if (start < n)
    {
      if constexpr (has_init)
      {
        if (tid == 0)
        {
          block_sums[tid] = ::cuda::std::reduce(first + start, first + end, init, wrapped_binary_op);
        }
        else
        {
          block_sums[tid] = ::cuda::std::reduce(first + start, first + end, accum_t{}, wrapped_binary_op);
        }
      }
      else
      {
        // No init value: reduce uses first element as init
        if (end > start)
        {
          block_sums[tid] = ::cuda::std::reduce(first + start + 1, first + end, *(first + start), wrapped_binary_op);
        }
      }
    }
  }

  // Step 2: Scan block sums
  if constexpr (has_init)
  {
    ::cuda::std::exclusive_scan(block_sums.begin(), block_sums.end(), block_sums.begin(), init, wrapped_binary_op);
  }
  else
  {
    // For no init inclusive scan, use inclusive_scan on block_sums, then shift
    if (num_threads > 1)
    {
      temporary_array<accum_t, DerivedPolicy> block_sums_copy(exec, num_threads);
      for (int i = 0; i < num_threads; ++i)
      {
        block_sums_copy[i] = block_sums[i];
      }
      ::cuda::std::inclusive_scan(
        block_sums_copy.begin(), block_sums_copy.begin() + num_threads - 1, block_sums.begin() + 1, wrapped_binary_op);
      // block_sums[0] stays as-is (no prefix for first block)
    }
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
      if constexpr (IsInclusive)
      {
        if constexpr (has_init)
        {
          const accum_t prefix = block_sums[tid];
          ::cuda::std::inclusive_scan(first + start, first + end, result + start, wrapped_binary_op, prefix);
        }
        else
        {
          // For no init: thread 0 has no prefix, others use block_sums
          if (tid == 0)
          {
            ::cuda::std::inclusive_scan(first + start, first + end, result + start, wrapped_binary_op);
          }
          else
          {
            const accum_t prefix = block_sums[tid];
            ::cuda::std::inclusive_scan(first + start, first + end, result + start, wrapped_binary_op, prefix);
          }
        }
      }
      else
      {
        const accum_t prefix = block_sums[tid];
        ::cuda::std::exclusive_scan(first + start, first + end, result + start, prefix, wrapped_binary_op);
      }
    }
  }

  return result + n;
}

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  return inclusive_scan(exec, first, last, result, __no_init_tag{}, binary_op);
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
  return scan_impl<true>(exec, first, last, result, init, binary_op);
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
  return scan_impl<false>(exec, first, last, result, init, binary_op);
}
} // namespace system::omp::detail
THRUST_NAMESPACE_END
