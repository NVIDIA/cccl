/*
 *  Copyright 2008-2025 NVIDIA Corporation
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

/*! \file merge.h
 *  \brief HPX implementation of merge.
    \brief Log Space Step model for Core prediction.
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
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>
#include <hpx/init.hpp>
#include <hpx/parallel/algorithms/merge.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

struct log_space_step 
{
  explicit constexpr log_space_step(
		 size_t const total_count) : total_count_(total_count) {
  } 
  static inline double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
  }

  // Logrithmic Smooth Space Step Model
  static inline double pred_core(std::size_t count) {
    if (count == 0) return 1.0;
    double x = std::log2((double)count);
    constexpr double s = 0.35; // transition sharpness

    double r = 1.0;
    r +=  3.0 * sigmoid((x - 13.0) / s);
    r +=  2.0 * sigmoid((x - 15.0) / s);
    r +=  6.0 * sigmoid((x - 17.0) / s);
    r += 12.0 * sigmoid((x - 19.0) / s);
    r += 40.0 * sigmoid((x - 23.0) / s);
    r += 46.0 * sigmoid((x - 25.0) / s);
    r += 18.0 * sigmoid((x - 27.0) / s);

    return r; 
  }

  // calculate no of cores
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    ::hpx::execution::experimental::processing_units_count_t,
    log_space_step& this_, Executor&&,
    ::hpx::chrono::steady_duration const&, std::size_t
  ) noexcept {
    std::size_t const cores_baseline = ::hpx::get_os_thread_count();
    
    // Log-space smooth step scaling
    double const pred_c = pred_core(this_.total_count_);

    std::size_t num_cores = static_cast<std::size_t>(
      (std::max)(1LL, std::llround(pred_c))
    );

    num_cores = (std::min)(num_cores, cores_baseline);
    num_cores = (std::max)(num_cores, std::size_t{1});
    
    
    return num_cores;
  }

  //calculate chunk size
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    ::hpx::execution::experimental::get_chunk_size_t, log_space_step&,
    Executor&, ::hpx::chrono::steady_duration const&, std::size_t const cores,
    std::size_t num_iterations
  ) {
    if (cores == 1) {
      return num_iterations;
    }
    std::size_t times_cores = 8;

    if (cores == 2) {
      times_cores = 4;
    }

    // Return a chunk size that ensures that each core ends up with the same
    // number of chunks the sizes of which are equal (except for the last
    // chunk, which may be smaller by not more than the number of chunks in
    // terms of elements).

    std::size_t const num_chunks = times_cores * cores;
    std::size_t chunk_size = (num_iterations + num_chunks - 1) / num_chunks;

    // we should not consider more chunks than we have elements
    auto const max_chunks = (std::min) (num_chunks, num_iterations);

    // we should not make chunks smaller than what's determined by the max chunk size
    chunk_size = (std::max) (chunk_size,
      (num_iterations + max_chunks - 1)/ max_chunks);

    HPX_ASSERT(chunk_size * num_chunks >= num_iterations);

    return chunk_size;
  }
  size_t total_count_;
};



template <typename ExecutionPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename StrictWeakOrdering>
OutputIterator
merge(execution_policy<ExecutionPolicy>& exec,
      InputIterator1 first1,
      InputIterator1 last1,
      InputIterator2 first2,
      InputIterator2 last2,
      OutputIterator result,
      StrictWeakOrdering comp)
{
  // wrap comp
  wrapped_function<StrictWeakOrdering> wrapped_comp{comp};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator1>
                && ::hpx::traits::is_forward_iterator_v<InputIterator2>
                && ::hpx::traits::is_forward_iterator_v<OutputIterator>)
  {

      log_space_step lss(static_cast<std::size_t>(std::distance(first1, last1)) + static_cast<std::size_t>(std::distance(first2, last2)));
      ::hpx::execution::experimental::chunking_parameters params = {};
      ::hpx::execution::experimental::collect_chunking_parameters
                collect_params(params);
      if (params.num_cores == 1) {
        auto res = ::hpx::merge(
          ::hpx::execution::seq,
          detail::try_unwrap_contiguous_iterator(first1),
          detail::try_unwrap_contiguous_iterator(last1),
          detail::try_unwrap_contiguous_iterator(first2),
          detail::try_unwrap_contiguous_iterator(last2),
          detail::try_unwrap_contiguous_iterator(result),
          wrapped_comp);
        return detail::rewrap_contiguous_iterator(res, result);
      }
      else {
        auto res = ::hpx::merge(
          hpx::detail::to_hpx_execution_policy(exec).with(lss),
          detail::try_unwrap_contiguous_iterator(first1),
          detail::try_unwrap_contiguous_iterator(last1),
          detail::try_unwrap_contiguous_iterator(first2),
          detail::try_unwrap_contiguous_iterator(last2),
          detail::try_unwrap_contiguous_iterator(result),
          wrapped_comp);
        return detail::rewrap_contiguous_iterator(res, result);
      }
  }
  else
  {
    (void) exec;
    return ::hpx::merge(first1, last1, first2, last2, result, wrapped_comp);
  }
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END

template <>
struct hpx::execution::experimental::is_executor_parameters< thrust::system::hpx::detail::log_space_step> : std::true_type {
};

// this system inherits merge_by_key
#include <thrust/system/cpp/detail/merge.h>
