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

using picoseconds = std::chrono::duration<long long, std::pico>;

struct adaptive_chunk_size 
{
  template <typename Rep1, typename Period1, typename Rep2, typename Period2>
  explicit constexpr adaptive_chunk_size(
    std::chrono::duration<Rep1, Period1> const time_per_iteration,
    std::chrono::duration<Rep2, Period2> const overhead_time = std::chrono::microseconds(1)
  ) : time_per_iteration_(std::chrono::duration_cast<picoseconds>(time_per_iteration)),
      overhead_time_(std::chrono::duration_cast<picoseconds>(overhead_time)) {

      }
    
  // calculate no of cores
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    ::hpx::execution::experimental::processing_units_count_t,
    adaptive_chunk_size& this_, Executor&& exec,
    ::hpx::chrono::steady_duration const&, std::size_t count
  ) noexcept {
    std::size_t const cores_baseline = 
      ::hpx::execution::experimental::processing_units_count(
        exec, this_.time_per_iteration_, count
      );
    auto const overall_time = static_cast<double>(
      (count + 1) * this_.time_per_iteration_.count()
    );
    constexpr double efficiency_factor = 0.052;
    if(this_.overhead_time_.count()==0) {
      this_.overhead_time_ = std::chrono::duration_cast<picoseconds>(std::chrono::microseconds(1));
    }
    auto const optimal_num_cores = 
      static_cast<std::size_t>(efficiency_factor * overall_time /
      static_cast<double>(this_.overhead_time_.count()));
    std::size_t num_cores = (std::min) (cores_baseline, optimal_num_cores);
    num_cores = (std::max) (num_cores, static_cast<std::size_t>(1));
    return num_cores;
  }

  //calculate chunk size
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    ::hpx::execution::experimental::get_chunk_size_t, adaptive_chunk_size&,
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

  picoseconds time_per_iteration_;
  picoseconds overhead_time_;
};

template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename FwdIter3>
double run_merge_benchmark_hpx(int const test_count, ExPolicy policy, FwdIter1 first1, FwdIter2 last1, FwdIter1 first2, FwdIter2 last2, FwdIter3 dest) {
  // warmup
  // ::hpx::merge(policy, first1, last1, first2, last2, dest);

  // actual measurement
  std::uint64_t time = ::hpx::chrono::high_resolution_clock::now();
  
  for(int i=0; i < test_count; ++i) {
     ::hpx::merge(policy, first1, last1, first2, last2, dest);
  }

  time = ::hpx::chrono::high_resolution_clock::now() - time;

  return (static_cast<double>(time) * 1e-9) / test_count;
}

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
      //const double seq_time = run_merge_benchmark_hpx(1, ::hpx::execution::seq, first1, last1, first2, last2, result);
      double overhead_time = 0.0;
      double const time_per_iteration = 0.000000001;
      //double const time_per_iteration = seq_time / 
      //  static_cast<double>((std::max)(std::distance(first1, last1), std::distance(first2, last2)));
      
      //::hpx::execution::experimental::num_cores nc(1);
      //::hpx::execution::experimental::max_num_chunks mnc(1);
      std::size_t const all_cores = ::hpx::get_num_worker_threads();
      
      //auto temp = run_merge_benchmark_hpx(1, ::hpx::execution::par.with(nc, mnc), first1, last1, first2, last2, result);
      const double temp = 0.000000105;
      overhead_time = (temp) / static_cast<double>(all_cores);
      
      picoseconds time_per_iteration_ps(
        static_cast<int64_t>(time_per_iteration * 1e12)
      );
      picoseconds overhead_time_ps(
        static_cast<int64_t>(overhead_time * 1e12)
      );

      adaptive_chunk_size acs(time_per_iteration_ps, overhead_time_ps);
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
          hpx::detail::to_hpx_execution_policy(exec).with(acs),
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
struct hpx::execution::experimental::is_executor_parameters< thrust::system::hpx::detail::adaptive_chunk_size> : std::true_type {
};

// this system inherits merge_by_key
#include <thrust/system/cpp/detail/merge.h>
