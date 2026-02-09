/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <chrono>
#include <hpx/init.hpp>

#include "nvbench_helper.cuh"

using picoseconds = std::chrono::duration<long long, std::pico>;

struct adaptive_chunk_size {
  explicit constexpr adaptive_chunk_size(
  size_t const data_type_size
  ): data_type_size_(data_type_size) {}
  // calculate no of cores
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    hpx::execution::experimental::processing_units_count_t,
    adaptive_chunk_size& this_, Executor&& exec,
    hpx::chrono::steady_duration const&, std::size_t count
  ) noexcept {
    std::size_t const all_cores = hpx::get_num_worker_threads();
    std::size_t num_cores = 1;
    std::size_t L1_cache = 786432;
    std::size_t L2_cache = 25952256;
    //auto pu_mask = hpx::execution::experimental::get_processing_units_mask(exec);
    //auto cache_bytes = hpx::threads::get_topology().get_cache_size(pu_mask, 1);
   
    if(count <= 4096) {
      num_cores = 1;
    }
    else if (this_.data_type_size_ * count <= L1_cache/8) {
      num_cores = 2;
    }
    else if(this_.data_type_size_ * count <= L1_cache/4) {
      num_cores = 12;
    }
    else if(this_.data_type_size_ * count < (L1_cache + L2_cache)/2) {
      num_cores = 20;
    }
    else {
      num_cores = all_cores;
    } 


    return num_cores;
  }

  //calculate chunk size
  template <typename Executor>
  friend std::size_t tag_override_invoke(
    hpx::execution::experimental::get_chunk_size_t, adaptive_chunk_size&, 
    Executor&, hpx::chrono::steady_duration const&, std::size_t const cores,
    std::size_t num_iterations
  ) {
    if (cores == 1) {
      return num_iterations;
    }
    std::size_t times_cores = 8;
    
    if (cores ==2) {
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
      (num_iterations + max_chunks -1)/ max_chunks);

    HPX_ASSERT(chunk_size * num_chunks >= num_iterations);
    return chunk_size;
  }
  size_t data_type_size_;
};

template <>
struct hpx::execution::experimental::is_executor_parameters<adaptive_chunk_size> : std::true_type {
};

struct enable_fast_idle_mode
{
    template <typename Executor>
    friend void tag_override_invoke(
        hpx::execution::experimental::mark_begin_execution_t,
        enable_fast_idle_mode, Executor&& exec)
    {
        auto const pu_mask =
            hpx::execution::experimental::get_processing_units_mask(exec);
        auto const full_pu_mask =
            hpx::resource::get_partitioner().get_used_pus_mask();

        // Enable fast-idle mode only for PU's that are not used by this
        // algorithm invocation.
        
        hpx::threads::add_remove_scheduler_mode(
            hpx::threads::policies::scheduler_mode::fast_idle_mode,
            hpx::threads::policies::scheduler_mode::enable_stealing |
                hpx::threads::policies::scheduler_mode::enable_stealing_numa,
            full_pu_mask & ~pu_mask);
    }

    template <typename Executor>
    friend void tag_override_invoke(
        hpx::execution::experimental::mark_end_execution_t,
        enable_fast_idle_mode, Executor&& exec)
    {
        auto const pu_mask =
            hpx::execution::experimental::get_processing_units_mask(exec);
        auto const full_pu_mask =
            hpx::resource::get_partitioner().get_used_pus_mask();

        hpx::threads::add_remove_scheduler_mode(
            hpx::threads::policies::scheduler_mode::enable_stealing |
                hpx::threads::policies::scheduler_mode::enable_stealing_numa,
            hpx::threads::policies::scheduler_mode::fast_idle_mode,
            full_pu_mask & ~pu_mask);
    }
};

template <>
struct hpx::execution::experimental::is_executor_parameters<
    enable_fast_idle_mode> : std::true_type
{
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto size_ratio      = static_cast<std::size_t>(state.get_int64("InputSizeRatio"));
  const auto entropy         = str_to_entropy(state.get_string("Entropy"));
  const auto elements_in_lhs = static_cast<std::size_t>(static_cast<double>(size_ratio * elements) / 100.0);

  thrust::device_vector<T> out(elements);
  thrust::device_vector<T> in = generate(elements, entropy);
  thrust::sort(in.begin(), in.begin() + elements_in_lhs);
  thrust::sort(in.begin() + elements_in_lhs, in.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  adaptive_chunk_size acs(sizeof(T));
  enable_fast_idle_mode efim;
  hpx::execution::experimental::chunking_parameters params = {};
  hpx::execution::experimental::collect_chunking_parameters
                collect_params(params);

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const stackless_policy = hpx::execution::experimental::with_stacksize(policy(alloc, launch), hpx::threads::thread_stacksize::nostack);
    auto exec = hpx::execution::experimental::with_priority(stackless_policy, hpx::threads::thread_priority::initially_bound);
    thrust::merge(
      exec.with(acs, collect_params),
      in.cbegin(),
      in.cbegin() + elements_in_lhs,
      in.cbegin() + elements_in_lhs,
      in.cend(),
      out.begin());
  });
  std::cout<<"Chunk size: "<<params.chunk_size<<" No of chunks: "<<params.num_chunks<<" No of cores: "<<params.num_cores<<std::endl;
}
//using d_types = nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t, nvbench::float32_t, nvbench::float64_t>;
using d_types = nvbench::type_list<nvbench::int8_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(d_types))
  .set_name("HPX")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(8,30,2))
  .add_string_axis("Entropy", {"1.000"})
  .add_int64_axis("InputSizeRatio", {25});

