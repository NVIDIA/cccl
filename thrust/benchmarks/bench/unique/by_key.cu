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
#include <thrust/unique.h>

#include "nvbench_helper.cuh"

template <class KeyT, class ValueT>
static void basic(nvbench::state &state, nvbench::type_list<KeyT, ValueT>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  const std::size_t min_segment_size = 1;
  const std::size_t max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<KeyT> in_keys =
    generate.uniform.key_segments(elements, min_segment_size, max_segment_size);
  thrust::device_vector<KeyT> out_keys(elements);
  thrust::device_vector<ValueT> in_vals(elements);

  caching_allocator_t alloc;
  const std::size_t unique_elements = thrust::distance(
      out_keys.begin(), thrust::unique_copy(policy(alloc), in_keys.cbegin(),
                                            in_keys.cend(), out_keys.begin()));

  thrust::device_vector<ValueT> out_vals(unique_elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(unique_elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(unique_elements);

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               thrust::unique_by_key_copy(
                   policy(alloc, launch), in_keys.cbegin(), in_keys.cend(),
                   in_vals.cbegin(), out_keys.begin(), out_vals.begin());
             });
}

using key_types = nvbench::type_list<int8_t,
                                     int16_t,
                                     int32_t,
                                     int64_t
#if NVBENCH_HELPER_HAS_I128
                                     ,
                                     int128_t
#endif
                                     >;
using value_types = all_types;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 8});
