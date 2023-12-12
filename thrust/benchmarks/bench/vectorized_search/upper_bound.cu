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
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements      = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto needles_ratio = static_cast<std::size_t>(state.get_int64("NeedlesRatio"));
  const auto needles       = needles_ratio *
                       static_cast<std::size_t>(static_cast<double>(elements) / 100.0);

  thrust::device_vector<T> data = generate(elements + needles);
  thrust::device_vector<T> result(needles);
  thrust::sort(data.begin(), data.begin() + elements);

  state.add_element_count(needles);

  caching_allocator_t alloc;
  thrust::upper_bound(policy(alloc), data.begin(), data.begin() + elements,
                      data.begin() + elements, data.end(), result.begin());

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               thrust::upper_bound(
                   policy(alloc, launch), data.begin(), data.begin() + elements,
                   data.begin() + elements, data.end(), result.begin());
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_axis("NeedlesRatio", {1, 25, 50});
