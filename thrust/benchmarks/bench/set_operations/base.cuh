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

#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <typename T, typename OpT>
static void basic(nvbench::state &state, nvbench::type_list<T>, OpT op)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto size_ratio     = static_cast<std::size_t>(state.get_int64("SizeRatio"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const auto elements_in_A =
    static_cast<std::size_t>(static_cast<double>(size_ratio * elements) / 100.0f);

  thrust::device_vector<T> input = generate(elements, entropy);
  thrust::device_vector<T> output(elements);

  thrust::sort(input.begin(), input.begin() + elements_in_A);
  thrust::sort(input.begin() + elements_in_A, input.end());

  caching_allocator_t alloc;
  const std::size_t elements_in_AB = thrust::distance(
      output.begin(),
      op(policy(alloc), input.cbegin(), input.cbegin() + elements_in_A,
         input.cbegin() + elements_in_A, input.cend(), output.begin()));

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements_in_AB);

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               op(policy(alloc, launch), input.cbegin(),
                  input.cbegin() + elements_in_A,
                  input.cbegin() + elements_in_A, input.cend(), output.begin());
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;
