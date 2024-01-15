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

template <class KeyT, class ValueT, class OpT>
static void basic(nvbench::state &state, nvbench::type_list<KeyT, ValueT>, OpT op)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto size_ratio     = static_cast<std::size_t>(state.get_int64("SizeRatio"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const auto elements_in_A =
    static_cast<std::size_t>(static_cast<double>(size_ratio * elements) / 100.0f);

  thrust::device_vector<KeyT> in_keys = generate(elements, entropy);
  thrust::device_vector<KeyT> out_keys(elements);

  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);

  thrust::sort(in_keys.begin(), in_keys.begin() + elements_in_A);
  thrust::sort(in_keys.begin() + elements_in_A, in_keys.end());

  caching_allocator_t alloc;
  auto result_ends =
      op(policy(alloc), in_keys.cbegin(), in_keys.cbegin() + elements_in_A,
         in_keys.cbegin() + elements_in_A, in_keys.cend(), in_vals.cbegin(),
         in_vals.cbegin() + elements_in_A, out_keys.begin(), out_vals.begin());

  const std::size_t elements_in_AB = thrust::distance(out_keys.begin(), result_ends.first);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_writes<KeyT>(elements_in_AB);
  state.add_global_memory_reads<ValueT>(OpT::read_all_values ? elements : elements_in_A); 
  state.add_global_memory_writes<ValueT>(elements_in_AB);

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch &launch) {
               op(policy(alloc, launch), in_keys.cbegin(),
                  in_keys.cbegin() + elements_in_A,
                  in_keys.cbegin() + elements_in_A, in_keys.cend(),
                  in_vals.cbegin(), in_vals.cbegin() + elements_in_A,
                  out_keys.begin(), out_vals.begin());
             });
}

using key_types   = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;
using value_types = nvbench::type_list<int8_t, int64_t>;
