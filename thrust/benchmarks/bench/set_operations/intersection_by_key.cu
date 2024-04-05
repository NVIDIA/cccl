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

#include "by_key.cuh"

struct op_t
{
  static constexpr bool read_all_values = false;
  
  template <class PolicyT,
            class InputIterator1,
            class InputIterator2,
            class InputIterator3,
            class InputIterator4,
            class OutputIterator1,
            class OutputIterator2>
  __host__ thrust::pair<OutputIterator1, OutputIterator2>
  operator()(const PolicyT& policy,
             InputIterator1 keys_first1,
             InputIterator1 keys_last1,
             InputIterator2 keys_first2,
             InputIterator2 keys_last2,
             InputIterator3 values_first1,
             InputIterator4 /* values_first2 */,
             OutputIterator1 keys_result,
             OutputIterator2 values_result) const
  {
    return thrust::set_intersection_by_key(policy,
                                           keys_first1,
                                           keys_last1,
                                           keys_first2,
                                           keys_last2,
                                           values_first1,
                                           keys_result,
                                           values_result);
  }
};

template <class KeyT, class ValueT>
static void basic(nvbench::state &state, nvbench::type_list<KeyT, ValueT> tl)
{
  basic(state, tl, op_t{});
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"})
  .add_int64_axis("SizeRatio", {25, 50, 75});
