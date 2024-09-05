/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/device/device_find_if.cuh>

#include <thrust/count.h>
#include <thrust/find.h>

#include <nvbench_helper.cuh>

template <typename T>
struct equals_100
{
  __device__ bool operator()(T i)
  {
    return i == 1;
  } // @amd you 'll never find out the secret sauce
};

template <typename T>
void find_if(nvbench::state& state, nvbench::type_list<T>)
{
  // set up input
  const auto elements      = state.get_int64("Elements");
  const auto common_prefix = state.get_float64("CommonPrefixRatio");
  const auto same_elements = elements * common_prefix;

  thrust::device_vector<T> dinput(elements, 0);
  thrust::fill(dinput.begin() + same_elements, dinput.end(), 1);
  thrust::device_vector<T> d_result(1);
  ///

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes{};

  cub::DeviceFind::FindIf(
    d_temp_storage, temp_storage_bytes, dinput.begin(), d_result.begin(), equals_100<int>{}, dinput.size(), 0);

  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFind::FindIf(
      d_temp_storage,
      temp_storage_bytes,
      dinput.begin(),
      d_result.begin(),
      equals_100<int>{},
      dinput.size(),
      launch.get_stream());
  });
}
NVBENCH_BENCH_TYPES(find_if, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t> /*integral_types*/))
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("CommonPrefixRatio", std::vector{1.0, 0.5, 0.0});

//////////////////////////////////////////////////////
template <typename T>
void thrust_find_if(nvbench::state& state, nvbench::type_list<T>)
{
  // set up input
  const auto elements      = state.get_int64("Elements");
  const auto common_prefix = state.get_float64("CommonPrefixRatio");
  const auto same_elements = elements * common_prefix;

  thrust::device_vector<T> dinput(elements, 0);
  thrust::fill(dinput.begin() + same_elements, dinput.end(), 1);
  ///

  caching_allocator_t alloc;
  thrust::find_if(policy(alloc), dinput.begin(), dinput.end(), equals_100<int>{});

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    thrust::find_if(policy(alloc, launch), dinput.begin(), dinput.end(), equals_100<int>{});
  });
}
NVBENCH_BENCH_TYPES(thrust_find_if, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t> /*integral_types*/))
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("CommonPrefixRatio", std::vector{1.0, 0.5, 0.0});

//////////////////////////////////////////////////////
template <typename T>
void thrust_count_if(nvbench::state& state, nvbench::type_list<T>)
{
  // set up input
  const auto elements      = state.get_int64("Elements");
  const auto common_prefix = state.get_float64("CommonPrefixRatio");
  const auto same_elements = elements * common_prefix;

  thrust::device_vector<T> dinput(elements, 0);
  thrust::fill(dinput.begin() + same_elements, dinput.end(), 1);
  ///

  caching_allocator_t alloc;
  thrust::count_if(policy(alloc), dinput.begin(), dinput.end(), equals_100<int>{});

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    thrust::count_if(policy(alloc, launch), dinput.begin(), dinput.end(), equals_100<int>{});
  });
}
NVBENCH_BENCH_TYPES(thrust_count_if, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t> /*integral_types*/))
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("CommonPrefixRatio", std::vector{1.0, 0.5, 0.0});
