// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cuda/experimental/mgmn/sort>

#include <memory>
#include <vector>

#include <nccl_test_common.h>
#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

namespace large_sort_test
{
constexpr int bins  = 11;
constexpr int guard = 123456789;
using counts_type   = cuda::std::array<cuda::std::int64_t, bins>;

template <class T>
__global__ void initialize(T* storage, cuda::std::int64_t count, int rank)
{
  for (cuda::std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count + 2;
       index += gridDim.x * blockDim.x)
  {
    storage[index] = index == 0 || index == count + 1 ? T(guard) : T((index - 1 + rank) % bins - bins / 2);
  }
}

template <class T>
__global__ void validate(
  const T* storage,
  cuda::std::int64_t count,
  cuda::std::int64_t offset,
  counts_type prefix,
  bool descending,
  int* errors)
{
  if (blockIdx.x == 0 && threadIdx.x == 0 && (storage[0] != T(guard) || storage[count + 1] != T(guard)))
  {
    atomicExch(errors, 1);
  }
  for (cuda::std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
  {
    int bin = 0;
    while (offset + index >= prefix[bin])
    {
      ++bin;
    }
    const T expected = T(descending ? bins / 2 - bin : bin - bins / 2);
    if (storage[index + 1] != expected)
    {
      atomicExch(errors, 1);
    }
  }
}
} // namespace large_sort_test

using large_sort_types = c2h::type_list<int, cuda::std::int64_t>;
MULTI_GPU_TEST("sort, repeated large and mixed NVLink redistribution", large_sort_types)
{
#if defined(_CUDAX_SORT_HAS_NCCL_DEVICE)
  using T = typename c2h::get<0, TestType>;
  using cuda::std::int64_t;
  auto comms = this->communicators();
  if (comms.size() < 2)
  {
    SKIP("Mixed redistribution kernels require at least two GPUs");
  }
  auto owned_streams = nccl_test_util::make_streams();
  std::vector<cuda::stream_ref> streams{owned_streams.begin(), owned_streams.end()};
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wmissing-field-initializers")
    _CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")
    ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    _CCCL_DIAG_POP
    REQUIRE(ncclCommQueryProperties(comms[rank].native_handle(), &properties) == ncclSuccess);
    if (!properties.deviceApiSupport || ncclTeamLsa(comms[rank].native_handle()).nRanks != comms[rank].size())
    {
      SKIP("Large LSA regression requires a device-capable NVLink domain");
    }
    const auto current           = cuda::__ensure_current_context{streams[rank]};
    cuda::std::size_t free_bytes = 0, total_bytes = 0;
    REQUIRE(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    if (free_bytes < (cuda::std::size_t{8} << 30))
    {
      SKIP("Large redistribution regression requires 8 GiB of free memory per GPU");
    }
  }
  std::vector<std::unique_ptr<cudax::mgmn::sort_context>> contexts;
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    contexts.emplace_back(new cudax::mgmn::sort_context);
  }
  using Env = decltype(contexts[0]->bind(streams[0]));
  std::vector<Env> envs;
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    envs.push_back(contexts[rank]->bind(streams[rank]));
  }
  constexpr int64_t boundary = (int64_t{1} << 30) / sizeof(T);
  // Below -> mixed -> above -> below, on the same persistent contexts. Mixed
  // ranks select different LSA kernels but must share barrier slots and epochs.
  for (int iteration = 0; iteration < 4; ++iteration)
  {
    CAPTURE(iteration, sizeof(T));
    const bool descending = iteration % 2 != 0;
    std::vector<int64_t> counts;
    large_sort_test::counts_type histogram{};
    std::vector<cuda::device_buffer<T>> buffers;
    std::vector<cuda::device_buffer<int>> errors;
    std::vector<T*> inputs;
    for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
    {
      const bool large = iteration == 2 || (iteration == 1 && rank % 2 != 0);
      const int64_t count =
        iteration == 1 && comms.size() > 2 && rank == comms.size() - 1 ? 0 : boundary + (large ? 19 : -17);
      counts.push_back(count);
      // Independent CPU reference: exact global key multiplicities and output
      // offsets, without storing multi-gigabyte CPU input/reference arrays.
      for (int bin = 0; bin < large_sort_test::bins; ++bin)
      {
        const int first = (bin + large_sort_test::bins - int(rank % large_sort_test::bins)) % large_sort_test::bins;
        histogram[bin] += count > first ? 1 + (count - 1 - first) / large_sort_test::bins : 0;
      }
      const auto device = comms[rank].logical_device().underlying_device();
      buffers.push_back(cuda::make_device_buffer<T>(streams[rank], device, count + 2, cuda::no_init));
      errors.push_back(cuda::make_device_buffer<int>(streams[rank], device, std::vector<int>{0}));
      inputs.push_back(buffers.back().data() + 1);
      cuda::launch(
        streams[rank],
        cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(256)),
        large_sort_test::initialize<T>,
        buffers.back().data(),
        count,
        int(rank));
    }
    if (descending)
    {
      cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts, cuda::std::greater<>{});
    }
    else
    {
      cudax::mgmn::sort(cudax::distributed, comms, envs, inputs, counts);
    }
    large_sort_test::counts_type prefix{};
    int64_t sum = 0, offset = 0;
    for (int bin = 0; bin < large_sort_test::bins; ++bin)
    {
      sum += histogram[descending ? large_sort_test::bins - 1 - bin : bin];
      prefix[bin] = sum;
    }
    for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
    {
      cuda::launch(
        streams[rank],
        cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(256)),
        large_sort_test::validate<T>,
        buffers[rank].data(),
        counts[rank],
        offset,
        prefix,
        descending,
        errors[rank].data());
      int error = -1;
      REQUIRE(cudaMemcpyAsync(&error, errors[rank].data(), sizeof(error), cudaMemcpyDeviceToHost, streams[rank].get())
              == cudaSuccess);
      streams[rank].sync();
      REQUIRE(error == 0);
      offset += counts[rank];
    }
  }
#else
  SKIP("Large device redistribution requires NCCL device headers");
#endif
}
