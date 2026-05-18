//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/block/block_reduce.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/iterator>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/execution>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
struct DeviceSegmentedSumKernel
{
  template <class Config, class T, cuda::std::size_t SegmentSize, class GroupFn, class UnitFn>
  __device__ void operator()(
    Config config,
    const T* in,
    T* out,
    cuda::std::size_t nsegments,
    cuda::std::integral_constant<cuda::std::size_t, SegmentSize>,
    GroupFn group_fn,
    UnitFn unit_fn)
  {
    constexpr auto nitems_per_thread = SegmentSize / cuda::gpu_thread.static_count(cuda::block, config);

    const auto segment_offset = SegmentSize * cuda::block.rank(cuda::grid, config);

    T items[nitems_per_thread];
    for (cuda::std::size_t i = 0; i < nitems_per_thread; ++i)
    {
      const auto offset = cuda::gpu_thread.rank(cuda::block, config) + i * cuda::gpu_thread.count(cuda::block, config);
      items[i]          = *(in + segment_offset + offset);
    }

    group_fn(cudax::this_block{config}, cuda::std::span{items});

    using BlockReduce = cub::BlockReduce<T, static_cast<int>(cuda::gpu_thread.static_count(cuda::block, config))>;
    __shared__ typename BlockReduce::TempStorage scratch;

    const auto result = BlockReduce{scratch}.Sum(items);
    if (cuda::gpu_thread.rank(cuda::block, config) == 0)
    {
      out[cuda::block.rank(cuda::grid, config)] = unit_fn(result);
    }
  }
};

template <class T, cuda::std::size_t SegmentSize, class GroupFn, class UnitFn>
void device_segmented_sum(
  cuda::stream_ref stream,
  const T* in,
  T* out,
  cuda::std::size_t nsegments,
  cuda::std::integral_constant<cuda::std::size_t, SegmentSize> segment_size,
  GroupFn group_fn,
  UnitFn unit_fn)
{
  const auto config =
    cuda::make_config(cuda::grid_dims(dim3{static_cast<unsigned>(nsegments)}), cuda::block_dims<SegmentSize>());
  cuda::launch(stream, config, DeviceSegmentedSumKernel{}, in, out, nsegments, segment_size, group_fn, unit_fn);
}
} // namespace

C2H_TEST("Segmented algorithm", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  auto in  = cuda::make_device_buffer<int>(stream, device, 1024, 1);
  auto out = cuda::make_device_buffer<int>(stream, device, 8, cuda::no_init);

  device_segmented_sum(
    stream,
    in.data(),
    out.data(),
    8,
    cuda::std::integral_constant<cuda::std::size_t, 128>{},
    [] __device__(auto group, auto items) {
      for (auto& item : items)
      {
        item *= 2;
      }
      group.sync();
    },
    [] __device__(auto value) {
      return value / 2;
    });
  stream.sync();

  CUDAX_CHECK(cuda::std::equal(cuda::execution::gpu, out.begin(), out.end(), cuda::constant_iterator{128}));
}
