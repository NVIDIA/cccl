// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/__cccl_config>

#if _CCCL_CUDA_COMPILATION()
#  include <cub/device/device_transform.cuh>
#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/__algorithm/copy.h>
#include <cuda/__algorithm/fill.h>
#include <cuda/__container/resizable_buffer.h>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstddef>

#include <c2h/checked_memory_resource.cuh>
#include <c2h/detail/current_device.cuh>
#include <c2h/generator_common.h>

namespace c2h
{
namespace detail
{
template <typename T>
[[nodiscard]] cuda::host_buffer<T>
device_buffer_to_host_buffer(const cuda::device_buffer<T>& d_items, std::size_t num_items)
{
  const auto stream = d_items.stream();
  const auto device = stream.device();

  auto h_items = ::c2h::make_host_buffer<T>(stream, device, d_items.first(num_items));
  stream.sync();

  return h_items;
}

template <typename T>
[[nodiscard]] cuda::device_buffer<T> make_generator_buffer(cuda::stream_ref stream, std::size_t num_items, const T&)
{
  return ::c2h::make_device_buffer<T>(stream, num_items, cuda::no_init);
}

// custom_type_t generation only overwrites key and val. Initialize policy members
// from min and padding bytes to zero before any whole-object copies.
template <template <typename> class... Ps>
[[nodiscard]] cuda::device_buffer<custom_type_t<Ps...>>
make_generator_buffer(cuda::stream_ref stream, std::size_t num_items, [[maybe_unused]] const custom_type_t<Ps...>& min)
{
  auto result = ::c2h::make_device_buffer<custom_type_t<Ps...>>(stream, num_items, cuda::no_init);
  if (num_items != 0)
  {
    ::cuda::fill_bytes(stream, result, 0);
#if _CCCL_CUDA_COMPILATION()
    _CCCL_TRY_RUNTIME_API(
      CUB_NS_QUALIFIER::DeviceTransform::Fill,
      "Failed to initialize custom-type policy members",
      result.data(),
      num_items,
      min,
      stream.get());
#else // _CCCL_CUDA_COMPILATION()
    static_assert(::cuda::std::__always_false_v<custom_type_t<Ps...>>,
                  "A CUDA compiler is required to initialize custom-type generator buffers");
#endif // _CCCL_CUDA_COMPILATION()
  }
  return result;
}
} // namespace detail

// Generates random values in data using its associated stream.
//
// Requires: Querying the device of the stream associated with data must succeed and return the current C2H test device.
template <template <typename> class... Ps>
void gen(seed_t seed,
         cuda::device_buffer<custom_type_t<Ps...>>& data,
         custom_type_t<Ps...> min = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::lowest(),
         custom_type_t<Ps...> max = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::max())
{
  ::c2h::detail::gen_custom_type_state(
    data.stream(), seed, reinterpret_cast<char*>(data.data()), min, max, data.size(), sizeof(custom_type_t<Ps...>));
}

// Generates random values in data using its associated stream.
//
// Requires: Querying the device of the stream associated with data must succeed and return the current C2H test device.
template <typename T>
void gen(seed_t seed,
         cuda::device_buffer<T>& data,
         T min = ::cuda::std::numeric_limits<T>::lowest(),
         T max = ::cuda::std::numeric_limits<T>::max())
{
  ::c2h::detail::gen_values_between(data.stream(), seed, data.first(data.size()), min, max);
}

// Holds generated device and host buffers for the same logical sequence.
//
// h_items contains the complete logical sequence. The corresponding sequence is stored in d_items, which may
// retain additional allocation capacity beyond its logical size.
template <typename T, typename DeviceBuffer = cuda::device_buffer<T>>
struct device_host_buffers
{
  DeviceBuffer d_items;
  cuda::host_buffer<T> h_items;

  // Returns the number of logical items shared by both buffers.
  [[nodiscard]] std::size_t size() const noexcept
  {
    _CCCL_ASSERT(d_items.size() == h_items.size(), "Device and host buffers must have the same logical size");
    return h_items.size();
  }
};

// Generates random data with the existing c2h device generator and returns device and host buffers.
//
// Requires: Querying the device of stream must succeed and return the current C2H test device.
template <typename T>
[[nodiscard]] device_host_buffers<T> gen_buffers(
  cuda::stream_ref stream,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto d_items = ::c2h::detail::make_generator_buffer(stream, num_items, min);
  ::c2h::gen(seed, d_items, min, max);

  const auto items_count = d_items.size();
  auto h_items           = ::c2h::detail::device_buffer_to_host_buffer(d_items, items_count);

  return {::cuda::std::move(d_items), ::cuda::std::move(h_items)};
}

// Generates random data with the existing c2h device generator and returns it in host pageable memory.
//
// Requires: Querying the device of stream must succeed and return the current C2H test device.
template <typename T>
[[nodiscard]] cuda::host_buffer<T> gen_host_buffer(
  cuda::stream_ref stream,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto buffers = ::c2h::gen_buffers<T>(stream, seed, num_items, min, max);
  return ::cuda::std::move(buffers.h_items);
}

// Generates uniform segment offsets with the existing c2h device generator and returns them in device memory.
//
// Requires: stream must not be the legacy default stream.
template <typename T>
[[nodiscard]] cuda::__resizable_buffer<T, cuda::mr::device_accessible> gen_uniform_offsets_device_buffer(
  cuda::stream_ref stream, seed_t seed, T total_elements, T min_segment_size, T max_segment_size)
{
  const auto offsets_size = ::c2h::detail::checked_uniform_offsets_size(total_elements);

  const auto device = stream.device();

  cuda::__resizable_buffer<T, cuda::mr::device_accessible> d_segment_offsets{
    ::c2h::make_device_buffer<T>(stream, device, offsets_size, cuda::no_init)};
  const auto num_offsets = ::c2h::detail::gen_uniform_offsets(
    stream, seed, d_segment_offsets.first(d_segment_offsets.size()), total_elements, min_segment_size, max_segment_size);

  d_segment_offsets.resize_unsynchronized(num_offsets, cuda::no_init);
  return d_segment_offsets;
}

// Generates uniform segment offsets with the existing c2h device generator and returns device and host buffers.
//
// Requires: stream must not be the legacy default stream.
template <typename T>
[[nodiscard]] device_host_buffers<T, cuda::__resizable_buffer<T, cuda::mr::device_accessible>>
gen_uniform_offsets_buffers(
  cuda::stream_ref stream, seed_t seed, T total_elements, T min_segment_size, T max_segment_size)
{
  auto d_segment_offsets =
    ::c2h::gen_uniform_offsets_device_buffer<T>(stream, seed, total_elements, min_segment_size, max_segment_size);

  auto h_segment_offsets = ::c2h::detail::device_buffer_to_host_buffer(d_segment_offsets, d_segment_offsets.size());

  return {::cuda::std::move(d_segment_offsets), ::cuda::std::move(h_segment_offsets)};
}
} // namespace c2h
