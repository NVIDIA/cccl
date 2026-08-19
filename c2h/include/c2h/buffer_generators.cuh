// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/detail/__config>

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/__algorithm/copy.h> // cuda::copy_bytes
#  include <cuda/buffer>
#  include <cuda/devices>
#  include <cuda/std/limits>
#  include <cuda/std/utility>
#  include <cuda/stream>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#include <cstddef>

#include <c2h/checked_memory_resource.cuh>
#include <c2h/generator_types.h>

namespace c2h
{
#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
namespace detail
{
[[nodiscard]] inline bool is_default_stream(cuda::stream_ref stream) noexcept
{
  return stream.get() == ::cudaStream_t{};
}

inline void sync_before_default_stream(cuda::stream_ref stream)
{
  if (!is_default_stream(stream))
  {
    stream.sync();
  }
}

template <typename T>
[[nodiscard]] cuda::host_buffer<T> device_buffer_to_host_buffer(
  cuda::stream_ref stream, cuda::device_ref device, const cuda::device_buffer<T>& d_items, std::size_t num_items)
{
  // Scope `device` for default-stream operations. Non-default stream/device
  // agreement is part of the public helper contract.
  const ::c2h::detail::scoped_current_device device_scope{device.get()};

  auto h_items = ::c2h::make_host_buffer<T>(stream, device, num_items, cuda::no_init);
  cuda::copy_bytes(stream, d_items.first(num_items), h_items);
  stream.sync();

  return h_items;
}

template <typename T>
void gen_into_device_buffer(seed_t seed, cuda::device_buffer<T>& d_items, T min, T max)
{
  ::c2h::detail::gen_values_between(seed, d_items.first(d_items.size()), min, max);
}

template <template <typename> class... Ps>
void gen_into_device_buffer(
  seed_t seed, cuda::device_buffer<custom_type_t<Ps...>>& d_items, custom_type_t<Ps...> min, custom_type_t<Ps...> max)
{
  ::c2h::detail::gen_custom_type_state(
    seed, reinterpret_cast<char*>(d_items.data()), min, max, d_items.size(), sizeof(custom_type_t<Ps...>));
}
} // namespace detail

// `size` is the number of generated items. The owning buffers may contain
// additional capacity that is not part of the generated sequence.
template <typename T>
struct sized_device_buffer
{
  cuda::device_buffer<T> d_items;
  std::size_t size;
};

// `size` is the number of generated items shared by both buffers. The owning
// buffers may contain additional capacity that is not part of the generated sequence.
template <typename T>
struct sized_device_host_buffers
{
  cuda::device_buffer<T> d_items;
  cuda::host_buffer<T> h_items;
  std::size_t size;
};

/**
 * @brief Generates random data with the existing c2h device generator and returns it in device memory.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] cuda::device_buffer<T> gen_device_buffer(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  // Scope `device` for default-stream operations.
  const ::c2h::detail::scoped_current_device device_scope{device.get()};

  auto d_items = ::c2h::make_device_buffer<T>(stream, device, num_items, cuda::no_init);
  ::c2h::detail::sync_before_default_stream(stream);
  ::c2h::detail::gen_into_device_buffer(seed, d_items, min, max);

  return d_items;
}

/**
 * @brief Generates random data with the existing c2h device generator and returns device and host buffers.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] sized_device_host_buffers<T> gen_buffers(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto d_items = ::c2h::gen_device_buffer<T>(stream, device, seed, num_items, min, max);

  const auto items_count = d_items.size();

  auto h_items = ::c2h::detail::device_buffer_to_host_buffer(stream, device, d_items, items_count);

  return {::cuda::std::move(d_items), ::cuda::std::move(h_items), items_count};
}

/**
 * @brief Generates random data with the existing c2h device generator and returns it in host pageable memory.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] cuda::host_buffer<T> gen_host_buffer(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto buffers = ::c2h::gen_buffers<T>(stream, device, seed, num_items, min, max);
  return ::cuda::std::move(buffers.h_items);
}

/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns them in device memory.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] sized_device_buffer<T> gen_uniform_offsets_device_buffer(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  T total_elements,
  T min_segment_size,
  T max_segment_size)
{
  // Scope `device` for default-stream operations.
  const ::c2h::detail::scoped_current_device device_scope{device.get()};

  auto d_segment_offsets =
    ::c2h::make_device_buffer<T>(stream, device, static_cast<std::size_t>(total_elements) + 2, cuda::no_init);
  ::c2h::detail::sync_before_default_stream(stream);
  const auto num_offsets = ::c2h::detail::gen_uniform_offsets(
    seed, d_segment_offsets.first(d_segment_offsets.size()), total_elements, min_segment_size, max_segment_size);

  return {::cuda::std::move(d_segment_offsets), num_offsets};
}

/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns device and host buffers.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] sized_device_host_buffers<T> gen_uniform_offsets_buffers(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  T total_elements,
  T min_segment_size,
  T max_segment_size)
{
  auto d_segment_offsets = ::c2h::gen_uniform_offsets_device_buffer<T>(
    stream, device, seed, total_elements, min_segment_size, max_segment_size);

  const auto num_items = d_segment_offsets.size;

  auto h_segment_offsets =
    ::c2h::detail::device_buffer_to_host_buffer(stream, device, d_segment_offsets.d_items, num_items);

  return {::cuda::std::move(d_segment_offsets.d_items), ::cuda::std::move(h_segment_offsets), num_items};
}

/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns them in host pageable
 * memory.
 *
 * @pre If `stream` is non-default, it must have been created for `device`.
 */
template <typename T>
[[nodiscard]] cuda::host_buffer<T> gen_uniform_offsets_host_buffer(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  T total_elements,
  T min_segment_size,
  T max_segment_size)
{
  auto buffers =
    ::c2h::gen_uniform_offsets_buffers<T>(stream, device, seed, total_elements, min_segment_size, max_segment_size);
  return ::cuda::std::move(buffers.h_items);
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
} // namespace c2h
