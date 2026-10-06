// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/detail/__config>

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/devices>
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
  ::c2h::detail::assert_current_device(device.get());

  auto h_items = ::c2h::make_host_buffer<T>(stream, device, num_items, cuda::no_init);
  cuda::copy_bytes(stream, d_items.first(num_items), h_items);
  stream.sync();

  return h_items;
}
} // namespace detail

//! @brief Generates random values in @p data using its associated stream.
//!
//! @pre Querying the device of the stream associated with @p data must succeed and return the current C2H test device.
template <template <typename> class... Ps>
void gen(seed_t seed,
         cuda::device_buffer<custom_type_t<Ps...>>& data,
         custom_type_t<Ps...> min = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::lowest(),
         custom_type_t<Ps...> max = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::max())
{
  ::c2h::detail::gen_custom_type_state(
    data.stream(), seed, reinterpret_cast<char*>(data.data()), min, max, data.size(), sizeof(custom_type_t<Ps...>));
}

//! @brief Generates random values in @p data using its associated stream.
//!
//! @pre Querying the device of the stream associated with @p data must succeed and return the current C2H test device.
template <typename T>
void gen(seed_t seed,
         cuda::device_buffer<T>& data,
         T min = ::cuda::std::numeric_limits<T>::lowest(),
         T max = ::cuda::std::numeric_limits<T>::max())
{
  ::c2h::detail::gen_values_between(data.stream(), seed, data.first(data.size()), min, max);
}

//! @brief Holds generated device and host buffers and their shared logical size.
//!
//! @c size is the number of generated items shared by both buffers. The owning buffers may contain additional
//! capacity that is not part of the generated sequence.
template <typename T>
struct sized_device_host_buffers
{
  cuda::device_buffer<T> d_items;
  cuda::host_buffer<T> h_items;
  std::size_t size;
};

//! @brief Generates random data with the existing c2h device generator and returns device and host buffers.
//!
//! @pre Querying the device of @p stream must succeed and return the current C2H test device.
template <typename T>
[[nodiscard]] sized_device_host_buffers<T> gen_buffers(
  cuda::stream_ref stream,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto d_items = ::c2h::make_device_buffer<T>(stream, num_items, cuda::no_init);
  ::c2h::gen(seed, d_items, min, max);

  const auto items_count = d_items.size();
  auto h_items           = ::c2h::detail::device_buffer_to_host_buffer(d_items, items_count);

  return {::cuda::std::move(d_items), ::cuda::std::move(h_items), items_count};
}

//! @brief Generates random data with the existing c2h device generator and returns it in host pageable memory.
//!
//! @pre Querying the device of @p stream must succeed and return the current C2H test device.
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
} // namespace c2h
