// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/detail/config/device_system.h>

#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/utility>

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/__algorithm/copy.h> // cuda::copy_bytes
#  include <cuda/buffer>
#  include <cuda/devices>
#  include <cuda/memory_resource>
#  include <cuda/stream>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#include <cstddef>

#include <c2h/checked_allocator.cuh>
#include <c2h/custom_type.h>
#include <c2h/vector.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  if _CCCL_HAS_NVFP16()
#    include <cuda_fp16.h>
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#    include <cuda_bf16.h>
_CCCL_DIAG_POP
#  endif // _CCCL_HAS_NVBF16

#  if _CCCL_HAS_NVFP8()
// cuda_fp8.h resets default for C4127, so we have to guard the inclusion
_CCCL_DIAG_PUSH
#    include <cuda_fp8.h>
_CCCL_DIAG_POP
#  endif // _CCCL_HAS_NVFP8()
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

namespace c2h
{
namespace detail
{
template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  using value_type = T;

  explicit value_wrapper_t(T val)
      : m_val(val)
  {}
  explicit value_wrapper_t(int val)
      : m_val(static_cast<T>(val))
  {}
  T get() const
  {
    return m_val;
  }
};
} // namespace detail

struct seed_t : detail::value_wrapper_t<unsigned long long int>
{
  using value_wrapper_t::value_wrapper_t;
};

struct modulo_t : detail::value_wrapper_t<std::size_t>
{
  using value_wrapper_t::value_wrapper_t;
};

namespace detail
{
void gen_custom_type_state(
  seed_t seed,
  char* data,
  custom_type_state_t min,
  custom_type_state_t max,
  std::size_t elements,
  std::size_t element_size);

template <typename OffsetT, typename KeyT>
void init_key_segments(::cuda::std::span<const OffsetT> segment_offsets, KeyT* d_out, std::size_t element_size);

template <typename T>
void gen_values_between(seed_t seed, ::cuda::std::span<T> data, T min, T max);

template <typename T>
void gen_values_cyclic(modulo_t mod, ::cuda::std::span<T> data);

template <typename T>
std::size_t gen_uniform_offsets(
  seed_t seed, cuda::std::span<T> segment_offsets, T total_elements, T min_segment_size, T max_segment_size);

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
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
  auto h_items =
    cuda::host_buffer<T>{stream, cuda::mr::legacy_pinned_memory_resource{device}, num_items, cuda::no_init};
  cuda::copy_bytes(stream, d_items.first(num_items), h_items);
  stream.sync();

  return h_items;
}

template <typename T>
void gen_into_device_buffer(seed_t seed, cuda::device_buffer<T>& d_items, T min, T max)
{
  gen_values_between(seed, d_items.first(d_items.size()), min, max);
}

template <template <typename> class... Ps>
void gen_into_device_buffer(
  seed_t seed, cuda::device_buffer<custom_type_t<Ps...>>& d_items, custom_type_t<Ps...> min, custom_type_t<Ps...> max)
{
  gen_custom_type_state(
    seed, reinterpret_cast<char*>(d_items.data()), min, max, d_items.size(), sizeof(custom_type_t<Ps...>));
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace detail

template <template <typename> class... Ps>
void gen(seed_t seed,
         device_vector<custom_type_t<Ps...>>& data,
         custom_type_t<Ps...> min = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::lowest(),
         custom_type_t<Ps...> max = ::cuda::std::numeric_limits<custom_type_t<Ps...>>::max())
{
  detail::gen_custom_type_state(
    seed,
    reinterpret_cast<char*>(THRUST_NS_QUALIFIER::raw_pointer_cast(data.data())),
    min,
    max,
    data.size(),
    sizeof(custom_type_t<Ps...>));
}

template <typename T>
void gen(seed_t seed,
         device_vector<T>& data,
         T min = ::cuda::std::numeric_limits<T>::lowest(),
         T max = ::cuda::std::numeric_limits<T>::max())
{
  detail::gen_values_between(seed, {THRUST_NS_QUALIFIER::raw_pointer_cast(data.data()), data.size()}, min, max);
}

template <typename T>
void gen(modulo_t mod, device_vector<T>& data)
{
  detail::gen_values_cyclic(mod, ::cuda::std::span<T>{THRUST_NS_QUALIFIER::raw_pointer_cast(data.data()), data.size()});
}

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
struct device_host_buffers
{
  cuda::device_buffer<T> d_items;
  cuda::host_buffer<T> h_items;
};

template <typename T>
struct sized_device_buffer
{
  cuda::device_buffer<T> d_items;
  std::size_t size;
};

/**
 * @brief Generates random data with the existing c2h device generator and returns it in device memory.
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
  auto d_items = c2h::make_device_buffer<T>(stream, device, num_items, cuda::no_init);
  detail::sync_before_default_stream(stream);
  detail::gen_into_device_buffer(seed, d_items, min, max);

  return d_items;
}

/**
 * @brief Generates random data with the existing c2h device generator and returns device and host buffers.
 */
template <typename T>
[[nodiscard]] device_host_buffers<T> gen_buffers(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  std::size_t num_items,
  T min = ::cuda::std::numeric_limits<T>::lowest(),
  T max = ::cuda::std::numeric_limits<T>::max())
{
  auto d_items = gen_device_buffer<T>(stream, device, seed, num_items, min, max);
  auto h_items = detail::device_buffer_to_host_buffer(stream, device, d_items, d_items.size());

  return {::cuda::std::move(d_items), ::cuda::std::move(h_items)};
}

/**
 * @brief Generates random data with the existing c2h device generator and returns it in pinned host memory.
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
  auto buffers = gen_buffers<T>(stream, device, seed, num_items, min, max);
  return ::cuda::std::move(buffers.h_items);
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

/**
 * @brief Generates an array of offsets with uniformly distributed segment sizes in the range
 * between [min_segment_size, max_segment_size]. The last offset in the array corresponds to
 * `total_element`. At most `total_element+2` offsets (or `total_elements+1` segments) and, because
 * the very last offset must corresponds to `total_element`, the last segment may comprise more than
 * `max_segment_size` items.
 */
template <typename T>
device_vector<T> gen_uniform_offsets(seed_t seed, T total_elements, T min_segment_size, T max_segment_size)
{
  device_vector<T> segment_offsets(total_elements + 2);
  const auto new_size = detail::gen_uniform_offsets(
    seed,
    {THRUST_NS_QUALIFIER::raw_pointer_cast(segment_offsets.data()), segment_offsets.size()},
    total_elements,
    min_segment_size,
    max_segment_size);
  segment_offsets.resize(new_size);
  return segment_offsets;
}

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns them in device memory.
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
  auto d_segment_offsets =
    c2h::make_device_buffer<T>(stream, device, static_cast<std::size_t>(total_elements) + 2, cuda::no_init);
  detail::sync_before_default_stream(stream);
  const auto num_offsets = detail::gen_uniform_offsets(
    seed, d_segment_offsets.first(d_segment_offsets.size()), total_elements, min_segment_size, max_segment_size);

  return {::cuda::std::move(d_segment_offsets), num_offsets};
}

/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns device and host buffers.
 */
template <typename T>
[[nodiscard]] device_host_buffers<T> gen_uniform_offsets_buffers(
  cuda::stream_ref stream,
  cuda::device_ref device,
  seed_t seed,
  T total_elements,
  T min_segment_size,
  T max_segment_size)
{
  auto d_segment_offsets =
    gen_uniform_offsets_device_buffer<T>(stream, device, seed, total_elements, min_segment_size, max_segment_size);
  auto h_segment_offsets =
    detail::device_buffer_to_host_buffer(stream, device, d_segment_offsets.d_items, d_segment_offsets.size);

  return {::cuda::std::move(d_segment_offsets.d_items), ::cuda::std::move(h_segment_offsets)};
}

/**
 * @brief Generates uniform segment offsets with the existing c2h device generator and returns them in pinned host
 * memory.
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
    gen_uniform_offsets_buffers<T>(stream, device, seed, total_elements, min_segment_size, max_segment_size);
  return ::cuda::std::move(buffers.h_items);
}
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

/**
 * @brief Generates key-segment ranges from an offsets-array like the one given by
 * `gen_uniform_offset`.
 */
template <typename OffsetT, typename KeyT>
void init_key_segments(const device_vector<OffsetT>& segment_offsets, device_vector<KeyT>& keys_out)
{
  detail::init_key_segments(
    ::cuda::std::span<const OffsetT>{
      THRUST_NS_QUALIFIER::raw_pointer_cast(segment_offsets.data()), segment_offsets.size()},
    THRUST_NS_QUALIFIER::raw_pointer_cast(keys_out.data()),
    sizeof(KeyT));
}

template <typename OffsetT, template <typename> class... Ps>
void init_key_segments(const device_vector<OffsetT>& segment_offsets, device_vector<custom_type_t<Ps...>>& keys_out)
{
  detail::init_key_segments(
    ::cuda::std::span<const OffsetT>{
      THRUST_NS_QUALIFIER::raw_pointer_cast(segment_offsets.data()), segment_offsets.size()},
    static_cast<custom_type_state_t*>(THRUST_NS_QUALIFIER::raw_pointer_cast(keys_out.data())),
    sizeof(custom_type_t<Ps...>));
}
} // namespace c2h
