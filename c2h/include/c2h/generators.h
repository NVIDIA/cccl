// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <thrust/detail/config/device_system.h>

#include <cuda/std/limits>

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
