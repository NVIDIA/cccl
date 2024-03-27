/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/util_type.cuh> // __CUDA_FP8_TYPES_EXIST__

#include <c2h/custom_type.cuh>
#include <c2h/vector.cuh>

#include <limits>

#if defined(__CUDA_FP8_TYPES_EXIST__)
#  include <cuda_fp8.h>

namespace std
{
template <>
class numeric_limits<__nv_fp8_e4m3>
{
public:
  static __nv_fp8_e4m3 max()
  {
    return cub::Traits<__nv_fp8_e4m3>::Max();
  }

  static __nv_fp8_e4m3 lowest()
  {
    return cub::Traits<__nv_fp8_e4m3>::Lowest();
  }
};

template <>
class numeric_limits<__nv_fp8_e5m2>
{
public:
  static __nv_fp8_e5m2 max()
  {
    return cub::Traits<__nv_fp8_e5m2>::Max();
  }

  static __nv_fp8_e5m2 lowest()
  {
    return cub::Traits<__nv_fp8_e5m2>::Lowest();
  }
};
} // namespace std
#endif // defined(__CUDA_FP8_TYPES_EXIST__)

namespace c2h
{

namespace detail
{

template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  explicit value_wrapper_t(T val)
      : m_val(val)
  {}
  explicit value_wrapper_t(int val)
      : m_val(static_cast<T>(val))
  {}
  T get() const { return m_val; }
};

} // namespace detail

class seed_t : public detail::value_wrapper_t<unsigned long long int>
{
  using value_wrapper_t::value_wrapper_t;
};

class modulo_t : public detail::value_wrapper_t<std::size_t>
{
  using value_wrapper_t::value_wrapper_t;
};

namespace detail
{

template <typename T>
struct le_comparator_op
{
  T maximum;

  template <typename ValueT>
  __host__ __device__ __forceinline__ bool operator()(const ValueT &val)
  {
    return (val <= maximum);
  }
};

void gen(seed_t seed,
         char *data,
         c2h::custom_type_state_t min,
         c2h::custom_type_state_t max,
         std::size_t elements,
         std::size_t element_size);

template <typename OffsetT, typename KeyT>
void init_key_segments(const c2h::device_vector<OffsetT> &segment_offsets,
                       KeyT *d_out,
                       std::size_t element_size);

} // namespace detail

template <template <typename> class... Ps>
void gen(seed_t seed,
         c2h::device_vector<c2h::custom_type_t<Ps...>> &data,
         c2h::custom_type_t<Ps...> min = std::numeric_limits<c2h::custom_type_t<Ps...>>::lowest(),
         c2h::custom_type_t<Ps...> max = std::numeric_limits<c2h::custom_type_t<Ps...>>::max())
{
  detail::gen(seed,
              reinterpret_cast<char *>(thrust::raw_pointer_cast(data.data())),
              min,
              max,
              data.size(),
              sizeof(c2h::custom_type_t<Ps...>));
}

template <typename T>
void gen(seed_t seed,
         c2h::device_vector<T> &data,
         T min = std::numeric_limits<T>::lowest(),
         T max = std::numeric_limits<T>::max());

template <typename T>
void gen(modulo_t mod, c2h::device_vector<T> &data);

/**
 * @brief Generates an array of offsets with uniformly distributed segment sizes in the range
 * between [min_segment_size, max_segment_size]. The last offset in the array corresponds to
 * `total_element`. At most `total_element+2` offsets (or `total_elements+1` segments) and, because
 * the very last offset must corresponds to `total_element`, the last segment may comprise more than
 * `max_segment_size` items.
 */
template <typename T>
c2h::device_vector<T>
gen_uniform_offsets(seed_t seed, T total_elements, T min_segment_size, T max_segment_size);

/**
 * @brief Generates key-segment ranges from an offsets-array like the one given by
 * `gen_uniform_offset`.
 */
template <typename OffsetT, typename KeyT>
void init_key_segments(const c2h::device_vector<OffsetT> &segment_offsets,
                       c2h::device_vector<KeyT> &keys_out)
{
  detail::init_key_segments(segment_offsets,
                            thrust::raw_pointer_cast(keys_out.data()),
                            sizeof(KeyT));
}

template <typename OffsetT, template <typename> class... Ps>
void init_key_segments(const c2h::device_vector<OffsetT> &segment_offsets,
                       c2h::device_vector<custom_type_t<Ps...>> &keys_out)
{
  detail::init_key_segments(segment_offsets,
                            reinterpret_cast<custom_type_state_t *>(
                              thrust::raw_pointer_cast(keys_out.data())),
                            sizeof(custom_type_t<Ps...>));
}

} // namespace c2h
