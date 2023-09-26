/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <type_traits>

template <typename T, typename = int>
struct has_x : std::false_type
{};

template <typename T>
struct has_x<T, decltype((void)T::x, 0)> : std::true_type
{};

template <typename T, typename = int>
struct has_y : std::false_type
{};

template <typename T>
struct has_y<T, decltype((void)T::y, 0)> : std::true_type
{};

template <typename T, typename = int>
struct has_z : std::false_type
{};

template <typename T>
struct has_z<T, decltype((void)T::z, 0)> : std::true_type
{};

template <typename T, typename = int>
struct has_w : std::false_type
{};

template <typename T>
struct has_w<T, decltype((void)T::w, 0)> : std::true_type
{};

template <typename ScalarT, typename = int>
struct component_type_impl_t
{
  using type = ScalarT;
};

template <typename VectorT>
struct component_type_impl_t<VectorT, decltype((void)VectorT::x, 0)>
{
  using type = decltype(std::declval<VectorT>().x);
};

template <typename T>
using component_type_t = typename component_type_impl_t<T>::type;

template <typename VectorT>
struct scalar_to_vec_t
{
  using component_t = component_type_t<VectorT>;

  template <typename T, typename V = VectorT>
  __host__ __device__ __forceinline__
    typename std::enable_if<std::is_same<V, VectorT>::value && !has_x<V>::value, V>::type
    operator()(T scalar)
  {
    return static_cast<component_t>(scalar);
  }

  template <typename T, typename V = VectorT>
  __host__ __device__ __forceinline__
    typename std::enable_if<std::is_same<V, VectorT>::value && has_x<V>::value && !has_y<V>::value,
                            V>::type
    operator()(T scalar)
  {
    V val;
    val.x = static_cast<component_t>(scalar);
    return val;
  }

  template <typename T, typename V = VectorT>
  __host__ __device__ __forceinline__
    typename std::enable_if<std::is_same<V, VectorT>::value && has_y<V>::value && !has_z<V>::value,
                            V>::type
    operator()(T scalar)
  {
    V val;
    val.x = static_cast<component_t>(scalar);
    val.y = static_cast<component_t>(scalar);
    return val;
  }

  template <typename T, typename V = VectorT>
  __host__ __device__ __forceinline__
    typename std::enable_if<std::is_same<V, VectorT>::value && has_z<V>::value && !has_w<V>::value,
                            V>::type
    operator()(T scalar)
  {
    V val;
    val.x = static_cast<component_t>(scalar);
    val.y = static_cast<component_t>(scalar);
    val.z = static_cast<component_t>(scalar);
    return val;
  }

  template <typename T, typename V = VectorT>
  __host__ __device__ __forceinline__
    typename std::enable_if<std::is_same<V, VectorT>::value && has_w<V>::value, V>::type
    operator()(T scalar)
  {
    V val;
    val.x = static_cast<component_t>(scalar);
    val.y = static_cast<component_t>(scalar);
    val.z = static_cast<component_t>(scalar);
    val.w = static_cast<component_t>(scalar);
    return val;
  }
};

template <int LogicalWarpThreads, int ItemsPerThread, int BlockThreads, typename IteratorT>
void fill_striped(IteratorT it)
{
  using T = cub::detail::value_t<IteratorT>;

  constexpr int warps_in_block = BlockThreads / LogicalWarpThreads;
  constexpr int items_per_warp = LogicalWarpThreads * ItemsPerThread;
  scalar_to_vec_t<T> convert;

  for (int warp_id = 0; warp_id < warps_in_block; warp_id++)
  {
    const int warp_offset_val = items_per_warp * warp_id;

    for (int lane_id = 0; lane_id < LogicalWarpThreads; lane_id++)
    {
      const int lane_offset = warp_offset_val + lane_id;

      for (int item = 0; item < ItemsPerThread; item++)
      {
        *(it++) = convert(lane_offset + item * LogicalWarpThreads);
      }
    }
  }
}
