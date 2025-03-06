/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/fast_modulo_division.cuh> // fast_div_mod
#include <cub/detail/mdspan_utils.cuh> // is_sub_size_static
#include <cub/detail/type_traits.cuh> // implicit_prom_t

#include <cuda/std/cstddef> // size_t
#include <cuda/std/mdspan> // dynamic_extent
#include <cuda/std/type_traits> // enable_if
#include <cuda/std/utility> // index_sequence

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace for_each_in_extents
{

template <int Rank, typename ExtendType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extent_size(ExtendType ext, FastDivModType extent_size)
{
  if constexpr (ExtendType::static_extent(Rank) != ::cuda::std::dynamic_extent)
  {
    using extent_index_type   = typename ExtendType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = ::cuda::std::make_unsigned_t<index_type>;
    return static_cast<unsigned_index_type>(ext.static_extent(Rank));
  }
  else
  {
    return extent_size;
  }
}

template <int Rank, typename ExtendType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extents_sub_size(ExtendType ext, FastDivModType extents_sub_size)
{
  if constexpr (cub::detail::is_sub_size_static<Rank + 1, ExtendType>())
  {
    return sub_size<Rank + 1>(ext);
  }
  else
  {
    return extents_sub_size;
  }
}

template <int Rank, typename IndexType, typename ExtendType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
coordinate_at(IndexType index, ExtendType ext, FastDivModType extents_sub_size, FastDivModType extent_size)
{
  using extent_index_type = typename ExtendType::index_type;
  return static_cast<extent_index_type>(
    (index / get_extents_sub_size<Rank>(ext, extents_sub_size)) % get_extent_size<Rank>(ext, extent_size));
}

template <typename IndexType,
          typename Func,
          typename ExtendType,
          typename FastDivModArrayType, //
          ::cuda::std::size_t... Ranks>
_CCCL_DEVICE _CCCL_FORCEINLINE void computation(
  IndexType id,
  [[maybe_unused]] IndexType stride,
  Func func,
  [[maybe_unused]] ExtendType ext,
  [[maybe_unused]] FastDivModArrayType sub_sizes_div_array,
  [[maybe_unused]] FastDivModArrayType extents_mod_array)
{
  using extent_index_type = typename ExtendType::index_type;
  if constexpr (ExtendType::rank() > 0)
  {
    for (auto i = id; i < cub::detail::size(ext); i += stride)
    {
      func(i, coordinate_at<Ranks>(i, ext, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
    }
  }
  else
  {
    using extent_index_type = typename ExtendType::index_type;
    if (id == 0)
    {
      func(0);
    }
  }
}

/***********************************************************************************************************************
 * Kernel entry points
 **********************************************************************************************************************/

template <typename ChainedPolicyT,
          typename Func,
          typename ExtendType,
          typename FastDivModArrayType,
          ::cuda::std::size_t... Ranks>
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads)
  CUB_DETAIL_KERNEL_ATTRIBUTES void static_kernel(
    [[maybe_unused]] Func func,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const ExtendType ext,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  using active_policy_t        = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using extent_index_type      = typename ExtendType::index_type;
  using offset_t               = implicit_prom_t<extent_index_type>;
  constexpr auto block_threads = offset_t{active_policy_t::block_threads};
  constexpr auto stride        = offset_t{block_threads * active_policy_t::items_per_thread};
  auto stride1                 = (stride >= cub::detail::size(ext)) ? block_threads : stride;
  auto id                      = static_cast<offset_t>(threadIdx.x + blockIdx.x * block_threads);
  computation<offset_t, Func, ExtendType, FastDivModArrayType, Ranks...>(
    id, stride1, func, ext, sub_sizes_div_array, extents_mod_array);
}

template <typename ChainedPolicyT,
          typename Func,
          typename ExtendType,
          typename FastDivModArrayType,
          ::cuda::std::size_t... Ranks>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(
  Func func,
  _CCCL_GRID_CONSTANT const ExtendType ext,
  _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
  _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  using active_policy_t   = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using extent_index_type = typename ExtendType::index_type;
  using offset_t          = implicit_prom_t<extent_index_type>;
  auto block_threads      = offset_t{blockDim.x};
  auto stride             = static_cast<offset_t>(blockDim.x * active_policy_t::items_per_thread);
  auto stride1            = (stride >= cub::detail::size(ext)) ? block_threads : stride;
  auto id                 = static_cast<offset_t>(threadIdx.x + blockIdx.x * blockDim.x);
  computation<offset_t, Func, ExtendType, FastDivModArrayType, Ranks...>(
    id, stride, func, ext, sub_sizes_div_array, extents_mod_array);
}

} // namespace for_each_in_extents
} // namespace detail

CUB_NAMESPACE_END
