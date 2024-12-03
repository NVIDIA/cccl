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

#if __cccl_lib_mdspan

#  include <cub/detail/fast_modulo_division.cuh> // fast_div_mod
#  include <cub/detail/mdspan_utils.cuh> // is_sub_size_static
#  include <cub/detail/type_traits.cuh> // implicit_prom_t

#  include <cuda/std/cstddef> // size_t
#  include <cuda/std/mdspan> // dynamic_extent
#  include <cuda/std/type_traits> // enable_if
#  include <cuda/std/utility> // index_sequence

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace for_each_in_extents
{

_CCCL_TEMPLATE(int Rank, typename ExtendType, typename FastDivModType)
_CCCL_REQUIRES((ExtendType::static_extent(Rank) != ::cuda::std::dynamic_extent))
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extent_size(ExtendType ext, FastDivModType extent_size)
{
  using extent_index_type   = typename ExtendType::index_type;
  using index_type          = implicit_prom_t<extent_index_type>;
  using unsigned_index_type = ::cuda::std::make_unsigned_t<index_type>;
  return static_cast<unsigned_index_type>(ext.static_extent(Rank));
}

_CCCL_TEMPLATE(int Rank, typename ExtendType, typename FastDivModType)
_CCCL_REQUIRES((ExtendType::static_extent(Rank) == ::cuda::std::dynamic_extent))
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extent_size(ExtendType ext, FastDivModType extent_size)
{
  return extent_size;
}

_CCCL_TEMPLATE(int Rank, typename ExtendType, typename FastDivModType)
_CCCL_REQUIRES((cub::detail::is_sub_size_static<Rank + 1, ExtendType>()))
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extents_sub_size(ExtendType ext, FastDivModType extents_sub_size)
{
  return sub_size<Rank + 1>(ext);
}

_CCCL_TEMPLATE(int Rank, typename ExtendType, typename FastDivModType)
_CCCL_REQUIRES((!cub::detail::is_sub_size_static<Rank + 1, ExtendType>()))
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extents_sub_size(ExtendType ext, FastDivModType extents_sub_size)
{
  return extents_sub_size;
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
_CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<(ExtendType::rank() > 0)> computation(
  IndexType id,
  IndexType stride,
  Func func,
  ExtendType ext,
  FastDivModArrayType sub_sizes_div_array,
  FastDivModArrayType extents_mod_array)
{
  using extent_index_type = typename ExtendType::index_type;
  for (auto i = id; i < cub::detail::size(ext); i += stride)
  {
    func(i, coordinate_at<Ranks>(i, ext, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }
}

template <typename IndexType, typename Func, typename ExtendType, typename FastDivModArrayType>
_CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::enable_if_t<ExtendType::rank() == 0>
computation(IndexType id, IndexType, Func func, ExtendType, FastDivModArrayType, FastDivModArrayType)
{
  using extent_index_type = typename ExtendType::index_type;
  if (id == 0)
  {
    func(0);
  }
}

/***********************************************************************************************************************
 * Kernel entry points
 **********************************************************************************************************************/

// GCC6/7/8/9 raises unused parameter warning
#  if _CCCL_COMPILER(GCC, <, 10)
#    define _CUB_UNUSED_ATTRIBUTE __attribute__((unused))
#  else
#    define _CUB_UNUSED_ATTRIBUTE
#  endif

template <typename ChainedPolicyT,
          typename Func,
          typename ExtendType,
          typename FastDivModArrayType,
          ::cuda::std::size_t... Ranks>
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads) //
  CUB_DETAIL_KERNEL_ATTRIBUTES void static_kernel(
    Func func _CUB_UNUSED_ATTRIBUTE,
    _CCCL_GRID_CONSTANT const ExtendType ext _CUB_UNUSED_ATTRIBUTE,
    _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array _CUB_UNUSED_ATTRIBUTE,
    _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array _CUB_UNUSED_ATTRIBUTE)
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

#endif // __cccl_lib_mdspan
