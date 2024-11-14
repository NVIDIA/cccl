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

#  include <cuda/std/__mdspan/extents.h> // dynamic_extent
#  include <cuda/std/__utility/integer_sequence.h> // index_sequence
#  include <cuda/std/cstddef> // size_t

CUB_NAMESPACE_BEGIN

namespace detail::for_each_in_extents
{

template <int Rank, typename IndexType, typename ExtendType, typename FastDivModType>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE auto
coordinate_at(IndexType index, ExtendType ext, FastDivModType div_mod_sub_size, FastDivModType div_mod_size)
{
  using extent_index_type = typename ExtendType::index_type;
  auto get_sub_size       = [&]() {
    if constexpr (cub::detail::is_sub_size_static<Rank + 1, ExtendType>())
    {
      return sub_size<Rank + 1>(ext);
    }
    else
    {
      return div_mod_sub_size;
    }
    _CCCL_UNREACHABLE();
  };
  auto get_ext_size = [&]() {
    if constexpr (ExtendType::static_extent(Rank) != ::cuda::std::dynamic_extent)
    {
      using U = ::cuda::std::make_unsigned_t<IndexType>;
      return static_cast<U>(ext.static_extent(Rank));
    }
    else
    {
      return div_mod_size;
    }
    _CCCL_UNREACHABLE();
  };
  return static_cast<extent_index_type>((index / get_sub_size()) % get_ext_size());
}

template <typename IndexType,
          typename Func,
          typename ExtendType,
          typename FastDivModArrayType, //
          ::cuda::std::size_t... Ranks>
_CCCL_DEVICE _CCCL_FORCEINLINE void computation(
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
    if constexpr (sizeof...(Ranks) == 0)
    {
      func(0);
    }
    else
    {
      func(i, coordinate_at<Ranks>(i, ext, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
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
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads) //
  CUB_DETAIL_KERNEL_ATTRIBUTES void static_kernel(
    Func func,
    _CCCL_GRID_CONSTANT const ExtendType ext,
    _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
    _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  using active_policy_t        = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using extent_index_type      = typename ExtendType::index_type;
  using OffsetT                = decltype(+extent_index_type{});
  constexpr auto block_threads = OffsetT{active_policy_t::block_threads};
  constexpr auto stride        = OffsetT{block_threads * active_policy_t::items_per_thread};
  auto stride1                 = (stride >= cub::detail::size(ext)) ? block_threads : stride;
  auto id                      = static_cast<OffsetT>(threadIdx.x + blockIdx.x * block_threads);
  computation<OffsetT, Func, ExtendType, FastDivModArrayType, Ranks...>(
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
  using OffsetT           = decltype(+extent_index_type{});
  auto block_threads      = OffsetT{blockDim.x};
  auto stride             = static_cast<OffsetT>(blockDim.x * active_policy_t::items_per_thread);
  auto stride1            = (stride >= cub::detail::size(ext)) ? block_threads : stride;
  auto id                 = static_cast<OffsetT>(threadIdx.x + blockIdx.x * blockDim.x);
  computation<OffsetT, Func, ExtendType, FastDivModArrayType, Ranks...>(
    id, stride, func, ext, sub_sizes_div_array, extents_mod_array);
}

} // namespace detail::for_each_in_extents

CUB_NAMESPACE_END

#endif // __cccl_lib_mdspan
