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

#if _CCCL_STD_VER >= 2017

#  include <cub/detail/fast_modulo_division.cuh> // fast_div_mod
#  include <cub/detail/mdspan_utils.cuh>

#  include <cuda/std/__utility/integer_sequence.h> // std::index_sequence
#  include <cuda/std/cstddef> // std::size_t
#  include <cuda/std/mdspan> // std::size_t

CUB_NAMESPACE_BEGIN

namespace detail::for_each_in_extents
{

template <typename IndexType>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE //
IndexType
coordinate_at(IndexType index, fast_div_mod<IndexType> sub_size, fast_div_mod<IndexType> extent)
{
  return (index / sub_size) % extent;
}

template <typename IndexType,
          typename Func,
          typename FastDivModArrayType,
          typename UIndexType = ::cuda::std::make_unsigned_t<IndexType>,
          ::cuda::std::size_t... Ranks>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(
  Func func,
  _CCCL_GRID_CONSTANT const UIndexType size,
  _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
  _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  auto id  = threadIdx.x + blockIdx.x * blockDim.x;
  auto id1 = static_cast<IndexType>(id);
  if (id1 < size)
  {
    func(id1, coordinate_at(id1, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }
}

template <int Rank, typename ExtendType, typename IndexType = typename ExtendType::index_type>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE //
IndexType
coordinate_at(
  IndexType index, ExtendType ext, fast_div_mod<IndexType> div_mod_sub_size, fast_div_mod<IndexType> div_mod_size)
{
  auto get_sub_size = [&]() {
    constexpr auto is_sub_size_static_v = cub::detail::is_sub_size_static<Rank + 1>(ext); // GCC <= 9 workaround
    if constexpr (is_sub_size_static_v)
    {
      return sub_size<Rank + 1>(ext);
    }
    else
    {
      return div_mod_sub_size;
    }
  };
  auto get_ext_size = [&]() {
    if constexpr (ExtendType::static_extent(Rank) != ::cuda::std::dynamic_extent)
    {
      return IndexType{ext.static_extent(Rank)};
    }
    else
    {
      return div_mod_size;
    }
  };
  return (index / get_sub_size()) % get_ext_size();
}

template <typename Func, typename ExtendType, typename FastDivModArrayType, ::cuda::std::size_t... Ranks>
CUB_DETAIL_KERNEL_ATTRIBUTES void for_each_in_extents_kernel(
  Func func,
  _CCCL_GRID_CONSTANT const ExtendType ext,
  _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
  _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < cub::detail::size(ext))
  {
    auto id1 = static_cast<typename ExtendType::index_type>(id);
    func(id1, coordinate_at<Ranks>(id1, ext, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }
}

} // namespace detail::for_each_in_extents

CUB_NAMESPACE_END

#endif // _CCCL_STD_VER >= 2017
