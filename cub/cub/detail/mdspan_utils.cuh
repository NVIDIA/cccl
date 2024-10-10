/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda/std/__mdspan/extents.h> // std::extents
#include <cuda/std/__utility/integer_sequence.h> // ::cuda::std::index_sequence
#include <cuda/std/array> // std::array
#include <cuda/std/cstddef> // size_t

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * Utilities
 **********************************************************************************************************************/

template <::cuda::std::size_t Rank, typename IndexType, ::cuda::std::size_t... Extents, ::cuda::std::size_t... Indices>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::size_t
sub_size(const ::cuda::std::extents<IndexType, Extents...>& ext, ::cuda::std::index_sequence<Indices...> = {})
{
  if constexpr (Rank >= ext.rank())
  {
    return IndexType{1};
  }
  else if constexpr (sizeof...(Indices) == 0)
  {
    return sub_size<Rank>(ext, ::cuda::std::make_index_sequence<sizeof...(Extents) - Rank>{});
  }
  else
  {
    return (ext.extent(Rank + Indices) * ...);
  }
}

template <typename IndexType, ::cuda::std::size_t... Extents>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::size_t
size(const ::cuda::std::extents<IndexType, Extents...>& ext)
{
  return sub_size<0>(ext);
}

template <typename IndexType, ::cuda::std::size_t... E, ::cuda::std::size_t... Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
sub_sizes_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  return ::cuda::std::array{fast_div_mod{sub_size<Ranks + 1>(ext)}...};
}

template <typename IndexType, ::cuda::std::size_t... E, ::cuda::std::size_t... Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
extends_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  return ::cuda::std::array{fast_div_mod{ext.extent(Ranks)}...};
}

} // namespace detail

CUB_NAMESPACE_END
