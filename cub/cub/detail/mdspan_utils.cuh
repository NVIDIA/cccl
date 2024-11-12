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

// check that mdspan is available
#if (_CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC)) || _CCCL_STD_VER >= 2020

#  include <cub/detail/fast_modulo_division.cuh> // fast_div_mod

#  include <cuda/std/__mdspan/extents.h> // std::extents
#  include <cuda/std/__utility/integer_sequence.h> // ::cuda::std::index_sequence
#  include <cuda/std/array> // std::array
#  include <cuda/std/cstddef> // size_t
#  include <cuda/std/type_traits> // make_unsigned_t

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * Utilities
 **********************************************************************************************************************/

// Compute the submdspan size of a given rank
template <::cuda::std::size_t Rank, typename IndexType, ::cuda::std::size_t... Extents, ::cuda::std::size_t... Indices>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::make_unsigned_t<IndexType>
sub_size(const ::cuda::std::extents<IndexType, Extents...>& ext, ::cuda::std::index_sequence<Indices...> = {})
{
  ::cuda::std::make_unsigned_t<IndexType> s = 1;
  for (IndexType i = Rank; i < sizeof...(Extents); i++)
  {
    s *= ext.extent(Rank + Indices);
  }
  return s;
}

// TODO: move to cuda::std
template <typename IndexType, ::cuda::std::size_t... Extents>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::make_unsigned_t<IndexType>
size(const ::cuda::std::extents<IndexType, Extents...>& ext)
{
  return sub_size<0>(ext);
}

// precompute modulo/division for each submdspan size (by rank)
template <typename IndexType, ::cuda::std::size_t... E, ::cuda::std::size_t... Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
sub_sizes_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  // deduction guides don't work with nvcc 11.x
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array<fast_mod_div_t, sizeof...(Ranks)>{fast_mod_div_t(sub_size<Ranks + 1>(ext))...};
}

// precompute modulo/division for each mdspan extent
template <typename IndexType, ::cuda::std::size_t... E, ::cuda::std::size_t... Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
extents_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array<fast_mod_div_t, sizeof...(Ranks)>{fast_mod_div_t(ext.extent(Ranks))...};
}

// GCC <= 9 workaround: Extent must be passed as type only, even const Extent& doesn't work
template <int Rank, typename Extent, ::cuda::std::size_t... Is>
_CCCL_NODISCARD _CCCL_HOST_DEVICE
_CCCL_FORCEINLINE constexpr bool is_sub_size_static(::cuda::std::index_sequence<Is...> = {})
{
  if constexpr (Rank >= Extent::rank())
  {
    return true;
  }
  else if constexpr (sizeof...(Is) == 0)
  {
    return is_sub_size_static<Rank, Extent>(::cuda::std::make_index_sequence<Extent::rank() - Rank>{});
  }
  else
  {
    return ((Extent::static_extent(Rank + Is) != ::cuda::std::dynamic_extent) && ...);
  }
}

} // namespace detail

CUB_NAMESPACE_END

#endif // if (_CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC)) || _CCCL_STD_VER >= 2020
