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

#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>

CUB_NAMESPACE_BEGIN

namespace detail
{

// Compute the submdspan size of a given rank
template <typename IndexType, size_t... Extents>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<IndexType>
sub_size(const ::cuda::std::extents<IndexType, Extents...>& ext, int start, int end)
{
  ::cuda::std::make_unsigned_t<IndexType> s = 1;
  for (auto i = start; i < end; i++)
  {
    s *= ext.extent(i);
  }
  return s;
}

// TODO: move to cuda::std
template <typename IndexType, size_t... Extents>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<IndexType>
size(const ::cuda::std::extents<IndexType, Extents...>& ext)
{
  return cub::detail::sub_size(ext, 0, ext.rank());
}

// precompute modulo/division for each submdspan size (by rank)
template <typename IndexType, size_t... E, size_t... Ranks>
[[nodiscard]] _CCCL_API auto
sub_sizes_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array{fast_mod_div_t(sub_size(ext, Ranks + 1, ext.rank()))...};
}

// precompute modulo/division for each mdspan extent
template <typename IndexType, size_t... E, size_t... Ranks>
[[nodiscard]] _CCCL_API auto
extents_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array{fast_mod_div_t(ext.extent(Ranks))...};
}

// GCC <= 9 constexpr workaround: Extent must be passed as type only, even const Extent& doesn't work
template <typename Extents>
[[nodiscard]] _CCCL_API constexpr bool is_extents_in_range_static(int start, int end)
{
  for (auto i = start; i < end; i++)
  {
    if (Extents::static_extent(i) == ::cuda::std::dynamic_extent)
    {
      return false;
    }
  }
  return true;
}

} // namespace detail

CUB_NAMESPACE_END
