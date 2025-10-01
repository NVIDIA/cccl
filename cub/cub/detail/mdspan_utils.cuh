// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
template <size_t Rank, typename IndexType, size_t Extent0, size_t... Extents>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::make_unsigned_t<IndexType>
sub_size(const ::cuda::std::extents<IndexType, Extent0, Extents...>& ext)
{
  ::cuda::std::make_unsigned_t<IndexType> s = 1;
  for (IndexType i = Rank; i < IndexType{1 + sizeof...(Extents)}; i++) // <- pointless comparison with zero-rank extent
  {
    s *= ext.extent(i);
  }
  return s;
}

// avoid pointless comparison of unsigned integer with zero (nvcc 11.x doesn't support nv_diag warning suppression)
template <size_t Rank, typename IndexType>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::make_unsigned_t<IndexType>
sub_size(const ::cuda::std::extents<IndexType>&)
{
  return ::cuda::std::make_unsigned_t<IndexType>{1};
}

// TODO: move to cuda::std
template <typename IndexType, size_t... Extents>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::make_unsigned_t<IndexType>
size(const ::cuda::std::extents<IndexType, Extents...>& ext)
{
  return cub::detail::sub_size<0>(ext);
}

// precompute modulo/division for each submdspan size (by rank)
template <typename IndexType, size_t... E, size_t... Ranks>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
sub_sizes_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  // deduction guides don't work with nvcc 11.x
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array<fast_mod_div_t, sizeof...(Ranks)>{fast_mod_div_t(sub_size<Ranks + 1>(ext))...};
}

// precompute modulo/division for each mdspan extent
template <typename IndexType, size_t... E, size_t... Ranks>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
extents_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Ranks...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  return ::cuda::std::array<fast_mod_div_t, sizeof...(Ranks)>{fast_mod_div_t(ext.extent(Ranks))...};
}

// GCC <= 9 constexpr workaround: Extent must be passed as type only, even const Extent& doesn't work
template <int Rank, typename Extents>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr bool is_sub_size_static()
{
  using index_type = typename Extents::index_type;
  for (index_type i = Rank; i < Extents::rank(); i++)
  {
    if (Extents::static_extent(i) == ::cuda::std::dynamic_extent)
    {
      return false;
    }
  }
  return true;
}

template <typename MappingTypeLhs, typename MappingTypeRhs>
[[nodiscard]] _CCCL_API bool have_same_strides(const MappingTypeLhs& mapping_lhs, const MappingTypeRhs& mapping_rhs)
{
  auto extents_lhs = mapping_lhs.extents();
  auto extents_rhs = mapping_rhs.extents();
  _CCCL_ASSERT(extents_lhs.rank() == extents_rhs.rank(), "extents must have the same rank");
  for (size_t i = 0; i < extents_lhs.rank(); i++)
  {
    if (mapping_lhs.stride(i) != mapping_rhs.stride(i))
    {
      return false;
    }
  }
  return true;
}

} // namespace detail

CUB_NAMESPACE_END
