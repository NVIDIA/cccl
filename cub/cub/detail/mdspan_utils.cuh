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

#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN
namespace detail
{
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code (even if there are no branches!)

// Compute the submdspan size of a given rank
template <typename IndexType, size_t... Extents>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<IndexType>
size_range(const ::cuda::std::extents<IndexType, Extents...>& ext, int start, int end)
{
  _CCCL_ASSERT(start >= 0 && end <= static_cast<int>(ext.rank()), "invalid start or end");
  ::cuda::std::make_unsigned_t<IndexType> s = 1;
  for (auto i = start; i < end; i++)
  {
    s *= ext.extent(i);
  }
  return s;
}

_CCCL_DIAG_POP // MSVC(4702)

  template <typename IndexType, size_t... Extents>
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<IndexType>
  size(const ::cuda::std::extents<IndexType, Extents...>& ext)
{
  return cub::detail::size_range(ext, 0, static_cast<int>(ext.rank()));
}

template <bool IsLayoutRight, int Position, typename IndexType, size_t... E>
[[nodiscard]] _CCCL_API auto sub_size_fast_div_mod_impl(const ::cuda::std::extents<IndexType, E...>& ext)
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  constexpr auto start = IsLayoutRight ? Position + 1 : 0;
  constexpr auto end   = IsLayoutRight ? sizeof...(E) : Position;
  return fast_mod_div_t(cub::detail::size_range(ext, start, end));
}

// precompute modulo/division for each submdspan size (by rank)
template <bool IsLayoutRight, typename IndexType, size_t... E, size_t... Positions>
[[nodiscard]] _CCCL_API auto
sub_sizes_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Positions...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  using array_t        = ::cuda::std::array<fast_mod_div_t, sizeof...(Positions)>;
  return array_t{cub::detail::sub_size_fast_div_mod_impl<IsLayoutRight, Positions>(ext)...};
}

// precompute modulo/division for each mdspan extent
template <typename IndexType, size_t... E, size_t... Positions>
[[nodiscard]] _CCCL_API auto
extents_fast_div_mod(const ::cuda::std::extents<IndexType, E...>& ext, ::cuda::std::index_sequence<Positions...> = {})
{
  using fast_mod_div_t = fast_div_mod<IndexType>;
  using array_t        = ::cuda::std::array<fast_mod_div_t, sizeof...(Positions)>;
  return array_t{fast_mod_div_t(ext.extent(Positions))...};
}

// GCC <= 9 constexpr workaround: Extent must be passed as type only, even const Extent& doesn't work
template <typename Extents>
[[nodiscard]] _CCCL_API constexpr bool are_extents_in_range_static(int start, int end)
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
