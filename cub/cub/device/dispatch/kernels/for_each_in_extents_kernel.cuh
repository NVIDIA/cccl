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
#include <cub/detail/mdspan_utils.cuh> // is_sub_size_static
#include <cub/detail/type_traits.cuh> // implicit_prom_t

#include <cuda/std/cstddef> // size_t
#include <cuda/std/mdspan> // dynamic_extent
#include <cuda/std/type_traits> // make_unsigned_t

CUB_NAMESPACE_BEGIN
namespace detail::for_each_in_extents
{

// Return the extents at the given rank. If the extents is static, return it, otherwise return the precomputed value
template <int Rank, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto dispatch_extent_at(ExtentType extents, FastDivModType dynamic_extent)
{
  if constexpr (ExtentType::static_extent(Rank) != _CUDA_VSTD::dynamic_extent)
  {
    using extent_index_type   = typename ExtentType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = _CUDA_VSTD::make_unsigned_t<index_type>;
    return static_cast<unsigned_index_type>(extents.static_extent(Rank));
  }
  else
  {
    return dynamic_extent;
  }
}

template <int Rank, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extents_sub_size(ExtentType extents, FastDivModType extent_sub_size)
{
  if constexpr (cub::detail::is_sub_size_static<Rank + 1, ExtentType>())
  {
    return cub::detail::sub_size<Rank + 1>(extents);
  }
  else
  {
    return extent_sub_size;
  }
}

template <int Rank, typename IndexType, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
coordinate_at(IndexType index, ExtentType extents, FastDivModType extent_sub_size, FastDivModType dynamic_extent)
{
  using cub::detail::for_each_in_extents::dispatch_extent_at;
  using cub::detail::for_each_in_extents::get_extents_sub_size;
  using extent_index_type = typename ExtentType::index_type;
  return static_cast<extent_index_type>(
    (index / get_extents_sub_size<Rank>(extents, extent_sub_size)) % dispatch_extent_at<Rank>(extents, dynamic_extent));
}

/***********************************************************************************************************************
 * Kernel entry points
 **********************************************************************************************************************/

template <typename ChainedPolicyT, typename Func, typename ExtentType, typename FastDivModArrayType, size_t... Ranks>
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads)
  CUB_DETAIL_KERNEL_ATTRIBUTES void static_kernel(
    [[maybe_unused]] Func func,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const ExtentType extents,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
    [[maybe_unused]] _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  using cub::detail::for_each_in_extents::coordinate_at;
  using active_policy_t   = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using extent_index_type = typename ExtentType::index_type;
  using offset_t          = implicit_prom_t<extent_index_type>;
  auto stride             = static_cast<offset_t>(gridDim.x * offset_t{active_policy_t::block_threads});
  auto id                 = static_cast<offset_t>(threadIdx.x + blockIdx.x * offset_t{active_policy_t::block_threads});
  auto size               = static_cast<offset_t>(cub::detail::size(extents));
  for (auto i = id; i < size; i += stride)
  {
    func(i, coordinate_at<Ranks>(i, extents, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }
}

template <typename Func, typename ExtentType, typename FastDivModArrayType, size_t... Ranks>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(
  Func func,
  _CCCL_GRID_CONSTANT const ExtentType extents,
  _CCCL_GRID_CONSTANT const FastDivModArrayType sub_sizes_div_array,
  _CCCL_GRID_CONSTANT const FastDivModArrayType extents_mod_array)
{
  using cub::detail::for_each_in_extents::coordinate_at;
  using extent_index_type = typename ExtentType::index_type;
  using offset_t          = implicit_prom_t<extent_index_type>;
  auto stride             = static_cast<offset_t>(gridDim.x * static_cast<offset_t>(blockDim.x));
  auto id                 = static_cast<offset_t>(threadIdx.x + blockIdx.x * static_cast<offset_t>(blockDim.x));
  auto size               = static_cast<offset_t>(cub::detail::size(extents));
  for (auto i = id; i < size; i += stride)
  {
    func(i, coordinate_at<Ranks>(i, extents, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }
}

} // namespace detail::for_each_in_extents
CUB_NAMESPACE_END
