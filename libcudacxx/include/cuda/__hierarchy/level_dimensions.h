//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_LEVEL_DIMENSIONS_H
#define _CUDA___HIERARCHY_LEVEL_DIMENSIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/block_level.h>
#  include <cuda/__hierarchy/cluster_level.h>
#  include <cuda/__hierarchy/grid_level.h>
#  include <cuda/__hierarchy/hierarchy_levels.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/__type_traits/is_integer.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
/* Keeping it around in case issues like
https://github.com/NVIDIA/cccl/issues/522 template <typename T, size_t...
Extents> struct extents_corrected : public ::cuda::std::extents<T, Extents...> {
    using ::cuda::std::extents<T, Extents...>::extents;

    template <typename ::cuda::std::extents<T, Extents...>::rank_type Id>
    _CCCL_API constexpr auto extent_corrected() const {
        if constexpr (::cuda::std::extents<T, Extents...>::static_extent(Id) !=
::cuda::std::dynamic_extent) { return this->static_extent(Id);
        }
        else {
            return this->extent(Id);
        }
    }
};
*/

template <class _Dims>
struct __dimensions_handler
{
  static constexpr bool __is_type_supported = ::cuda::std::__cccl_is_integer_v<_Dims>;

  [[nodiscard]] _CCCL_API static constexpr auto __translate(const _Dims& __dims) noexcept
  {
    return ::cuda::std::extents<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>(
      static_cast<unsigned>(__dims));
  }
};

template <>
struct __dimensions_handler<::dim3>
{
  static constexpr bool __is_type_supported = true;

  [[nodiscard]] _CCCL_API static constexpr auto __translate(const ::dim3& __dims) noexcept
  {
    return ::cuda::std::dims<3, dimensions_index_type>(__dims.x, __dims.y, __dims.z);
  }
};

template <class _Dims, _Dims _Val>
struct __dimensions_handler<::cuda::std::integral_constant<_Dims, _Val>>
{
  static constexpr bool __is_type_supported = ::cuda::std::__cccl_is_integer_v<_Dims>;

  [[nodiscard]] _CCCL_API static constexpr auto __translate(const _Dims& __dims) noexcept
  {
    return ::cuda::std::extents<dimensions_index_type, static_cast<::cuda::std::size_t>(__dims), 1, 1>();
  }
};
} // namespace __detail

/**
 * @brief Type representing dimensions of a level in a thread hierarchy.
 *
 * This type combines a level type like grid_level or block_level with
 * a cuda::std::extents object to describe dimensions of a level in a thread
 * hierarchy. This type is not intended to be created explicitly and *_dims
 * functions creating them should be used instead. They will translate the input
 * arguments to a correct cuda::std::extents to be stored inside
 * hierarchy_level_desc.
 * While this type can be used to access the stored dimensions,
 * the main usage is to pass a number of hierarchy_level_desc objects
 * to make_hierarchy function in order to create a hierarchy.
 * This type does not store what the unit is for the stored dimensions,
 * it is instead implied by the level below it in a hierarchy object.
 * In case there is a need to store more information about a specific level,
 * for example some library-specific information, this type can be derived
 * from and the resulting type can be used to build the hierarchy.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * auto hierarchy = make_hierarchy(grid_dims(256), block_dims<8, 8, 8>());
 * assert(hierarchy.level(grid).dims.x == 256);
 * @endcode
 * @par
 *
 * @tparam Level
 *   Type indicating which hierarchy level this is
 *
 * @tparam Dimensions
 *   Type holding the dimensions of this level
 */
template <class _Level, class _Exts>
class _CCCL_DECLSPEC_EMPTY_BASES hierarchy_level_desc : __hierarchy_level_desc_base
{
  static_assert(__is_hierarchy_level_v<_Level>);
  static_assert(::cuda::std::__is_cuda_std_extents_v<_Exts>);

  // Needs alignas to work around an issue with tuple
  alignas(16) _Exts __exts_; // Unit for dimensions is implicit

public:
  using level_type   = _Level;
  using extents_type = _Exts;

  _CCCL_HIDE_FROM_ABI constexpr hierarchy_level_desc() noexcept = default;

  _CCCL_API constexpr hierarchy_level_desc(const _Exts& __exts) noexcept
      : __exts_(__exts)
  {}

  [[nodiscard]] _CCCL_API constexpr _Exts extents() const noexcept
  {
    return __exts_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const hierarchy_level_desc& __lhs, const hierarchy_level_desc& __rhs) noexcept
  {
    return __lhs.__exts_ == __rhs.__exts_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const hierarchy_level_desc& __lhs, const hierarchy_level_desc& __rhs) noexcept
  {
    return __lhs.__exts_ != __rhs.__exts_;
  }
};

/**
 * @brief Creates an instance of hierarchy_level_desc describing grid_level
 *
 * This function creates a statically sized level from up to three template
 * arguments.
 */
template <::cuda::std::size_t _XDim, ::cuda::std::size_t _YDim = 1, ::cuda::std::size_t _ZDim = 1>
[[nodiscard]] _CCCL_API constexpr auto grid_dims() noexcept
{
  return hierarchy_level_desc<grid_level, ::cuda::std::extents<dimensions_index_type, _XDim, _YDim, _ZDim>>();
}

/**
 * @brief Creates an instance of hierarchy_level_desc describing grid_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <class _Dims>
[[nodiscard]] _CCCL_API constexpr auto grid_dims(_Dims __dims) noexcept
{
  static_assert(__detail::__dimensions_handler<_Dims>::__is_type_supported);
  auto __translated_dims = __detail::__dimensions_handler<_Dims>::__translate(__dims);
  return hierarchy_level_desc<grid_level, decltype(__translated_dims)>(__translated_dims);
}

/**
 * @brief Creates an instance of hierarchy_level_desc describing cluster_level
 *
 * This function creates a statically sized level from up to three template
 * arguments.
 */
template <::cuda::std::size_t _XDim, ::cuda::std::size_t _YDim = 1, ::cuda::std::size_t _ZDim = 1>
[[nodiscard]] _CCCL_API constexpr auto cluster_dims() noexcept
{
  return hierarchy_level_desc<cluster_level, ::cuda::std::extents<dimensions_index_type, _XDim, _YDim, _ZDim>>();
}

/**
 * @brief Creates an instance of hierarchy_level_desc describing cluster_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <class _Dims>
[[nodiscard]] _CCCL_API constexpr auto cluster_dims(_Dims __dims) noexcept
{
  static_assert(__detail::__dimensions_handler<_Dims>::__is_type_supported);
  auto __translated_dims = __detail::__dimensions_handler<_Dims>::__translate(__dims);
  return hierarchy_level_desc<cluster_level, decltype(__translated_dims)>(__translated_dims);
}

/**
 * @brief Creates an instance of hierarchy_level_desc describing block_level
 *
 * This function creates a statically sized level from up to three template
 * arguments.
 */
template <::cuda::std::size_t _XDim, ::cuda::std::size_t _YDim = 1, ::cuda::std::size_t _ZDim = 1>
[[nodiscard]] _CCCL_API constexpr auto block_dims() noexcept
{
  return hierarchy_level_desc<block_level, ::cuda::std::extents<dimensions_index_type, _XDim, _YDim, _ZDim>>();
}

/**
 * @brief Creates an instance of hierarchy_level_desc describing block_level
 *
 * This function creates the level from an integral or dim3 argument.
 */
template <class _Dims>
[[nodiscard]] _CCCL_API constexpr auto block_dims(_Dims __dims) noexcept
{
  static_assert(__detail::__dimensions_handler<_Dims>::__is_type_supported);
  auto __translated_dims = __detail::__dimensions_handler<_Dims>::__translate(__dims);
  return hierarchy_level_desc<block_level, decltype(__translated_dims)>(__translated_dims);
}
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_LEVEL_DIMENSIONS_H
