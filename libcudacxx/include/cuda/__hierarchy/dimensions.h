//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_DIMENSIONS_H
#define _CUDA___HIERARCHY_DIMENSIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/std/__mdspan/extents.h>
#  include <cuda/std/functional>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, size_t... _Extents>
using dimensions = ::cuda::std::extents<_Tp, _Extents...>;

using dimensions_index_type = unsigned;

/**
 * @brief Type representing a result of a multi-dimensional hierarchy query.
 *
 * Returned from extents and index queries.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 *
 * template <typename Dimensions>
 * __global__ void kernel(Dimensions dims)
 * {
 *     auto ext = dims.extents();
 *
 *     // Can be accessed like cuda::std::extents or like dim3
 *     assert(ext.extent(0) == expected);
 *     assert(ext.x == expected);
 *
 *     // Can be converted to dim3
 *     dim3 dimensions = ext;
 * }
 * @endcode
 * @par
 *
 * @tparam T
 *   Type of the result for each dimension
 *
 * @tparam Extents
 *   Extents of the result
 */
template <class _Tp, size_t... _Extents>
struct hierarchy_query_result_org : public dimensions<_Tp, _Extents...>
{
  using _Dims = dimensions<_Tp, _Extents...>;
  using _Dims::_Dims;

  _CCCL_API constexpr hierarchy_query_result_org()
      : _Dims()
      , x(_Dims::extent(0))
      , y(_Dims::rank() > 1 ? _Dims::extent(1) : 1)
      , z(_Dims::rank() > 2 ? _Dims::extent(2) : 1)
  {}

  _CCCL_API explicit constexpr hierarchy_query_result_org(const _Dims& dims)
      : _Dims(dims)
      , x(_Dims::extent(0))
      , y(_Dims::rank() > 1 ? _Dims::extent(1) : 1)
      , z(_Dims::rank() > 2 ? _Dims::extent(2) : 1)
  {}

  static_assert(_Dims::rank() > 0 && _Dims::rank() <= 3);

  const _Tp x;
  const _Tp y;
  const _Tp z;

  _CCCL_API constexpr operator ::dim3() const
  {
    return ::dim3(static_cast<::cuda::std::uint32_t>(x),
                  static_cast<::cuda::std::uint32_t>(y),
                  static_cast<::cuda::std::uint32_t>(z));
  }
};

namespace __detail
{
template <class _Op>
[[nodiscard]] _CCCL_API constexpr size_t __merge_extents(size_t __e1, size_t __e2)
{
  if (__e1 == ::cuda::std::dynamic_extent || __e2 == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::dynamic_extent;
  }
  else
  {
    _Op __op{};
    return __op(__e1, __e2);
  }
}

template <class _Dst, class _Op, class _T1, size_t... _E1, class _T2, size_t... _E2>
[[nodiscard]] _CCCL_API constexpr auto
__dims_op(const _Op& __op, const dimensions<_T1, _E1...>& __h1, const dimensions<_T2, _E2...>& __h2) noexcept
{
  // For now target only 3 dim extents
  static_assert(sizeof...(_E1) == sizeof...(_E2));
  static_assert(sizeof...(_E1) == 3);

  return dimensions<_Dst, __merge_extents<_Op>(_E1, _E2)...>(
    __op(static_cast<_Dst>(__h1.extent(0)), __h2.extent(0)),
    __op(static_cast<_Dst>(__h1.extent(1)), __h2.extent(1)),
    __op(static_cast<_Dst>(__h1.extent(2)), __h2.extent(2)));
}

template <class _Dst, class _T1, size_t... _E1, class _T2, size_t... _E2>
[[nodiscard]] _CCCL_API constexpr auto
__dims_product(const dimensions<_T1, _E1...>& __h1, const dimensions<_T2, _E2...>& __h2) noexcept
{
  return __dims_op<_Dst>(::cuda::std::multiplies<void>(), __h1, __h2);
}

template <class _Dst, class _T1, size_t... _E1, class _T2, size_t... _E2>
[[nodiscard]] _CCCL_API constexpr auto
__dims_sum(const dimensions<_T1, _E1...>& __h1, const dimensions<_T2, _E2...>& __h2) noexcept
{
  return __dims_op<_Dst>(::cuda::std::plus<void>(), __h1, __h2);
}

template <class _Tp, size_t... _Extents>
[[nodiscard]] _CCCL_API constexpr auto __convert_to_query_result(const dimensions<_Tp, _Extents...>& __result)
{
  return hierarchy_query_result_org<_Tp, _Extents...>(__result);
}

[[nodiscard]] _CCCL_API constexpr auto __dim3_to_dims(const ::dim3& dims)
{
  return dimensions<dimensions_index_type,
                    ::cuda::std::dynamic_extent,
                    ::cuda::std::dynamic_extent,
                    ::cuda::std::dynamic_extent>(dims.x, dims.y, dims.z);
}

template <class _TyTrunc, class _Index, class _Dims>
[[nodiscard]] _CCCL_API constexpr auto __index_to_linear(const _Index& __index, const _Dims& __dims)
{
  static_assert(_Dims::rank() == 3);

  return (static_cast<_TyTrunc>(__index.extent(2)) * __dims.extent(1) + __index.extent(1)) * __dims.extent(0)
       + __index.extent(0);
}
} // namespace __detail
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_DIMENSIONS_H
