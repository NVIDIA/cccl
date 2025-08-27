//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__HIERARCHY_DIMENSIONS_CUH
#define _CUDAX__HIERARCHY_DIMENSIONS_CUH

#include <cuda/std/functional>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

template <typename T, size_t... Extents>
using dimensions = ::cuda::std::extents<T, Extents...>;

// not unsigned because of a bug in ::cuda::std::extents
using dimensions_index_type = int;

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
template <typename T, size_t... Extents>
struct hierarchy_query_result : public dimensions<T, Extents...>
{
  using Dims = dimensions<T, Extents...>;
  using Dims::Dims;

  _CCCL_HOST_DEVICE constexpr hierarchy_query_result()
      : Dims()
      , x(Dims::extent(0))
      , y(Dims::rank() > 1 ? Dims::extent(1) : 1)
      , z(Dims::rank() > 2 ? Dims::extent(2) : 1)
  {}

  _CCCL_HOST_DEVICE explicit constexpr hierarchy_query_result(const Dims& dims)
      : Dims(dims)
      , x(Dims::extent(0))
      , y(Dims::rank() > 1 ? Dims::extent(1) : 1)
      , z(Dims::rank() > 2 ? Dims::extent(2) : 1)
  {}

  static_assert(Dims::rank() > 0 && Dims::rank() <= 3);

  const T x;
  const T y;
  const T z;

  _CCCL_HOST_DEVICE constexpr operator dim3() const
  {
    return dim3(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z));
  }
};

namespace __detail
{
template <typename OpType>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr size_t merge_extents(size_t e1, size_t e2)
{
  if (e1 == ::cuda::std::dynamic_extent || e2 == ::cuda::std::dynamic_extent)
  {
    return ::cuda::std::dynamic_extent;
  }
  else
  {
    OpType op;
    return op(e1, e2);
  }
}

template <typename DstType, typename OpType, typename T1, size_t... Extents1, typename T2, size_t... Extents2>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
dims_op(const OpType& op, const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  // For now target only 3 dim extents
  static_assert(sizeof...(Extents1) == sizeof...(Extents2));
  static_assert(sizeof...(Extents1) == 3);

  return dimensions<DstType, merge_extents<OpType>(Extents1, Extents2)...>(
    op(static_cast<DstType>(h1.extent(0)), h2.extent(0)),
    op(static_cast<DstType>(h1.extent(1)), h2.extent(1)),
    op(static_cast<DstType>(h1.extent(2)), h2.extent(2)));
}

template <typename DstType, typename T1, size_t... Extents1, typename T2, size_t... Extents2>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
dims_product(const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  return dims_op<DstType>(::cuda::std::multiplies(), h1, h2);
}

template <typename DstType, typename T1, size_t... Extents1, typename T2, size_t... Extents2>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
dims_sum(const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  return dims_op<DstType>(::cuda::std::plus(), h1, h2);
}

template <typename T, size_t... Extents>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto convert_to_query_result(const dimensions<T, Extents...>& result)
{
  return hierarchy_query_result<T, Extents...>(result);
}

[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto dim3_to_dims(const dim3& dims)
{
  return dimensions<dimensions_index_type,
                    ::cuda::std::dynamic_extent,
                    ::cuda::std::dynamic_extent,
                    ::cuda::std::dynamic_extent>(dims.x, dims.y, dims.z);
}

template <typename TyTrunc, typename Index, typename Dims>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto index_to_linear(const Index& index, const Dims& dims)
{
  static_assert(Dims::rank() == 3);

  return (static_cast<TyTrunc>(index.extent(2)) * dims.extent(1) + index.extent(1)) * dims.extent(0) + index.extent(0);
}

} // namespace __detail
} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__HIERARCHY_DIMENSIONS_CUH
