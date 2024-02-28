//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_NEXT_DETAIL_DIMENSIONS
#define _CUDA_NEXT_DETAIL_DIMENSIONS

#include <cuda/std/mdspan>

namespace cuda_next
{

template <typename T, size_t... Extents>
using dimensions = ::cuda::std::extents<T, Extents...>;

// not unsigned because of a bug in ::cuda::std::extents
using index_type = int;

template <typename T, size_t... Extents>
struct hierarchy_query_result : public dimensions<T, Extents...>
{
  using Dims = dimensions<T, Extents...>;
  using Dims::Dims;
  constexpr explicit _CCCL_HOST_DEVICE hierarchy_query_result(const Dims& dims)
      : Dims(dims)
  {}
  static_assert(Dims::rank() > 0);

  T x = Dims::extent(0);
  T y = Dims::rank() > 1 ? Dims::extent(1) : 1;
  T z = Dims::rank() > 2 ? Dims::extent(2) : 1;

  constexpr _CCCL_HOST_DEVICE operator dim3() const
  {
    return dim3(x, y, z);
  }
};

namespace detail
{
template <typename OpType>
constexpr size_t _CCCL_HOST_DEVICE merge_extents(size_t e1, size_t e2)
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
constexpr auto _CCCL_HOST_DEVICE
dims_op(const OpType& op, const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  // For now target only 3 dim extents
  static_assert(h1.rank() == h2.rank());
  static_assert(h1.rank() == 3);

  return dimensions<DstType, merge_extents<OpType>(Extents1, Extents2)...>(
    op(h1.extent(0), h2.extent(0)), op(h1.extent(1), h2.extent(1)), op(h1.extent(2), h2.extent(2)));
}

template <typename DstType, typename T1, size_t... Extents1, typename T2, size_t... Extents2>
constexpr auto _CCCL_HOST_DEVICE
dims_product(const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  return dims_op<DstType>(::cuda::std::multiplies(), h1, h2);
}

template <typename DstType, typename T1, size_t... Extents1, typename T2, size_t... Extents2>
constexpr auto _CCCL_HOST_DEVICE
dims_sum(const dimensions<T1, Extents1...>& h1, const dimensions<T2, Extents2...>& h2) noexcept
{
  return dims_op<DstType>(::cuda::std::plus(), h1, h2);
}

template <typename T, size_t... Extents>
auto constexpr _CCCL_HOST_DEVICE convert_to_query_result(const dimensions<T, Extents...>& result)
{
  return hierarchy_query_result<T, Extents...>(result);
}

auto constexpr _CCCL_HOST_DEVICE dim3_to_query_result(const dim3& dims)
{
  return dimensions<index_type, ::cuda::std::dynamic_extent, ::cuda::std::dynamic_extent, ::cuda::std::dynamic_extent>(
    dims.x, dims.y, dims.z);
}

template <typename TyTrunc, typename Index, typename Dims>
__device__ constexpr auto index_to_linear(const Index& index, const Dims& dims)
{
  static_assert(dims.rank() == 3);

  return (index.extent(2) * dims.extent(1) + index.extent(1)) * dims.extent(0) + index.extent(0);
}

} // namespace detail
} // namespace cuda_next
#endif
