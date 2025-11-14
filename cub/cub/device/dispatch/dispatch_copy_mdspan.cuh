// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
#pragma once

#include <cub/config.cuh>

#include <cuda/std/__type_traits/is_same.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/util_debug.cuh>

#include <cuda/std/functional>
#include <cuda/std/mdspan>

CUB_NAMESPACE_BEGIN

namespace detail::copy_mdspan
{
template <typename MdspanIn, typename MdspanOut>
struct copy_mdspan_t
{
  MdspanIn mdspan_in;
  MdspanOut mdspan_out;

  _CCCL_API copy_mdspan_t(MdspanIn mdspan_in, MdspanOut mdspan_out)
      : mdspan_in{mdspan_in}
      , mdspan_out{mdspan_out}
  {}

  template <typename Idx, typename... Indices>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void operator()(Idx, Indices... indices)
  {
    mdspan_out(indices...) = mdspan_in(indices...);
  }
};

template <typename T_In,
          typename E_In,
          typename L_In,
          typename A_In,
          typename T_Out,
          typename E_Out,
          typename L_Out,
          typename A_Out>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
copy(::cuda::std::mdspan<T_In, E_In, L_In, A_In> mdspan_in,
     ::cuda::std::mdspan<T_Out, E_Out, L_Out, A_Out> mdspan_out,
     ::cudaStream_t stream)
{
  if (mdspan_in.is_exhaustive() && mdspan_out.is_exhaustive()
      && detail::have_same_strides(mdspan_in.mapping(), mdspan_out.mapping()))
  {
    return cub::DeviceTransform::Transform(
      mdspan_in.data_handle(),
      mdspan_out.data_handle(),
      mdspan_in.size(),
      ::cuda::proclaim_copyable_arguments(::cuda::std::identity{}),
      stream);
  }
  // TODO (fbusato): add ForEachInLayout when mdspan_in and mdspan_out have compatible layouts
  // Compatible layouts could use more efficient iteration patterns
  return cub::DeviceFor::ForEachInExtents(mdspan_in.extents(), copy_mdspan_t{mdspan_in, mdspan_out}, stream);
}
} // namespace detail::copy_mdspan

CUB_NAMESPACE_END
