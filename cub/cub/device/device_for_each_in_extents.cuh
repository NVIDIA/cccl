/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#if __cccl_lib_mdspan

#  include <cub/detail/nvtx.cuh>
#  include <cub/device/dispatch/dispatch_for_each_in_extents.cuh>
#  include <cub/util_namespace.cuh>

#  include <cuda/std/__mdspan/extents.h>

CUB_NAMESPACE_BEGIN

struct DeviceForEachInExtents
{
public:
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Unfold a multi-dimensional extents into
  //!
  //! - a single linear index that represents the current iteration
  //! - indices of each extent dimension
  //!
  //! Then apply a function object to the results.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use ``ForEachInExtents`` to tabulate a 3D array with its
  //! coordinates.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-t
  //!     :end-before: example-end bulk-square-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-wo-temp-storage
  //!     :end-before: example-end bulk-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam IndexType
  //!   is an integral type that represents the extent index space (automatically deduced)
  //!
  //! @tparam Extents
  //!   are the extent sizes for each rank index (automatically deduced)
  //!
  //! @tparam OpType
  //!   is a function object with arity equal to the number of extents + 1 for the linear index (iteration)
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`,
  //!   the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] extents
  //!   Extents object that represents a multi-dimensional index space
  //!
  //! @param[in] op
  //!   Function object to apply to each linear index (iteration) and multi-dimensional coordinates
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `NULL`
  //!
  //! @return cudaError_t
  //!   error status
  template <typename IndexType, ::cuda::std::size_t... Extents, typename OpType>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachInExtents(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const ::cuda::std::extents<IndexType, Extents...>& extents,
    OpType op,
    cudaStream_t stream = {})
  {
    // TODO: check dimensions overflows
    // TODO: check tha arity of OpType is equal to sizeof...(ExtentsType)
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }
    return DeviceForEachInExtents::ForEachInExtents(extents, op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Unfold a multi-dimensional extents into
  //!
  //! - a single linear index that represents the current iteration
  //! - indices of each extent dimension
  //!
  //! Then apply a function object to the results.
  //!
  //! - The return value of ``op``, if any, is ignored.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use ``ForEachInExtents`` to tabulate a 3D array with its
  //! coordinates.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-square-t
  //!     :end-before: example-end bulk-square-t
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_for_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin bulk-wo-temp-storage
  //!     :end-before: example-end bulk-wo-temp-storage
  //!
  //! @endrst
  //!
  //! @tparam IndexType
  //!   is an integral type that represents the extent index space (automatically deduced)
  //!
  //! @tparam Extents
  //!   are the extent sizes for each rank index (automatically deduced)
  //!
  //! @tparam OpType
  //!   is a function object with arity equal to the number of extents + 1 for the linear index (iteration)
  //!
  //! @param[in] extents
  //!   Extents object that represents a multi-dimensional index space
  //!
  //! @param[in] op
  //!   Function object to apply to each linear index (iteration) and multi-dimensional coordinates
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within. Default stream is `NULL`
  //!
  //! @return cudaError_t
  //!   error status
  template <typename IndexType, ::cuda::std::size_t... Extents, typename OpType>
  CUB_RUNTIME_FUNCTION static cudaError_t
  ForEachInExtents(const ::cuda::std::extents<IndexType, Extents...>& extents, OpType op, cudaStream_t stream = {})
  {
    using ExtentsType = ::cuda::std::extents<IndexType, Extents...>;
    // TODO: check dimensions overflows
    // TODO: check tha arity of OpType is equal to sizeof...(ExtentsType)
    CUB_DETAIL_NVTX_RANGE_SCOPE("cub::DeviceFor::ForEachInExtents");
    return detail::for_each_in_extents::dispatch_t<ExtentsType, OpType>::dispatch(extents, op, stream);
  }
};

CUB_NAMESPACE_END

#endif // __cccl_lib_mdspan
