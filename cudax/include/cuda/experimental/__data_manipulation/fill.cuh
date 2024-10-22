//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DATA_MANIPULATION_FILL
#define __CUDAX_DATA_MANIPULATION_FILL

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/span>

#include <cuda/experimental/__launch/launch_transform.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

template <typename _DstTy, ::std::size_t _DstSize>
void __fill_bytes_impl(stream_ref __stream, _CUDA_VSTD::span<_DstTy, _DstSize> __dst, uint8_t __value)
{
  static_assert(!_CUDA_VSTD::is_const_v<_DstTy>, "Fill destination can't be const");
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<_DstTy>);

  // TODO do a host callback if not device accessible?
  _CCCL_TRY_CUDA_API(
    ::cudaMemsetAsync, "Failed to perform a fill", __dst.data(), __value, __dst.size_bytes(), __stream.get());
}

//! @brief Launches an operation to bytewise fill the memory into the provided stream.
//!
//! Destination needs to either be a `contiguous_range` or implicitly/launch transform
//! into one. It can't reside in pagable host memory.
//! Destination type is required to be trivially copyable.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into every byte in the destination
_LIBCUDACXX_TEMPLATE(typename _DstTy)
_LIBCUDACXX_REQUIRES(_CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_DstTy>>)
void fill_bytes(stream_ref __stream, _DstTy&& __dst, uint8_t __value)
{
  __fill_bytes_impl(__stream,
                    _CUDA_VSTD::span(static_cast<detail::__as_copy_arg_t<_DstTy>>(
                      detail::__launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst)))),
                    __value);
}

} // namespace cuda::experimental
#endif // __CUDAX_DATA_MANIPULATION_FILL
