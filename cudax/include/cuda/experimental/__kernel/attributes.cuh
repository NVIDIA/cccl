//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_ATTRIBUTES_CUH
#define _CUDAX___KERNEL_ATTRIBUTES_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__kernel/kernel_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace __detail
{

template <::CUfunction_attribute _Attr, typename _Type>
struct __kernel_attr_impl
{
  using type = _Type;

  [[nodiscard]] constexpr operator ::CUfunction_attribute() const noexcept
  {
    return _Attr;
  }

  template <class _Signature>
  [[nodiscard]] type operator()(kernel_ref<_Signature> __kernel, device_ref __dev) const
  {
    auto __res = __detail::driver::kernelGetAttribute(_Attr, __kernel.get(), __detail::driver::deviceGet(__dev.get()));

    // Adjust the architecture version attributes to match the expected format
    if constexpr (_Attr == ::CU_FUNC_ATTRIBUTE_PTX_VERSION || _Attr == ::CU_FUNC_ATTRIBUTE_BINARY_VERSION)
    {
      __res *= 10;
    }
    return static_cast<type>(__res);
  }
};

template <::CUfunction_attribute _Attr>
struct __kernel_attr : __kernel_attr_impl<_Attr, int>
{};

template <>
struct __kernel_attr<::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES> //
    : __kernel_attr_impl<::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, _CUDA_VSTD::size_t>
{};
template <>
struct __kernel_attr<::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES> //
    : __kernel_attr_impl<::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, _CUDA_VSTD::size_t>
{};
template <>
struct __kernel_attr<::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES> //
    : __kernel_attr_impl<::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, _CUDA_VSTD::size_t>
{};
template <>
struct __kernel_attr<::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA> //
    : __kernel_attr_impl<::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, bool>
{};
template <>
struct __kernel_attr<::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET> //
    : __kernel_attr_impl<::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, bool>
{};

} // namespace __detail

struct kernel_attributes
{
  // Maximum number of threads per block
  using max_threads_per_block_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK>;
  static constexpr max_threads_per_block_t max_threads_per_block{};

  // The size in bytes of statically-allocated shared memory required by this kernel
  using shared_size_bytes_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES>;
  static constexpr shared_size_bytes_t shared_memory_size{};

  // The size in bytes of user-allocated constant memory required by this kernel
  using const_size_bytes_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES>;
  static constexpr const_size_bytes_t const_memory_size{};

  // The size in bytes of local memory used by each thread of this kernel
  using local_size_bytes_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES>;
  static constexpr local_size_bytes_t local_memory_size{};

  // The number of registers used by each thread of this kernel
  using num_regs_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_NUM_REGS>;
  static constexpr num_regs_t num_regs{};

  // The virtual architecture for which the kernel was compiled
  using virtual_arch_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_PTX_VERSION>;
  static constexpr virtual_arch_t virtual_arch{};

  // The real architecture for which the kernel was compiled
  using real_arch_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_BINARY_VERSION>;
  static constexpr real_arch_t real_arch{};

  // The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set
  using cache_mode_ca_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA>;
  static constexpr cache_mode_ca_t cache_mode_ca{};

  // Must the kernel be launched with a cluster size?
  using cluster_size_must_be_set_t = __detail::__kernel_attr<::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET>;
  static constexpr cluster_size_must_be_set_t cluster_size_must_be_set{};
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___KERNEL_ATTRIBUTES_CUH
