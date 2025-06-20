//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_KERNEL_REF
#define _CUDAX___KERNEL_KERNEL_REF

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/address_space.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/string_view>

#include <cuda/experimental/__device/device_ref.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#include <string>

#include <cuda.h>

namespace cuda::experimental
{

//! @brief A non-owning representation of a CUDA kernel
//!
//! @tparam _Signature The signature of the kernel
//!
//! @note The return type of the kernel must be `void`
template <class _Signature>
class kernel_ref;

template <class... _Args>
class kernel_ref<void(_Args...)>
{
  [[nodiscard]] int __get_attrib(::CUfunction_attribute __attr, device_ref __dev) const
  {
    return __detail::driver::kernelGetAttribute(__attr, __kernel_, __detail::driver::deviceGet(__dev.get()));
  }

public:
#if _CCCL_CTK_BELOW(12, 1)
  using value_type = ::CUkernel;
#else // ^^^ _CCCL_CTK_BELOW(12, 1) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 1) vvv
  using value_type = ::cudaKernel_t;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 1) ^^^

  kernel_ref(_CUDA_VSTD::nullptr_t) = delete;

  //! @brief Constructs a `kernel_ref` from a kernel object
  //!
  //! @param __kernel The kernel object
  explicit constexpr kernel_ref(value_type __kernel) noexcept
      : __kernel_(__kernel)
  {}

#if _CCCL_CTK_AT_LEAST(12, 1)
  //! @brief Constructs a `kernel_ref` from an entry function address
  //!
  //! @param __entry_func_address The entry function address
  //!
  //! @throws cuda_error if the kernel cannot be obtained from the entry function address
  kernel_ref(void (*__entry_func_address)(_Args...))
  {
    _CCCL_TRY_CUDA_API(
      ::cudaGetKernel, "Failed to get kernel from entry function address", &__kernel_, __entry_func_address);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 1)

  kernel_ref(const kernel_ref&) = default;

#if _CCCL_CTK_AT_LEAST(12, 3)
  //! @brief Get the mangled name of the kernel
  //!
  //! @return The mangled name of the kernel
  //!
  //! @throws cuda_error if the kernel name cannot be obtained
  [[nodiscard]] _CUDA_VSTD::string_view name() const
  {
    return __detail::driver::kernelGetName(__kernel_);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 3)

  //! @brief Get the maximum number of threads per block for the kernel
  //!
  //! @param __dev The device for which to query the maximum threads per block
  //!
  //! @return The maximum number of threads per block for the kernel on the specified device
  //!
  //! @throws cuda_error if the maximum threads per block cannot be obtained
  [[nodiscard]] unsigned max_threads_per_block(device_ref __dev) const
  {
    return static_cast<unsigned>(__get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, __dev));
  }

  //! @brief Get the size of statically allocated shared memory for the kernel
  //!
  //! @param __dev The device for which to query the shared memory size
  //!
  //! @return The size in bytes of statically allocated shared memory for the kernel on the specified device
  //!
  //! @throws cuda_error if the shared memory size cannot be obtained
  [[nodiscard]] _CUDA_VSTD::size_t static_shared_size(device_ref __dev) const
  {
    return static_cast<_CUDA_VSTD::size_t>(
      __get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, __dev));
  }

  //! @brief Get the size of user-allocated constant memory for the kernel
  //!
  //! @param __dev The device for which to query the constant memory size
  //!
  //! @return The size in bytes of user-allocated constant memory for the kernel on the specified device
  //!
  //! @throws cuda_error if the constant memory size cannot be obtained
  [[nodiscard]] _CUDA_VSTD::size_t const_size(device_ref __dev) const
  {
    return static_cast<_CUDA_VSTD::size_t>(
      __get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, __dev));
  }

  //! @brief Get the size of local memory used by each thread of the kernel
  //!
  //! @param __dev The device for which to query the local memory size
  //!
  //! @return The size in bytes of local memory used by each thread of the kernel on the specified device
  //!
  //! @throws cuda_error if the local memory size cannot be obtained
  [[nodiscard]] _CUDA_VSTD::size_t local_size(device_ref __dev) const
  {
    return static_cast<_CUDA_VSTD::size_t>(
      __get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, __dev));
  }

  //! @brief Get the number of registers used by each thread of the kernel
  //!
  //! @param __dev The device for which to query the number of registers
  //!
  //! @return The number of registers used by each thread of the kernel on the specified device
  //!
  //! @throws cuda_error if the number of registers cannot be obtained
  [[nodiscard]] _CUDA_VSTD::size_t num_regs(device_ref __dev) const
  {
    return static_cast<_CUDA_VSTD::size_t>(__get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS, __dev));
  }

  //! @brief Get the PTX version for which the kernel was compiled
  //!
  //! @param __dev The device for which to query the PTX version
  //!
  //! @return The PTX version for which the kernel was compiled on the specified device
  //!
  //! @throws cuda_error if the PTX version cannot be obtained
  [[nodiscard]] int ptx_version(device_ref __dev) const
  {
    return __get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION, __dev) * 10;
  }

  //! @brief Get the binary version for which the kernel was compiled
  //!
  //! @param __dev The device for which to query the binary version
  //!
  //! @return The binary version for which the kernel was compiled on the specified device
  //!
  //! @throws cuda_error if the binary version cannot be obtained
  [[nodiscard]] int binary_version(device_ref __dev) const
  {
    return __get_attrib(::CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION, __dev) * 10;
  }

  //! @brief Retrieve the native kernel handle
  //!
  //! @return The native kernel handle
  [[nodiscard]] constexpr value_type get() const noexcept
  {
    return __kernel_;
  }

  //! @brief Compares two `kernel_ref` for equality
  //!
  //! @param __lhs The first `kernel_ref` to compare
  //! @param __rhs The second `kernel_ref` to compare
  //! @return true if `lhs` and `rhs` refer to the same kernel
  [[nodiscard]] friend constexpr bool operator==(kernel_ref __lhs, kernel_ref __rhs) noexcept
  {
    return __lhs.__kernel_ == __rhs.__kernel_;
  }

  //! @brief Compares two `kernel_ref` for inequality
  //!
  //! @param __lhs The first `kernel_ref` to compare
  //! @param __rhs The second `kernel_ref` to compare
  //! @return true if `lhs` and `rhs` refer to a different kernels
  [[nodiscard]] friend constexpr bool operator!=(kernel_ref __lhs, kernel_ref __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

private:
  value_type __kernel_;
};

#if _CCCL_CTK_AT_LEAST(12, 1)
template <class... _Args>
kernel_ref(void (*)(_Args...)) -> kernel_ref<void(_Args...)>;
#endif // _CCCL_CTK_AT_LEAST(12, 1)

} // namespace cuda::experimental

#endif // _CUDAX___KERNEL_KERNEL_REF
