//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/__device/device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__memory/address_space.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/string_view>

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
class kernel_ref
{
  static_assert(::cuda::std::__always_false_v<_Signature>,
                "kernel_ref must have a signature of the form `void(Args...)`");
};

template <class... _Args>
class kernel_ref<void(_Args...)>
{
  static_assert((true && ... && ::cuda::std::is_trivially_copyable_v<_Args>),
                "All kernel_ref argument types must be trivially copyable.");

public:
#if _CCCL_CTK_BELOW(12, 1)
  using value_type = ::CUkernel;
#else // ^^^ _CCCL_CTK_BELOW(12, 1) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 1) vvv
  using value_type = ::cudaKernel_t;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 1) ^^^

  kernel_ref(::cuda::std::nullptr_t) = delete;

  //! @brief Constructs a `kernel_ref` from a kernel object
  //!
  //! @param __kernel The kernel object
  explicit constexpr kernel_ref(value_type __kernel) noexcept
      : __kernel_((::CUkernel) __kernel)
  {}

#if _CCCL_CTK_AT_LEAST(12, 1)
  //! @brief Constructs a `kernel_ref` from an entry function address
  //!
  //! @param __entry_func_address The entry function address
  //!
  //! @throws cuda_error if the kernel cannot be obtained from the entry function address
  kernel_ref(void (*__entry_func_address)(_Args...))
  {
    _CCCL_TRY_CUDA_API(::cudaGetKernel,
                       "Failed to get kernel from entry function address",
                       (cudaKernel_t*) &__kernel_,
                       (const void*) __entry_func_address);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 1)

  kernel_ref(const kernel_ref&) = default;

#if _CCCL_CTK_AT_LEAST(12, 3)
  //! @brief Get the mangled name of the kernel
  //!
  //! @return The mangled name of the kernel
  //!
  //! @throws cuda_error if the kernel name cannot be obtained
  [[nodiscard]] ::cuda::std::string_view name() const
  {
    return ::cuda::__driver::__kernelGetName(__kernel_);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 3)

  //! @brief Retrieve the specified attribute for the kernel on the specified device
  //!
  //! @param __attr The attribute to query. See `kernel::attributes` for the available
  //!        attributes.
  //! @param __dev The device for which to query the attribute
  //!
  //! @throws cuda_error if the attribute query fails
  //!
  //! @sa kernel::attributes
  template <typename _Attr>
  [[nodiscard]] auto attribute(_Attr __attr, device_ref __dev) const
  {
    return __attr(*this, __dev);
  }

  //! @brief Retrieve the native kernel handle
  //!
  //! @return The native kernel handle
  [[nodiscard]] constexpr value_type get() const noexcept
  {
    return (value_type) __kernel_;
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
  ::CUkernel __kernel_;
};

#if _CCCL_CTK_AT_LEAST(12, 1)
template <class... _Args>
kernel_ref(void (*)(_Args...)) -> kernel_ref<void(_Args...)>;
#endif // _CCCL_CTK_AT_LEAST(12, 1)

namespace __detail
{
template <class _Tp>
inline constexpr bool __is_kernel_ref_v = false;
template <class _Tp>
inline constexpr bool __is_kernel_ref_v<const _Tp> = __is_kernel_ref_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_kernel_ref_v<volatile _Tp> = __is_kernel_ref_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_kernel_ref_v<const volatile _Tp> = __is_kernel_ref_v<_Tp>;
template <class... _Signature>
inline constexpr bool __is_kernel_ref_v<kernel_ref<_Signature...>> = true;
} // namespace __detail
} // namespace cuda::experimental

#endif // _CUDAX___KERNEL_KERNEL_REF
