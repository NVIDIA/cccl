//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MEMORY_SHARED_MEMORY_PTR_H
#define _CUDA_EXPERIMENTAL___MEMORY_SHARED_MEMORY_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/address_space.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/experimental/__memory/shared_memory_ptr.cuh>

namespace cuda::experimental
{
enum class __smem_addr_t : unsigned
{
};

//! @brief A pointer to shared memory.
template <class _Tp>
class shared_memory_ptr
{
  unsigned __smem_addr_; //!< The address of the shared memory.

public:
  using element_type = _Tp; //!< The element type.
  using pointer      = _Tp*; //!< The pointer type.

  shared_memory_ptr() = delete;

  shared_memory_ptr(::cuda::std::nullptr_t) = delete;

  //! @brief Constructs the object to uninitialized state.
  _CCCL_DEVICE_API explicit shared_memory_ptr(::cuda::no_init_t) noexcept {}

  //! @brief Constructs the object from shared memory address.
  //!
  //! @param __addr The shared memory address.
  _CCCL_DEVICE_API explicit shared_memory_ptr(__smem_addr_t __addr) noexcept
      : __smem_addr_{static_cast<unsigned>(__addr)}
  {}

  //! @brief Constructs the object from shared memory pointer.
  //!
  //! @param __ptr The shared memory pointer.
  _CCCL_DEVICE_API explicit shared_memory_ptr(_Tp* __ptr) noexcept
  {
    reset(__ptr);
  }

  //! @brief Constructs the object from another \c shared_memory_ptr with different element type such that the other
  //!        \c pointer_type is convertible to this \c pointer_type.
  //!
  //! @param __other The other \c shared_memory_ptr.
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(::cuda::std::is_convertible_v<_Up*, _Tp*>)
  _CCCL_DEVICE_API shared_memory_ptr(shared_memory_ptr<_Up> __other) noexcept
      : __smem_addr_{__other.__smem_addr_}
  {}

  _CCCL_HIDE_FROM_ABI shared_memory_ptr(const shared_memory_ptr&) noexcept = default;

  _CCCL_HIDE_FROM_ABI shared_memory_ptr(shared_memory_ptr&&) noexcept = default;

  _CCCL_HIDE_FROM_ABI shared_memory_ptr& operator=(const shared_memory_ptr&) noexcept = default;

  //! @brief Resets the pointer to the given pointer.
  //!
  //! @param __ptr The pointer to reset to.
  //!
  //! @returns The previous pointer.
  _CCCL_DEVICE_API _Tp* reset(_Tp* __ptr) noexcept
  {
    _CCCL_ASSERT(::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::shared),
                 "pointer is not from shared memory");
    _Tp* __ret   = get();
    __smem_addr_ = ::__cvta_generic_to_shared(__ptr);
    return __ret;
  }

  //! @brief Swaps the pointers of two shared_memory_ptrs.
  _CCCL_DEVICE_API constexpr void swap(shared_memory_ptr& __other) noexcept
  {
    ::cuda::std::swap(__smem_addr_, __other.__smem_addr_);
  }

  //! @brief Gets the stored address.
  //!
  //! @returns The stored address.
  [[nodiscard]] _CCCL_DEVICE_API __smem_addr_t __get_smem_addr() const noexcept
  {
    return __smem_addr_t{__smem_addr_};
  }

  //! @brief Gets the stored pointer.
  //!
  //! @returns The pointer.
  [[nodiscard]] _CCCL_DEVICE_API _Tp* get() const noexcept
  {
    return static_cast<_Tp*>(::__cvta_shared_to_generic(__smem_addr_));
  }

  //! @brief Conversion operator to bool (always returns \c true).
  _CCCL_DEVICE_API explicit constexpr operator bool() const noexcept
  {
    return true;
  }

  //! @brief Arrow operator.
  //!
  //! @return The stored pointer.
  [[nodiscard]] _CCCL_DEVICE_API _Tp* operator->() const noexcept
  {
    return get();
  }

  //! @brief Dereference operator.
  //!
  //! @return Reference to the object pointed to by the stored pointer.
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES((!::cuda::std::is_void_v<_Tp2>) )
  [[nodiscard]] _CCCL_DEVICE_API _Tp2& operator*() const noexcept
  {
    return *get();
  }

  //! @brief Conversion operator to \c pointer_type.
  //!
  //! @returns The stored pointer.
  _CCCL_DEVICE_API explicit operator _Tp*() const noexcept
  {
    return get();
  }

  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator==(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ == __rhs.__smem_addr_;
  }
  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator!=(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ != __rhs.__smem_addr_;
  }
  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator<(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ < __rhs.__smem_addr_;
  }
  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator<=(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ <= __rhs.__smem_addr_;
  }
  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator>(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ > __rhs.__smem_addr_;
  }
  template <class _Rhs>
  [[nodiscard]] _CCCL_DEVICE_API friend bool
  operator>=(const shared_memory_ptr& __lhs, const shared_memory_ptr<_Rhs>& __rhs) noexcept
  {
    return __lhs.__smem_addr_ >= __rhs.__smem_addr_;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator==(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator!=(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator<(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator<=(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator>(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator>=(shared_memory_ptr, ::cuda::std::nullptr_t) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator==(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator!=(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator<(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator<=(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return true;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator>(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return false;
  }
  [[nodiscard]] _CCCL_DEVICE_API friend bool operator>=(::cuda::std::nullptr_t, shared_memory_ptr) noexcept
  {
    return false;
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE shared_memory_ptr(_Tp*) -> shared_memory_ptr<_Tp>;

// todo: constraints
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<_Tp> static_pointer_cast(shared_memory_ptr<_Up> __ptr) noexcept
{
  return shared_memory_ptr<_Tp>{__ptr.__get_smem_addr()};
}

// todo: constraints
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<_Tp> const_pointer_cast(shared_memory_ptr<_Up> __ptr) noexcept
{
  return shared_memory_ptr<_Tp>{__ptr.__get_smem_addr()};
}

// todo: constraints
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_DEVICE_API shared_memory_ptr<_Tp> reinterpret_pointer_cast(shared_memory_ptr<_Up> __ptr) noexcept
{
  return shared_memory_ptr<_Tp>{__ptr.__get_smem_addr()};
}
} // namespace cuda::experimental

#endif // _CUDA_EXPERIMENTAL___MEMORY_SHARED_MEMORY_PTR_H
