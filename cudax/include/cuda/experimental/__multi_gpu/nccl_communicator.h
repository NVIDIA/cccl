//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/no_init.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>
#include <cuda/experimental/__nccl/nccl_api.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
//! @brief An owning wrapper around a NCCL communicator (`ncclComm_t`).
//!
//! This type takes ownership of the communicator supplied. It destroys the communicator when
//! this object is destroyed. It cannot be copied, but ownership is transferable by moving the
//! object or by calling `release()`.
//!
//! @snippet communicators/nccl/basic.cu nccl_communicator_construction
class _CCCL_VISIBILITY_DEFAULT nccl_communicator : public nccl_communicator_ref
{
public:
  //! @brief Disallow direct construction from `NCCL_COMM_NULL`.
  //!
  //! Use the `nccl_communicator(cuda::no_init_t)` constructor instead.
  static nccl_communicator from_native_handle(::cuda::std::nullptr_t) = delete;

  //! @brief Construct an owning communicator from an existing NCCL communicator handle.
  //!
  //! The returned object takes ownership of `__handle` and destroys it when the object is
  //! destroyed. The logical device, rank, and size are retrieved from NCCL.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_construction
  //!
  //! @param __handle The NCCL communicator handle to take ownership of.
  //!
  //! @return An owning communicator that wraps `__handle`.
  //!
  //! @throws std::invalid_argument If `__handle` is `NCCL_COMM_NULL`.
  [[nodiscard]] static _CCCL_HOST_API nccl_communicator from_native_handle(native_handle_type __handle)
  {
    return nccl_communicator{__handle};
  }

  //! @brief Construct an owning communicator from a NCCL handle and a logical device.
  //!
  //! The returned object takes ownership of `__handle` and destroys it when the object is
  //! destroyed. The NCCL communicator must be associated with `__device`.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_construction_with_logical_device
  //!
  //! @param __handle The NCCL communicator handle to take ownership of.
  //! @param __device The logical device the communicator is expected to be associated with.
  //!
  //! @return An owning communicator that wraps `__handle`.
  //!
  //! @throws std::invalid_argument If `__handle` is `NCCL_COMM_NULL`.
  //! @throws std::runtime_error If the device reported by NCCL does not match `__device`.
  [[nodiscard]] static _CCCL_HOST_API nccl_communicator
  from_native_handle(native_handle_type __handle, ::cuda::experimental::logical_device __device)
  {
    return nccl_communicator{__handle, ::cuda::std::move(__device)};
  }

  //! @brief Construct a communicator with no native NCCL communicator.
  //!
  //! The constructed object owns no communicator. It cannot be used to make communication
  //! calls. The only guarantee is that the returned handle is `NCCL_COMM_NULL`; the rank,
  //! size, and logical device are implementation defined.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_no_init_construction
  _CCCL_HOST_API explicit nccl_communicator(::cuda::no_init_t) noexcept
      : nccl_communicator_ref{::cuda::experimental::__nccl::__NCCL_COMM_NULL,
                              ::cuda::experimental::logical_device{0},
                              /*__rank=*/0,
                              /*__size=*/0}
  {}

  nccl_communicator(const nccl_communicator&)            = delete;
  nccl_communicator& operator=(const nccl_communicator&) = delete;

  //! @brief Move-construct an owning NCCL communicator.
  //!
  //! Transfers ownership of the native communicator from `__other`. After construction,
  //! `__other` does not own a communicator.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_move_construction
  //!
  //! @param[in,out] __other The communicator whose ownership is transferred.
  //!
  //! @post `__other` does not own a communicator.
  _CCCL_HOST_API nccl_communicator(nccl_communicator&& __other) noexcept
      : nccl_communicator_ref{__other.release(),
                              ::cuda::std::move(__other.__device_),
                              ::cuda::std::exchange(__other.__rank_, 0),
                              ::cuda::std::exchange(__other.__size_, 0)}
  {}

  //! @brief Transfer ownership from another NCCL communicator.
  //!
  //! Destroys the communicator currently owned by this object, if any. It then transfers
  //! ownership from `__other`.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_move_assignment
  //!
  //! @param[in,out] __other The communicator whose ownership is transferred.
  //!
  //! @return A reference to this communicator.
  //!
  //! @post `__other` does not own a communicator.
  _CCCL_HOST_API nccl_communicator& operator=(nccl_communicator&& __other) noexcept
  {
    if (this != &__other)
    {
      __reset();
      __comm_   = __other.release();
      __device_ = ::cuda::std::move(__other.__device_);
      __rank_   = ::cuda::std::exchange(__other.__rank_, 0);
      __size_   = ::cuda::std::exchange(__other.__size_, 0);
    }
    return *this;
  }

  //! @brief Destroy the owned NCCL communicator.
  //!
  //! If this object owns a communicator, it is destroyed. Any NCCL error encountered
  //! during destruction is ignored.
  _CCCL_HOST_API ~nccl_communicator()
  {
    __reset();
  }

  //! @brief Relinquish ownership of the native NCCL communicator.
  //!
  //! The returned communicator is not destroyed by this object. The caller becomes responsible
  //! for destroying it.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_release
  //!
  //! @return The native NCCL communicator previously owned by this object.
  //!
  //! @post This object does not own a communicator.
  [[nodiscard]] _CCCL_HOST_API constexpr native_handle_type release() noexcept
  {
    return ::cuda::std::exchange(__comm_, ::cuda::experimental::__nccl::__NCCL_COMM_NULL);
  }

private:
  _CCCL_HOST_API explicit nccl_communicator(native_handle_type __handle)
      : nccl_communicator_ref{__handle}
  {}

  _CCCL_HOST_API explicit nccl_communicator(native_handle_type __handle, ::cuda::experimental::logical_device __device)
      : nccl_communicator_ref{__handle, ::cuda::std::move(__device)}
  {}

  _CCCL_HOST_API void __reset() noexcept
  {
    if (__comm_ != ::cuda::experimental::__nccl::__NCCL_COMM_NULL)
    {
      static_cast<void>(::cuda::experimental::__nccl::__ncclCommDestroyNoThrow(release()));
    }
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H
