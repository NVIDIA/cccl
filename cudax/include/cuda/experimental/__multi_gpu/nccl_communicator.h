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

#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
//! @brief An owning wrapper around a NCCL communicator (`ncclComm_t`).
//!
//! This type takes ownership of the communicator passed to an inherited constructor.
//! It destroys the communicator when this object is destroyed. It cannot be copied, but
//! ownership is transferable by moving the object or by calling `release()`. NCCL errors
//! encountered while destroying the communicator are ignored.
//!
//! @snippet communicators/nccl/basic.cu nccl_communicator_construction
class _CCCL_VISIBILITY_DEFAULT nccl_communicator : public nccl_communicator_ref
{
public:
  using nccl_communicator_ref::nccl_communicator_ref;

  nccl_communicator(const nccl_communicator&)            = delete;
  nccl_communicator& operator=(const nccl_communicator&) = delete;

  //! @brief Move-construct an owning NCCL communicator.
  //!
  //! Transfers ownership of the native communicator from `__other`. After construction,
  //! `__other` does not own a native communicator.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_move_construction
  //!
  //! @param[in,out] __other The communicator whose ownership is transferred.
  //!
  //! @post `__other` does not own a native communicator.
  _CCCL_HOST_API nccl_communicator(nccl_communicator&& __other) noexcept
      : nccl_communicator_ref{__other.release(),
                              ::cuda::std::move(__other.__device_),
                              ::cuda::std::exchange(__other.__rank_, 0),
                              ::cuda::std::exchange(__other.__size_, 0)}
  {}

  //! @brief Transfer ownership from another NCCL communicator.
  //!
  //! Destroys the communicator currently owned by this object, if any. It then transfers
  //! ownership from `__other`. Errors while destroying the previously owned communicator are
  //! ignored.
  //!
  //! @snippet communicators/nccl/basic.cu nccl_communicator_move_assignment
  //!
  //! @param[in,out] __other The communicator whose ownership is transferred.
  //!
  //! @return A reference to this communicator.
  //!
  //! @post `__other` does not own a native communicator.
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
  //! If this object owns a native communicator, it is destroyed. Any NCCL error encountered
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
  //! @post This object does not own a native communicator.
  [[nodiscard]] _CCCL_HOST_API constexpr native_handle_type release() noexcept
  {
    return ::cuda::std::exchange(__comm_, nullptr);
  }

private:
  _CCCL_HOST_API void __reset() noexcept
  {
    if (__comm_)
    {
      static_cast<void>(::cuda::experimental::__nccl::__ncclCommDestroyNoThrow(release()));
    }
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_COMMUNICATOR_H
