//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/std/__cccl/memory_wrapper.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/atomic>

#include <cufile.h>

namespace cuda::experimental
{

//! @brief The \c cuda::at_submission placeholder type.
enum class cufile_at_submission_t
{
};

//! @brief Placeholder for cuFile parameter to be specified at submission time, rather than enqueue time.
inline constexpr cufile_at_submission_t at_submission;

//! @brief Trait that checks if \c _Tp is the \c cufile_at_submission_t type or the \c _Up type.
template <class _Tp, class _Up>
inline constexpr bool __is_at_submission_or =
  ::cuda::std::is_same_v<_Tp, cufile_at_submission_t> || ::cuda::std::is_convertible_v<_Tp, _Up>;

//! @brief Class to hold the \c cufile_enqueue_result data.
class __cufile_enqueue_result_data
{
  int __nrefs_{2}; //!< Reference count.

public:
  //! @brief Atomically decreases the reference count and deletes the object if the reference count is less than or
  //!        equal to 1.
  static _CCCL_HOST_API void __decrease_ref_count(__cufile_enqueue_result_data* __data) noexcept
  {
    if (__data == nullptr)
    {
      return;
    }

    const auto __old_nrefs = cuda::std::atomic_ref{__data->__nrefs_}.fetch_sub(1, ::cuda::std::memory_order_relaxed);
    if (__old_nrefs <= 1)
    {
      delete __data;
    }
  }

  //! @brief The file offset.
  ::off_t __foffset_{::cuda::std::numeric_limits<::off_t>::min()};

  //! @brief The number of bytes to transfer.
  ::cuda::std::size_t __nbytes_{::cuda::std::numeric_limits<::cuda::std::size_t>::max()};

  //! @brief The buffer offset.
  ::off_t __doffset_{::cuda::std::numeric_limits<::off_t>::min()};

  //! @brief The async cuFile read/write call return value.
  ::ssize_t __result_{};

  //! @brief The pointer to a pinned memory where the read/write call stores the return value. Externally managed.
  ::CUdeviceptr __pinned_result_ptr_{};
};

//! @brief The type returned from cuFile async read/write operations.
template <class _FOffset, class _NBytes, class _DOffset>
class cufile_enqueue_result
{
  __cufile_enqueue_result_data* __data_{}; //!< The data.

  //! @brief Constructor from \c __cufile_enqueue_result_data pointer.
  //!
  //! @pre The \c __cufile_enqueue_result_data object must be dynamically allocated using operator \c new.
  _CCCL_HIDE_FROM_ABI cufile_io_submit_result(__cufile_enqueue_result_data* __data) noexcept
      : __data_{__data}
  {}

public:
  //! @brief Makes the instance from allocated data.
  [[nodiscard]] static _CCCL_HOST_API cufile_enqueue_result
  __make_instance(__cufile_enqueue_result_data* __data) noexcept
  {
    return cufile_at_submission_t{__data};
  }

  cufile_enqueue_result(const cufile_enqueue_result&) = delete;

  //! @brief Move constructor.
  cufile_enqueue_result(cufile_enqueue_result&& __other) noexcept
      : __data_{::cuda::std::exchange(__other.__data_, nullptr)}
  {}

  cufile_enqueue_result& operator=(const cufile_enqueue_result&) = delete;

  //! @brief Move assignment operator.
  cufile_enqueue_result& operator=(cufile_enqueue_result&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      __cufile_enqueue_result_data::__decrease_ref_count(__data_);
      __data_ = ::cuda::std::exchange(__other.__data_, nullptr);
    }
    return *this;
  }

  //! @brief Destructor.
  _CCCL_HOST_API ~cufile_enqueue_result()
  {
    __cufile_enqueue_result_data::__decrease_ref_count(__data_);
  }

  //! @brief Sets the file offset. Only allowed if \c cuda::at_submission was passed as the file offset argument to the
  //!        IO call.
  //!
  //! @param __offset The file offset to be used.
  _CCCL_TEMPLATE(class _FOffset2 = _FOffset)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_FOffset2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_file_offset(cufile_ref::off_type __offset) const noexcept
  {
    __data_->__foffset_ = __offset;
  }

  //! @brief Sets the IO operation size in bytes. Only allowed if \c cuda::at_submission was passed as the nbytes
  //!        argument to the IO call.
  //!
  //! @param __nbytes The number of bytes to be read to be used.
  _CCCL_TEMPLATE(class _NBytes2 = _NBytes)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_NBytes2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_size(::cuda::std::size_t __nbytes) const noexcept
  {
    __data_->__nbytes_ = __nbytes;
  }

  //! @brief Sets the buffer offset. Only allowed if \c cuda::at_submission was passed as the buffer offset argument to
  //! the IO call.
  //!
  //! @param __offset The buffer offset to be used.
  _CCCL_TEMPLATE(class _DOffset2 = _DOffset)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_DOffset2, cufile_at_submission_t>)
  _CCCL_HOST_API void set_buffer_offset(::cuda::std::ptrdiff_t __offset) const noexcept
  {
    __data_->__doffset_ = __offset;
  }

  //! @brief Get the number of bytes transferred by the asynchronous IO operation.
  //!
  //! @return The number of bytes transferred.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t transferred_nbytes() const noexcept
  {
    // todo: check error
    return static_cast<::cuda::std::size_t>(__data_->__result);
  }
};

//! @brief The \c cufile_enqueue_result with the default types.
using cufile_enqueue_result_default =
  cufile_enqueue_result<cufile_ref::off_type, ::cuda::std::size_t, ::cuda::std::ptrdiff_t>;

#if _CCCL_CUDA_COMPILATION()
//! @brief The callback
_CCCL_DEVICE_API void __cufile_post_async_io_callback(::CUstream, ::CUresult, void* __data_void_ptr) noexcept
{
  const auto __data = static_cast<__cufile_enqueue_result_data*>(__data_void_ptr);
  __data->__result_ = *reinterpret_cast<::ssize_t*>(__data->__pinned_result_ptr_);
  __cufile_enqueue_result_data::__decrease_ref_count(__data);
}
#endif // _CCCL_CUDA_COMPILATION()

} // namespace cuda::experimental
