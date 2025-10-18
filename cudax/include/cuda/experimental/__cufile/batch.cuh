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

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cccl/memory_wrapper.h>
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__memory/uninitialized_algorithms.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstddef>
#include <cuda/std/optional>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/exception.cuh>

#include <vector>

#include <cufile.h>

namespace cuda::experimental
{

//! @brief Enum for the cuFile batch request state.
enum class cufile_batch_request_state
{
  waiting, //!< Job hasn't been submitted yet.
  pending, //!< Job was enqueued.
  invalid, //!< The request was ill-formed or could not be enqueued.
  canceled, //!< The request was successfully canceled.
  complete, //!< The request was successfully completed.
  timeout, //!< The request timed out.
  failed, //!< The request was unable to complete.
};

//! @brief Class wrapping the cuFile batch request state operations.
class cufile_batch_request
{
  friend class cufile_batch;

  ::CUfileIOEvents_t __io_events_;

  _CCCL_HIDE_FROM_ABI cufile_batch_request() noexcept = default;

public:
  cufile_batch_request(const cufile_batch_request&)            = delete;
  cufile_batch_request(cufile_batch_request&&)                 = delete;
  cufile_batch_request& operator=(const cufile_batch_request&) = delete;
  cufile_batch_request& operator=(cufile_batch_request&&)      = delete;

  //! @brief Gets the request state.
  //!
  //! @return The request state.
  [[nodiscard]] _CCCL_API cufile_batch_request_state state() const noexcept
  {
    switch (__io_events_.status)
    {
      case ::CUFILE_WAITING:
        return cufile_batch_request_state::waiting;
      case ::CUFILE_PENDING:
        return cufile_batch_request_state::pending;
      case ::CUFILE_INVALID:
        return cufile_batch_request_state::invalid;
      case ::CUFILE_CANCELED:
        return cufile_batch_request_state::canceled;
      case ::CUFILE_COMPLETE:
        return cufile_batch_request_state::complete;
      case ::CUFILE_TIMEOUT:
        return cufile_batch_request_state::timeout;
      case ::CUFILE_FAILED:
        return cufile_batch_request_state::failed;
      default:
        _CCCL_UNREACHABLE();
    }
  }

  //! @brief Gets the number of transferred bytes by the request, written or read.
  //!
  //! @return The number of transferred bytes. If the state is not \c cuda::cufile_batch_request_state::complete, an
  //!         empty optional is returned.
  [[nodiscard]] _CCCL_API ::cuda::std::optional<::cuda::std::size_t> transferred_nbytes() const noexcept
  {
    return (state() == cufile_batch_request_state::complete)
           ? ::cuda::std::optional{__io_events_.ret}
           : cuda::std::nullopt;
  }
};

//! @brief Type wrapping the cuFile batch query operations.
class cufile_batch
{
  friend class cufile_batch_requests;

  ::CUfileBatchHandle_t __batch_{};
  unsigned __nrequests_{};
  ::std::unique_ptr<cufile_batch_request[]> __requests_{};

  _CCCL_HOST_API cufile_batch(::CUfileBatchHandle_t __batch, unsigned __nrequests) noexcept
      : __batch_{__batch}
      , __nrequests_{__nrequests}
      , __requests_{new cufile_batch_request[__nrequests]}
  {}

  //! @brief Checks whether the object is valid.
  [[nodiscard]] _CCCL_HOST_API bool __valid() const noexcept
  {
    return __batch_ != nullptr;
  }

  //! @brief Updates the tracked requests.
  //!
  //! @return The number of active requests.
  [[nodiscard]] _CCCL_HOST_API unsigned __update_requests(::timespec* __timeout = nullptr) const
  {
    if (!__valid())
    {
      return 0;
    }

    auto __nr = __nrequests_;
    _CCCL_TRY_CUFILE_API(
      ::cuFileBatchIOGetStatus,
      "Failed to query IO batch status.",
      __batch_,
      __nr,
      &__nr,
      reinterpret_cast<::CUfileIOEvents_t*>(__requests_.get()),
      __timeout);
    return __nr;
  }

public:
  //! @brief Default constructor.
  _CCCL_HIDE_FROM_ABI cufile_batch() noexcept = default;

  cufile_batch(const cufile_batch&) = delete;

  //! @brief Move constructor.
  _CCCL_HOST_API cufile_batch(cufile_batch&& __other) noexcept
      : __batch_{::cuda::std::exchange(__other.__batch_, nullptr)}
      , __nrequests_{::cuda::std::exchange(__other.__nrequests_, 0)}
      , __requests_{::cuda::std::move(__other.__requests_)}
  {}

  cufile_batch& operator=(const cufile_batch&) = delete;

  //! @brief Move assignment operator.
  cufile_batch& operator=(cufile_batch&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      if (__valid())
      {
        ::cuFileBatchIODestroy(__batch_);
        __requests_.reset();
      }
      __batch_     = ::cuda::std::exchange(__other.__batch_, nullptr);
      __nrequests_ = ::cuda::std::exchange(__other.__nrequests_, 0);
      __requests_  = ::cuda::std::move(__other.__requests_);
    }
    return *this;
  }

  //! @brief Destructor.
  _CCCL_HOST_API ~cufile_batch()
  {
    if (__valid())
    {
      ::cuFileBatchIODestroy(__batch_);
    }
  }

  //! @brief Cancels all tracked requests.
  _CCCL_HOST_API void cancel() const
  {
    if (__valid())
    {
      _CCCL_TRY_CUFILE_API(::cuFileBatchIOCancel, "Failed to cancel IO batch.", __batch_);
    }
  }

  //! @brief Checks if all of the tracked requests are done.
  //!
  //! @return \c true if done, \c false otherwise.
  [[nodiscard]] _CCCL_HOST_API bool is_done() const
  {
    return __update_requests() == 0;
  }

  //! @brief Gets the span of the tracked requests.
  //!
  //! @return The span of tracked requests.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<const cufile_batch_request> requests() const noexcept
  {
    __update_requests();
    return ::cuda::std::span{__requests_.get(), static_cast<::cuda::std::size_t>(__nrequests_)};
  }
};

//! @brief Class wrapping the setup of the \c cuda::cufile_batch.
class cufile_batch_requests
{
  ::std::vector<::CUfileIOParams_t> __io_params_{};

public:
  //! @brief Adds a read request.
  //!
  //! todo: params
  _CCCL_API void add_read_bytes(
    cufile_ref __file,
    cufile_ref::off_type __foffset,
    ::cuda::std::byte* __dst,
    ::cuda::std::size_t __nbytes,
    ::cuda::std::ptrdiff_t __doffset = 0)
  {
    ::CUfileIOParams_t& __new   = __io_params_.emplace_back();
    __new.mode                  = ::CUFILE_BATCH;
    __new.u.batch.devPtr_base   = __dst;
    __new.u.batch.file_offset   = __foffset;
    __new.u.batch.devPtr_offset = __doffset;
    __new.u.batch.size          = __nbytes;
    __new.fh                    = __file.get();
    __new.opcode                = ::CUFILE_READ;
    // todo: add cookie?
  }

  // todo: add other read/write overloads

  //! @brief Clears the stored requests.
  _CCCL_HOST_API void clear() noexcept
  {
    __io_params_.clear();
  }

  //! @brief Gets the request count.
  //!
  //! @return The number of requests.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t size() const noexcept
  {
    return __io_params_.size();
  }

  //! @brief Gets the request count.
  //!
  //! @return The number of requests.
  [[nodiscard]] _CCCL_HOST_API bool empty() const noexcept
  {
    return __io_params_.empty();
  }

  //! @brief Commit the stored requests.
  //!
  //! @return The \c cuda::cufile_batch object.
  [[nodiscard]] _CCCL_API cufile_batch commit()
  {
    if (__io_params_.empty())
    {
      return cufile_batch{};
    }

    const auto __nr = static_cast<unsigned>(__io_params_.size());

    ::CUfileBatchHandle_t __batch{};
    _CCCL_TRY_CUFILE_API(::cuFileBatchIOSetUp, "Failed to setup IO batch", &__batch, __nr);

    try
    {
      _CCCL_TRY_CUFILE_API(::cuFileBatchIOSubmit, "Failed to submit IO batch.", __batch, __nr, __io_params_.data(), 0);
      return cufile_batch{__batch, __nr};
    }
    catch (...)
    {
      ::cuFileBatchIODestroy(__batch);
      throw;
    }
  }
};

} // namespace cuda::experimental
