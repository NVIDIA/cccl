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
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstddef>
#include <cuda/std/ctime>
#include <cuda/std/optional>

#include <cuda/experimental/__cufile/cufile_ref.cuh>
#include <cuda/experimental/__cufile/exception.cuh>

#include <vector>

#include <cufile.h>

namespace cuda::experimental
{

//! @brief Class wrapping the cuFile batch request.
class cufile_batch_request
{
  friend class cufile_batch;

  ::CUfileIOParams_t __data_;

  _CCCL_HIDE_FROM_ABI cufile_batch_request() noexcept = default;

public:
  _CCCL_HIDE_FROM_ABI cufile_batch_request(const cufile_batch_request&)            = default;
  _CCCL_HIDE_FROM_ABI cufile_batch_request(cufile_batch_request&&)                 = default;
  _CCCL_HIDE_FROM_ABI cufile_batch_request& operator=(const cufile_batch_request&) = default;
  _CCCL_HIDE_FROM_ABI cufile_batch_request& operator=(cufile_batch_request&&)      = default;
};

//! @brief Enum for the cuFile batch job state.
enum class cufile_batch_job_state
{
  waiting, //!< Job hasn't been submitted yet.
  pending, //!< Job was enqueued.
  invalid, //!< The request was ill-formed or could not be enqueued.
  canceled, //!< The job was successfully canceled.
  complete, //!< The job was successfully completed.
  timeout, //!< The job timed out.
  failed, //!< The job was unable to complete.
};

//! @brief Class wrapping the cuFile batch job state operations.
class cufile_batch_job
{
  friend class cufile_batch;

  ::CUfileIOEvents_t __io_events_;

  _CCCL_HIDE_FROM_ABI cufile_batch_job() noexcept = default;

public:
  cufile_batch_job(const cufile_batch_job&)            = delete;
  cufile_batch_job(cufile_batch_job&&)                 = delete;
  cufile_batch_job& operator=(const cufile_batch_job&) = delete;
  cufile_batch_job& operator=(cufile_batch_job&&)      = delete;

  //! @brief Gets the job state.
  //!
  //! @return The job state.
  [[nodiscard]] _CCCL_HOST_API cufile_batch_job_state state() const noexcept
  {
    switch (__io_events_.status)
    {
      case ::CUFILE_WAITING:
        return cufile_batch_job_state::waiting;
      case ::CUFILE_PENDING:
        return cufile_batch_job_state::pending;
      case ::CUFILE_INVALID:
        return cufile_batch_job_state::invalid;
      case ::CUFILE_CANCELED:
        return cufile_batch_job_state::canceled;
      case ::CUFILE_COMPLETE:
        return cufile_batch_job_state::complete;
      case ::CUFILE_TIMEOUT:
        return cufile_batch_job_state::timeout;
      case ::CUFILE_FAILED:
        return cufile_batch_job_state::failed;
      default:
        _CCCL_UNREACHABLE();
    }
  }

  //! @brief Gets the number of transferred bytes by the job, written or read.
  //!
  //! @return The number of transferred bytes. If the state is not \c cuda::cufile_batch_job_state::complete, an
  //!         empty optional is returned.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::optional<::cuda::std::size_t> transferred_nbytes() const noexcept
  {
    return (state() == cufile_batch_job_state::complete)
           ? ::cuda::std::optional{__io_events_.ret}
           : ::cuda::std::nullopt;
  }
};

//! @brief Type wrapping the cuFile batch query operations.
class cufile_batch
{
  ::CUfileBatchHandle_t __batch_{};
  unsigned __njobs_{};
  ::std::unique_ptr<cufile_batch_job[]> __jobs_{};

  _CCCL_HOST_API cufile_batch(::CUfileBatchHandle_t __batch, unsigned __nrequests)
      : __batch_{__batch}
      , __njobs_{__nrequests}
      , __jobs_{new cufile_batch_job[__nrequests]}
  {}

  //! @brief Checks whether the object is valid.
  [[nodiscard]] _CCCL_HOST_API bool __valid() const noexcept
  {
    return __batch_ != nullptr;
  }

  //! @brief Updates the tracked requests.
  //!
  //! @return The number of active requests.
  [[nodiscard]] _CCCL_HOST_API unsigned __update_jobs(::timespec* __timeout = nullptr)
  {
    if (!__valid())
    {
      return 0;
    }

    auto __nr = __njobs_;
    _CCCL_TRY_CUFILE_API(
      ::cuFileBatchIOGetStatus,
      "Failed to query IO batch status.",
      __batch_,
      __nr,
      &__nr,
      &__jobs_.get()->__io_events_,
      __timeout);
    return __nr;
  }

public:
  //! @brief Makes a read request.
  //!
  //! todo: params
  [[nodiscard]] static _CCCL_HOST_API cufile_batch_request make_read_bytes_request(
    cufile_ref __file,
    cufile_ref::off_type __foffset,
    ::cuda::std::byte* __dst,
    ::cuda::std::size_t __nbytes,
    ::cuda::std::ptrdiff_t __doffset = 0,
    void* __cookie                   = nullptr) noexcept
  {
    cufile_batch_request __ret{};
    __ret.__data_.mode                  = ::CUFILE_BATCH;
    __ret.__data_.u.batch.devPtr_base   = __dst;
    __ret.__data_.u.batch.file_offset   = __foffset;
    __ret.__data_.u.batch.devPtr_offset = __doffset;
    __ret.__data_.u.batch.size          = __nbytes;
    __ret.__data_.fh                    = __file.get();
    __ret.__data_.opcode                = ::CUFILE_READ;
    __ret.__data_.cookie                = __cookie;
    return __ret;
  }

  // todo: add other ops

  //! @brief Commits a contiguous range of batch requests.
  //!
  //! @param __requests A contiguous range of batch requests.
  //!
  //! @return The new batch.
  [[nodiscard]] static _CCCL_HOST_API cufile_batch commit(::cuda::std::span<const cufile_batch_request> __requests)
  {
    if (__requests.empty())
    {
      return cufile_batch{};
    }

    const auto __nr = static_cast<unsigned>(__requests.size());

    ::CUfileBatchHandle_t __batch{};
    _CCCL_TRY_CUFILE_API(::cuFileBatchIOSetUp, "Failed to setup IO batch", &__batch, __nr);

    try
    {
      _CCCL_TRY_CUFILE_API(
        ::cuFileBatchIOSubmit,
        "Failed to submit IO batch.",
        __batch,
        __nr,
        const_cast<::CUfileIOParams_t*>(&__requests.data()->__data_),
        0);
      return cufile_batch{__batch, __nr};
    }
    catch (...)
    {
      ::cuFileBatchIODestroy(__batch);
      throw;
    }
  }

  //! @brief Default constructor.
  _CCCL_HIDE_FROM_ABI cufile_batch() noexcept = default;

  cufile_batch(const cufile_batch&) = delete;

  //! @brief Move constructor.
  _CCCL_HOST_API cufile_batch(cufile_batch&& __other) noexcept
      : __batch_{::cuda::std::exchange(__other.__batch_, nullptr)}
      , __njobs_{::cuda::std::exchange(__other.__njobs_, 0)}
      , __jobs_{::cuda::std::move(__other.__jobs_)}
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
        __jobs_.reset();
      }
      __batch_ = ::cuda::std::exchange(__other.__batch_, nullptr);
      __njobs_ = ::cuda::std::exchange(__other.__njobs_, 0);
      __jobs_  = ::cuda::std::move(__other.__jobs_);
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
  _CCCL_HOST_API void cancel()
  {
    if (__valid())
    {
      _CCCL_TRY_CUFILE_API(::cuFileBatchIOCancel, "Failed to cancel IO batch.", __batch_);
    }
  }

  //! @brief Checks if all of the tracked requests are done.
  //!
  //! @return \c true if done, \c false otherwise.
  [[nodiscard]] _CCCL_HOST_API bool is_done()
  {
    return __update_jobs() == 0;
  }

  //! @brief Updates the jobs state and returns a view over the batch jobs.
  //!
  //! @return The view over the batch jobs.
  _CCCL_HOST_API ::cuda::std::span<const cufile_batch_job> update()
  {
    __update_jobs();
    return ::cuda::std::span<const cufile_batch_job>{__jobs_.get(), __njobs_};
  }

  //! @brief Waits for up to \c __timeout nanoseconds before updating the jobs state.
  //!
  //! @param __timeout The maximum ns to wait before updating.
  //!
  //! @return The view over the batch jobs.
  _CCCL_HOST_API ::cuda::std::span<const cufile_batch_job>
  wait_and_update(::cuda::std::chrono::duration<::cuda::std::chrono::nanoseconds> __timeout)
  {
    ::cuda::std::timespec __timeout_spec{};
    __timeout_spec.tv_sec  = static_cast<::cuda::std::time_t>(__timeout.count() / 1'000'000'000);
    __timeout_spec.tv_nsec = static_cast<long>(__timeout.count() % 1'000'000'000);
    __update_jobs(&__timeout_spec);
    return ::cuda::std::span<const cufile_batch_job>{__jobs_.get(), __njobs_};
  }
};

} // namespace cuda::experimental
