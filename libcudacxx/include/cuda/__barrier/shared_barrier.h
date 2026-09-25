//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_SHARED_BARRIER_H
#define _CUDA___BARRIER_SHARED_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)
#  include <cuda/__barrier/shared_mbarrier.h>
#  include <cuda/__fwd/barrier.h>
#  include <cuda/__memory/address_space.h>
#  include <cuda/__utility/status_policy.h>
#  include <cuda/std/__atomic/scopes.h>
#  include <cuda/std/__chrono/duration.h>
#  include <cuda/std/__chrono/high_resolution_clock.h>
#  include <cuda/std/__chrono/time_point.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda_runtime_api.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE
//! @brief Returns the native shared-memory mbarrier address for a `cuda::shared_barrier`.
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b);
_CCCL_END_NAMESPACE_CUDA_DEVICE

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief A block-scope shared-memory barrier backed by an mbarrier layout-v1 object.
//!
//! `cuda::shared_barrier` is a device-side barrier type for local shared-memory storage. It exposes the common
//! `cuda::barrier<cuda::thread_scope_block>` arrival and wait operations, plus layout-v1 status-reporting waits,
//! conditional-phase waits, and transaction-count operations.
//!
//! The object must be placed in local shared memory and initialized with `cuda::init`. Host and non-shared-memory
//! fallback behavior is not provided by this type.
class shared_barrier : private ::cuda::__detail::__shared_mbarrier_impl
{
  _CCCL_DEVICE_API friend ::cuda::std::uint64_t* ::cuda::device::_LIBCUDACXX_ABI_NAMESPACE::barrier_native_handle(
    ::cuda::shared_barrier& __b);

public:
  //! @brief Result of a status-bearing wait operation.
  //!
  //! An `operation_status` records whether the waited phase completed and whether a status report was produced. If a
  //! report is present, the report must be inspected with `has_report`, `get_error_count`, `for_each_error`, or
  //! `classify` before the object is destroyed or overwritten.
  class operation_status
  {
    bool __complete_                     = false;
    bool __report_predicate_             = false;
    ::cuda::std::uint8_t __report_value_ = 0;
    mutable bool __report_inspected_     = false;

    _CCCL_HOST_DEVICE_API constexpr operation_status(
      bool __complete, bool __report_predicate, ::cuda::std::uint8_t __report_value) noexcept
        : __complete_(__complete)
        , __report_predicate_(__report_predicate)
        , __report_value_(__report_value)
    {}

    _CCCL_HOST_DEVICE_API constexpr operation_status(::cuda::__detail::__mbarrier_wait_status __result) noexcept
        : operation_status(__result.__complete, __result.__report_predicate, __result.__report_value)
    {}

    friend class shared_barrier;

    _CCCL_HOST_DEVICE_API void __assert_report_inspected() const noexcept
    {
      if (__report_predicate_ && !__report_inspected_)
      {
        NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
        _CCCL_UNREACHABLE();
      }
    }

  public:
    //! @brief Constructs an incomplete status with no report.
    _CCCL_HOST_DEVICE_API constexpr operation_status() noexcept {}

    operation_status(const operation_status&)            = delete;
    operation_status& operator=(const operation_status&) = delete;

    //! @brief Move-constructs an operation status.
    _CCCL_HOST_DEVICE_API operation_status(operation_status&& __other) noexcept
        : __complete_(__other.__complete_)
        , __report_predicate_(__other.__report_predicate_)
        , __report_value_(__other.__report_value_)
        , __report_inspected_(__other.__report_inspected_)
    {
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
    }

    //! @brief Move-assigns an operation status.
    //!
    //! If this object currently owns an uninspected report, the assignment traps on device.
    _CCCL_HOST_DEVICE_API operation_status& operator=(operation_status&& __other) noexcept
    {
      __assert_report_inspected();
      __complete_                 = __other.__complete_;
      __report_predicate_         = __other.__report_predicate_;
      __report_value_             = __other.__report_value_;
      __report_inspected_         = __other.__report_inspected_;
      __other.__report_predicate_ = false;
      __other.__report_inspected_ = true;
      return *this;
    }

    //! @brief Destroys an operation status.
    //!
    //! If this object owns an uninspected report, the destructor traps on device.
    _CCCL_HOST_DEVICE_API ~operation_status() noexcept
    {
      __assert_report_inspected();
    }

    //! @brief Checks whether the wait operation completed.
    //!
    //! This does not inspect any status report.
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool complete() const noexcept
    {
      return __complete_;
    }

    //! @brief Checks whether the completed wait produced a status report.
    //!
    //! If a report is present, this marks the report as inspected.
    [[nodiscard]] _CCCL_HOST_DEVICE_API bool has_report() const noexcept
    {
      if (__report_predicate_)
      {
        __report_inspected_ = true;
      }
      return __report_predicate_;
    }

  private:
    _CCCL_DEVICE_API static void __assert_fabric_status(::cudaError_t __status) noexcept
    {
      _CCCL_ASSERT(__status == ::cudaSuccess, "failed to decode shared_barrier status");
      if (__status != ::cudaSuccess)
      {
        ::cuda::std::terminate();
      }
    }

    [[nodiscard]] _CCCL_HOST_DEVICE_API static bool __encodes_fabric_errors(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::generic_fabric:
          return true;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API static ::cudaFabricOpStatusSource
    __cuda_status_source(status_source __source) noexcept
    {
      switch (__source)
      {
        case status_source::generic_fabric:
          return ::cudaFabricOpStatusSourceMbarrierV1;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_DEVICE_API unsigned int __error_count(status_source __source) const noexcept
    {
      if (!__encodes_fabric_errors(__source))
      {
        return 0;
      }
      unsigned int __count = 0;
      auto __report_value  = __report_value_;
      __assert_fabric_status(::cudaFabricOpErrorStatusCount(&__report_value, __cuda_status_source(__source), &__count));
      return __count;
    }

    [[nodiscard]] _CCCL_DEVICE_API ::cudaFabricOpStatusInfo
    __error_status(status_source __source, unsigned int __status_index) const noexcept
    {
      _CCCL_ASSERT(__encodes_fabric_errors(__source), "shared_barrier status source does not encode fabric errors");
      ::cudaFabricOpStatusInfo __status_info{};
      auto __report_value = __report_value_;
      __assert_fabric_status(
        ::cudaFabricOpErrorStatusGet(&__report_value, __cuda_status_source(__source), __status_index, &__status_info));
      return __status_info;
    }

  public:
    //! @brief Returns the number of decoded errors for a status source.
    //!
    //! Marks the report as inspected.
    [[nodiscard]] _CCCL_DEVICE_API unsigned int get_error_count(status_source __source) const noexcept
    {
      __report_inspected_ = true;
      return __error_count(__source);
    }

    //! @brief Invokes `__fn` once for each decoded error for a status source.
    //!
    //! Marks the report as inspected.
    template <class _Fn>
    _CCCL_DEVICE_API void for_each_error(status_source __source, _Fn __fn) const noexcept
    {
      __report_inspected_        = true;
      const unsigned int __count = __error_count(__source);
      for (unsigned int __index = 0; __index != __count; ++__index)
      {
        // TODO: Consider wrapping cudaFabricOpStatusInfo before making this API public.
        __fn(__error_status(__source, __index));
      }
    }

    //! @brief Classifies a status report as a retryable or abort condition for a status source.
    //!
    //! Marks the report as inspected. Calling this function without a report is a programming error.
    [[nodiscard]] _CCCL_DEVICE_API status_action classify(status_source __source) const noexcept
    {
      __report_inspected_ = true;
      _CCCL_ASSERT(__report_predicate_, "cannot classify a shared_barrier operation_status without a report");
      if (!__report_predicate_)
      {
        NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
        _CCCL_UNREACHABLE();
      }
      return __error_count(__source) == 0 ? status_action::retry : status_action::abort;
    }
  };

  //! @brief Opaque token identifying a barrier phase to wait on.
  //!
  //! Tokens are returned by `arrive` and `arrive_tx`. They are intended to be passed to wait operations by the same
  //! thread that performed the arrival.
  class arrival_token
  {
    ::cuda::std::uint64_t __token_ = 0;

    _CCCL_HOST_DEVICE_API explicit constexpr arrival_token(::cuda::std::uint64_t __token) noexcept
        : __token_(__token)
    {}

    friend class shared_barrier;

  public:
    //! @brief Constructs a token with no associated arrival.
    _CCCL_HOST_DEVICE_API constexpr arrival_token() noexcept {}

    _CCCL_HOST_DEVICE_API constexpr arrival_token(const arrival_token& __other) noexcept
        : __token_(__other.__token_)
    {}

    _CCCL_HOST_DEVICE_API constexpr arrival_token(arrival_token&& __other) noexcept
        : __token_(__other.__token_)
    {}

    _CCCL_HOST_DEVICE_API constexpr arrival_token& operator=(const arrival_token& __other) noexcept
    {
      __token_ = __other.__token_;
      return *this;
    }

    _CCCL_HOST_DEVICE_API constexpr arrival_token& operator=(arrival_token&& __other) noexcept
    {
      __token_ = __other.__token_;
      return *this;
    }
  };

  //! @brief Constructs an uninitialized `shared_barrier` object.
  //!
  //! The object must be initialized with `cuda::init` before use.
  _CCCL_HIDE_FROM_ABI shared_barrier() = default;

  shared_barrier(const shared_barrier&)            = delete;
  shared_barrier& operator=(const shared_barrier&) = delete;

private:
  [[noreturn]] _CCCL_HOST_DEVICE_API static void __unsupported_storage() noexcept
  {
    _CCCL_ASSERT(false, "shared_barrier requires local shared memory and mbarrier layout v1 support");
    NV_IF_ELSE_TARGET(NV_IS_HOST, (::cuda::std::terminate();), (::__trap();))
    _CCCL_UNREACHABLE();
  }

  // TODO: Add a public layout-aware count-limit query once the layout-v0 exposure story is settled.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t __max_expected_count() noexcept
  {
    return (1 << 9) - 1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::ptrdiff_t __max_transaction_count_update() noexcept
  {
    return (1 << 20) - 1;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr ::cuda::std::uint64_t
  __token_value(arrival_token __token) noexcept
  {
    return __token.__token_;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void __assert_supported_storage() const
  {
    if (!::cuda::device::is_object_from(__storage_ref(), ::cuda::device::address_space::shared))
    {
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (_CCCL_ASSERT(!::cuda::device::is_object_from(__storage_ref(), ::cuda::device::address_space::cluster_shared),
                      "shared_barrier must not be in another block's cluster shared memory");))
      __unsupported_storage();
    }
  }

  template <class _PollFn, class _CompleteFn>
  [[nodiscard]] _CCCL_HOST_DEVICE_API static auto __wait_until_complete(_PollFn __poll, _CompleteFn __complete)
  {
    auto __result = __poll();
    while (!__complete(__result))
    {
      __result = __poll();
    }
    return __result;
  }

  template <class _Rep, class _Period, class _PollFn, class _TestFn, class _CompleteFn>
  [[nodiscard]] _CCCL_HOST_DEVICE_API static auto __try_wait_for_impl(
    const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, _PollFn __poll, _TestFn __test, _CompleteFn __complete)
  {
    const auto __nanosec = ::cuda::std::chrono::duration_cast<::cuda::std::chrono::nanoseconds>(__dur);

    if (__nanosec.count() < 1)
    {
      return __test();
    }

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (auto __result = __poll(static_cast<::cuda::std::uint32_t>(__nanosec.count()));
       const ::cuda::std::chrono::high_resolution_clock::time_point __start =
         ::cuda::std::chrono::high_resolution_clock::now();
       ::cuda::std::chrono::nanoseconds __elapsed = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       while (!__complete(__result) && (__nanosec > __elapsed)) {
         const ::cuda::std::uint32_t __wait_nsec = static_cast<::cuda::std::uint32_t>((__nanosec - __elapsed).count());
         __result                                = __poll(__wait_nsec);
         __elapsed                               = ::cuda::std::chrono::high_resolution_clock::now() - __start;
       } return __result;))

    __unsupported_storage();
  }

public:
  //! @brief Destroys the mbarrier object.
  //!
  //! The storage must not be reused for another purpose until the mbarrier object is invalidated.
  _CCCL_HOST_DEVICE_API ~shared_barrier()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__inval(); return;))

    __unsupported_storage();
  }

  //! @brief Initializes a `shared_barrier`.
  //!
  //! @param __b Pointer to a `shared_barrier` object in local shared memory.
  //! @param __expected Expected arrival count for each phase.
  _CCCL_HOST_DEVICE_API inline friend void init(shared_barrier* __b, ::cuda::std::ptrdiff_t __expected)
  {
    _CCCL_ASSERT(1 <= __expected, "Expected arrival count must be at least one.");
    _CCCL_ASSERT(__expected <= __max_expected_count(),
                 "Expected arrival count cannot exceed the shared_barrier layout-v1 limit.");

    NV_IF_TARGET(
      NV_PROVIDES_SM_90,
      (__b->__assert_supported_storage(); __b->__init_status_reporting(static_cast<::cuda::std::uint32_t>(__expected));
       return;))

    __unsupported_storage();
  }

  //! @brief Arrives at the current barrier phase.
  //!
  //! @param __update Arrival count update.
  //! @return An arrival token that can be waited on.
  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token arrive(::cuda::std::ptrdiff_t __update = 1)
  {
    _CCCL_ASSERT(1 <= __update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__update <= __max_expected_count(),
                 "Arrival count update cannot exceed the shared_barrier layout-v1 limit.");

    NV_IF_TARGET(NV_PROVIDES_SM_90, (return arrival_token(__arrive(__update));))

    __unsupported_storage();
  }

  //! @brief Increases the expected transaction count for the current phase.
  //!
  //! @param __transaction_count_update Transaction count update.
  _CCCL_HOST_DEVICE_API void expect_tx(::cuda::std::ptrdiff_t __transaction_count_update)
  {
    _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
    _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
                 "Transaction count update cannot exceed the mbarrier transaction count limit.");

    NV_IF_TARGET(NV_PROVIDES_SM_90, (__expect_tx(__transaction_count_update); return;))

    __unsupported_storage();
  }

  //! @brief Arrives at the current phase and increases the expected transaction count.
  //!
  //! @param __arrive_count_update Arrival count update.
  //! @param __transaction_count_update Transaction count update.
  //! @return An arrival token that can be waited on.
  [[nodiscard]] _CCCL_HOST_DEVICE_API arrival_token
  arrive_tx(::cuda::std::ptrdiff_t __arrive_count_update, ::cuda::std::ptrdiff_t __transaction_count_update)
  {
    _CCCL_ASSERT(1 <= __arrive_count_update, "Arrival count update must be at least one.");
    _CCCL_ASSERT(__arrive_count_update <= __max_expected_count(),
                 "Arrival count update cannot exceed the shared_barrier layout-v1 limit.");
    _CCCL_ASSERT(0 <= __transaction_count_update, "Transaction count update must be non-negative.");
    _CCCL_ASSERT(__transaction_count_update <= __max_transaction_count_update(),
                 "Transaction count update cannot exceed the mbarrier transaction count limit.");

    NV_IF_TARGET(NV_PROVIDES_SM_90,
                 (return arrival_token(__arrive_tx(__arrive_count_update, __transaction_count_update));))

    __unsupported_storage();
  }

  //! @brief Tests whether an arrival token's phase completed and returns the status.
  //!
  //! This performs a single completion check.
  //!
  //! @param __token Arrival token to test.
  //! @return Completion and report status for the wait operation.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(arrival_token __token, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__test_wait_status(__token_value(__token)));))

    __unsupported_storage();
  }

  //! @brief Tests whether an arrival token's phase completed and ignores any status report.
  //!
  //! This performs a single completion check.
  //!
  //! @param __token Arrival token to test.
  //! @return `true` if the phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(arrival_token __token, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait(__token_value(__token));))

    __unsupported_storage();
  }

  //! @brief Tries to wait for an arrival token's phase to complete and returns the status.
  //!
  //! This may wait for an implementation-defined bounded interval before returning.
  //!
  //! @param __token Arrival token to wait on.
  //! @return Completion and report status for the wait operation.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(arrival_token __token, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__try_wait_status(__token_value(__token)));))

    __unsupported_storage();
  }

  //! @brief Tries to wait for an arrival token's phase to complete and ignores any status report.
  //!
  //! This may wait for an implementation-defined bounded interval before returning.
  //!
  //! @param __token Arrival token to wait on.
  //! @return `true` if the phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(arrival_token __token, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait(__token_value(__token));))

    __unsupported_storage();
  }

  //! @brief Waits until an arrival token's phase completes and returns the status.
  //!
  //! @param __token Arrival token to wait on.
  //! @return Completed operation status for the waited phase.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(arrival_token __token, return_status_t) const
  {
    return __wait_until_complete(
      [&] {
        return try_wait(__token, return_status);
      },
      [](const operation_status& __result) {
        return __result.complete();
      });
  }

  //! @brief Waits until an arrival token's phase completes and ignores any status report.
  //!
  //! @param __token Arrival token to wait on.
  _CCCL_HOST_DEVICE_API void wait(arrival_token __token, ignore_status_t) const
  {
    (void) __wait_until_complete(
      [&] {
        return try_wait(__token, ignore_status);
      },
      [](bool __complete) {
        return __complete;
      });
  }

  //! @brief Arrives at the barrier and waits for the resulting phase to complete with status.
  //!
  //! @return Completed operation status for the waited phase.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status arrive_and_wait(return_status_t)
  {
    return wait(arrive(), return_status);
  }

  //! @brief Arrives at the barrier and waits for the resulting phase to complete, ignoring status.
  _CCCL_HOST_DEVICE_API void arrive_and_wait(ignore_status_t)
  {
    wait(arrive(), ignore_status);
  }

  //! @brief Arrives at the barrier and removes one expected arrival from subsequent phases.
  _CCCL_HOST_DEVICE_API void arrive_and_drop()
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (__arrive_and_drop(); return;))

    __unsupported_storage();
  }

  //! @brief Tests whether a primary phase completed and returns the status.
  //!
  //! This performs a single completion check.
  //!
  //! @param __phase Primary phase value to test.
  //! @return Completion and report status for the wait operation.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status test_wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__test_wait_phase_status(__phase));))

    __unsupported_storage();
  }

  //! @brief Tests whether a primary phase completed and ignores any status report.
  //!
  //! This performs a single completion check.
  //!
  //! @param __phase Primary phase value to test.
  //! @return `true` if the phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait_phase(__phase);))

    __unsupported_storage();
  }

  //! @brief Tries to wait for a primary phase to complete and returns the status.
  //!
  //! This may wait for an implementation-defined bounded interval before returning.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @return Completion and report status for the wait operation.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return operation_status(__try_wait_phase_status(__phase));))

    __unsupported_storage();
  }

  //! @brief Tries to wait for a primary phase to complete and ignores any status report.
  //!
  //! This may wait for an implementation-defined bounded interval before returning.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @return `true` if the phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait_phase(__phase);))

    __unsupported_storage();
  }

  //! @brief Waits until a primary phase completes and returns the status.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @return Completed operation status for the waited phase.
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status wait(::cuda::std::uint32_t __phase, return_status_t) const
  {
    return __wait_until_complete(
      [&] {
        return try_wait(__phase, return_status);
      },
      [](const operation_status& __result) {
        return __result.complete();
      });
  }

  //! @brief Waits until a primary phase completes and ignores any status report.
  //!
  //! @param __phase Primary phase value to wait on.
  _CCCL_HOST_DEVICE_API void wait(::cuda::std::uint32_t __phase, ignore_status_t) const
  {
    (void) __wait_until_complete(
      [&] {
        return try_wait(__phase, ignore_status);
      },
      [](bool __complete) {
        return __complete;
      });
  }

  //! @brief Tests whether a conditional phase completed.
  //!
  //! For layout v1, the conditional phase advances only when the primary phase completes without a report.
  //!
  //! @param __phase Conditional phase value to test.
  //! @return `true` if the conditional phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool test_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __test_wait_conditional_phase(__phase);))

    __unsupported_storage();
  }

  //! @brief Tries to wait for a conditional phase to complete.
  //!
  //! For layout v1, the conditional phase advances only when the primary phase completes without a report.
  //!
  //! @param __phase Conditional phase value to wait on.
  //! @return `true` if the conditional phase completed, otherwise `false`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, (return __try_wait_conditional_phase(__phase);))

    __unsupported_storage();
  }

  //! @brief Waits until a conditional phase completes.
  //!
  //! For layout v1, the conditional phase advances only when the primary phase completes without a report.
  //!
  //! @param __phase Conditional phase value to wait on.
  _CCCL_HOST_DEVICE_API void wait_conditional_phase(::cuda::std::uint32_t __phase) const
  {
    (void) __wait_until_complete(
      [&] {
        return try_wait_conditional_phase(__phase);
      },
      [](bool __complete) {
        return __complete;
      });
  }

  //! @brief Tries to wait for an arrival token's phase to complete before a relative timeout and returns the status.
  //!
  //! If `__dur` is nonpositive, this performs a single `test_wait` check.
  //!
  //! @param __token Arrival token to wait on.
  //! @param __dur Relative timeout.
  //! @return Completion and report status for the wait operation.
  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    return __try_wait_for_impl(
      __dur,
      [=] _CCCL_DEVICE(::cuda::std::uint32_t __wait_nsec) {
        return operation_status(__try_wait_status(__token_value(__token), __wait_nsec));
      },
      [&] {
        return test_wait(__token, return_status);
      },
      [](const operation_status& __result) {
        return __result.complete();
      });
  }

  //! @brief Tries to wait for an arrival token's phase to complete before a relative timeout and ignores status.
  //!
  //! If `__dur` is nonpositive, this performs a single `test_wait` check.
  //!
  //! @param __token Arrival token to wait on.
  //! @param __dur Relative timeout.
  //! @return `true` if the phase completed before the timeout, otherwise `false`.
  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool
  try_wait_for(arrival_token __token, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    return __try_wait_for_impl(
      __dur,
      [=] _CCCL_DEVICE(::cuda::std::uint32_t __wait_nsec) {
        return __try_wait(__token_value(__token), __wait_nsec);
      },
      [&] {
        return test_wait(__token, ignore_status);
      },
      [](bool __complete) {
        return __complete;
      });
  }

  //! @brief Tries to wait for an arrival token's phase to complete before an absolute timeout and returns the status.
  //!
  //! @param __token Arrival token to wait on.
  //! @param __time Absolute timeout.
  //! @return Completion and report status for the wait operation.
  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, return_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), return_status);
  }

  //! @brief Tries to wait for an arrival token's phase to complete before an absolute timeout and ignores status.
  //!
  //! @param __token Arrival token to wait on.
  //! @param __time Absolute timeout.
  //! @return `true` if the phase completed before the timeout, otherwise `false`.
  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_until(
    arrival_token __token, const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time, ignore_status_t) const
  {
    return try_wait_for(__token, (__time - _Clock::now()), ignore_status);
  }

  //! @brief Tries to wait for a primary phase to complete before a relative timeout and returns the status.
  //!
  //! If `__dur` is nonpositive, this performs a single `test_wait` check.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @param __dur Relative timeout.
  //! @return Completion and report status for the wait operation.
  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_for(
    ::cuda::std::uint32_t __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, return_status_t) const
  {
    return __try_wait_for_impl(
      __dur,
      [=] _CCCL_DEVICE(::cuda::std::uint32_t __wait_nsec) {
        return operation_status(__try_wait_phase_status(__phase, __wait_nsec));
      },
      [&] {
        return test_wait(__phase, return_status);
      },
      [](const operation_status& __result) {
        return __result.complete();
      });
  }

  //! @brief Tries to wait for a primary phase to complete before a relative timeout and ignores status.
  //!
  //! If `__dur` is nonpositive, this performs a single `test_wait` check.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @param __dur Relative timeout.
  //! @return `true` if the phase completed before the timeout, otherwise `false`.
  template <class _Rep, class _Period>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_for(
    ::cuda::std::uint32_t __phase, const ::cuda::std::chrono::duration<_Rep, _Period>& __dur, ignore_status_t) const
  {
    return __try_wait_for_impl(
      __dur,
      [=] _CCCL_DEVICE(::cuda::std::uint32_t __wait_nsec) {
        return __try_wait_phase(__phase, __wait_nsec);
      },
      [&] {
        return test_wait(__phase, ignore_status);
      },
      [](bool __complete) {
        return __complete;
      });
  }

  //! @brief Tries to wait for a primary phase to complete before an absolute timeout and returns the status.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @param __time Absolute timeout.
  //! @return Completion and report status for the wait operation.
  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API operation_status try_wait_until(
    ::cuda::std::uint32_t __phase,
    const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time,
    return_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), return_status);
  }

  //! @brief Tries to wait for a primary phase to complete before an absolute timeout and ignores status.
  //!
  //! @param __phase Primary phase value to wait on.
  //! @param __time Absolute timeout.
  //! @return `true` if the phase completed before the timeout, otherwise `false`.
  template <class _Clock, class _Duration>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool try_wait_until(
    ::cuda::std::uint32_t __phase,
    const ::cuda::std::chrono::time_point<_Clock, _Duration>& __time,
    ignore_status_t) const
  {
    return try_wait_for(__phase, (__time - _Clock::now()), ignore_status);
  }
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief Returns the native shared-memory mbarrier address for a `cuda::shared_barrier`.
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint64_t* barrier_native_handle(::cuda::shared_barrier& __b)
{
  return __b.__native_handle();
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

#endif // _CUDA___BARRIER_SHARED_BARRIER_H
