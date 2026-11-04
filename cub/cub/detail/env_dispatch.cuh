// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/device_memory_resource.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_runs_on.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/guarantee.h>
#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__utility/forward.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @cond

//! @brief Validates an advertised environment or a query-providing object.
//!
//! An environment with property-key metadata discoverable through
//! @c cuda::execution::property_keys_t must provide every advertised member customization and support
//! invocation of its query customization point object on a const object.
//! An object without advertised metadata must provide at least a stream or a memory resource.
//!
//! @tparam EnvT The environment or object type to validate.
template <typename EnvT>
CUB_RUNTIME_FUNCTION void validate_env_or_object()
{
  if constexpr (::cuda::std::execution::__detail::__has_property_keys<EnvT>)
  {
    ::cuda::std::execution::__detail::__validate_env<EnvT>();
  }
  else
  {
    static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, const EnvT&>
                    || ::cuda::std::__is_callable_v<::cuda::mr::get_memory_resource_t, const EnvT&>,
                  "A non-env object passed to DeviceReduce must provide a stream or a memory resource.");
  }
}

//! @brief Dispatches an algorithm with automatically managed temporary storage.
//!
//! This function resolves the stream, memory resource, and tuning environment; queries the
//! required temporary-storage size; allocates the storage; invokes the algorithm; and then
//! releases the storage. Queries are used when callable with @p env, and unavailable queries use
//! their defaults. This function does not validate @p env.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The temporary-storage sizing or allocation error, if any; otherwise the execution error, if any;
//! otherwise the deallocation status.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env(const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  // Query stream from environment
  auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // Query memory resource from environment
  auto mr = ::cuda::__call_or(::cuda::mr::get_memory_resource, detail::device_memory_resource{}, env);

  // Query tuning from environment
  const auto tuning = ::cuda::__call_or(::cuda::execution::__get_tuning, ::cuda::std::execution::env<>{}, env);

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;

  // Phase 1: Query temporary storage size
  if (const auto error = algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get()))
  {
    return error;
  }

  // Allocate temporary storage
  if (const auto error = CubDebug(detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr)))
  {
    return error;
  }

  // Phase 2: Execute algorithm
  const auto error = algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get());

  // Deallocate temporary storage (always attempt, even on error)
  const auto deallocate_error =
    CubDebug(detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr));

  // Algorithm error takes precedence over deallocation error
  return (error != cudaSuccess) ? error : deallocate_error;
}

//! @brief Dispatches with automatic storage and passes the runs_on guarantee to the algorithm.
//!
//! Queries the bundled guarantees using callability-based lookup and appends the resolved
//! @c runs_on value to the arguments passed to @p algorithm_callable. This function does not
//! validate advertised query metadata.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the runs_on guarantee.
//! @return The status returned by automatic-storage dispatch.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t
dispatch_with_env_and_runs_on(const EnvT& env, const AlgorithmCallable& algorithm_callable)
{
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, experimental::__get_runs_on_t>,
                "runs_on should be used inside guarantee to have an effect.");
  auto guarantees = ::cuda::__call_or(::cuda::execution::__get_guarantees, ::cuda::std::execution::env<>{}, env);
  auto runs_on    = ::cuda::__call_or(experimental::__get_runs_on, experimental::RunsOn{}, guarantees);

  return detail::dispatch_with_env(
    env, [&](auto&& tuning, void* d_temp_storage, size_t& temp_storage_bytes, ::cudaStream_t stream) {
      return algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream, runs_on);
    });
}

//! @brief Validates an environment or object and dispatches with automatically managed temporary storage.
//!
//! Advertised query expressions are validated before dispatch. Stream, memory-resource, and tuning
//! queries are also used when callable without being advertised. Objects without metadata must
//! provide at least a stream or memory resource.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The temporary-storage sizing or allocation error, if any; otherwise the execution error, if any;
//! otherwise the deallocation status.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t
dispatch_with_env_validation(const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  detail::validate_env_or_object<EnvT>();
  return detail::dispatch_with_env(env, ::cuda::std::forward<AlgorithmCallable>(algorithm_callable));
}

//! @brief Dispatches with automatic storage and selects a tuning policy from the environment.
//!
//! This interface uses callability-based property lookup and does not validate advertised query metadata.
//!
//! @tparam DefaultPolicySelector The policy selector used when the tuning environment does not provide one.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the selected policy.
//! @return The temporary-storage sizing or allocation error, if any; otherwise the execution error, if any;
//! otherwise the deallocation status.
template <typename DefaultPolicySelector, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t
dispatch_with_env_and_tuning(const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  // A forwarding reference accepts temporary callables, but the callable is deliberately invoked as an lvalue because
  // automatic-storage dispatch invokes it twice.
  return detail::dispatch_with_env(
    env,
    [&algorithm_callable](
      [[maybe_unused]] auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      using policy_t = decltype(DefaultPolicySelector{}(::cuda::compute_capability{}));
      using policy_selector =
        ::cuda::std::execution::__query_result_or_t<decltype(tuning_env), policy_t, DefaultPolicySelector>;
      return algorithm_callable(policy_selector{}, d_temp_storage, temp_storage_bytes, stream);
    });
}

//! @brief Validates an environment or object, dispatches with automatic storage, and selects a tuning policy.
//!
//! This interface applies the advertised-query validation and lookup rules of
//! @c dispatch_with_env_validation before selecting the tuning policy.
//!
//! @tparam DefaultPolicySelector The policy selector used when the tuning environment does not provide one.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the selected policy.
//! @return The temporary-storage sizing or allocation error, if any; otherwise the execution error, if any;
//! otherwise the deallocation status.
template <typename DefaultPolicySelector, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t
dispatch_with_env_validation_and_tuning(const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  return detail::dispatch_with_env_validation(
    env,
    [&algorithm_callable](
      [[maybe_unused]] auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      using policy_t = decltype(DefaultPolicySelector{}(::cuda::compute_capability{}));
      using policy_selector =
        ::cuda::std::execution::__query_result_or_t<decltype(tuning_env), policy_t, DefaultPolicySelector>;
      return algorithm_callable(policy_selector{}, d_temp_storage, temp_storage_bytes, stream);
    });
}

//! @brief Dispatches an algorithm with user-provided temporary storage.
//!
//! This function resolves the stream and tuning environment and invokes the algorithm once with
//! the supplied temporary storage. Queries are used when callable with @p env, and unavailable queries
//! use their defaults. This function does not validate @p env.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The status returned by @p algorithm_callable.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  // Query stream from environment
  auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // Query tuning from environment
  const auto tuning = ::cuda::__call_or(::cuda::execution::__get_tuning, ::cuda::std::execution::env<>{}, env);

  return algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get());
}

//! @brief Dispatches with user-provided storage and passes the runs_on guarantee to the algorithm.
//!
//! Queries the bundled guarantees using callability-based lookup and appends the resolved
//! @c runs_on value to the arguments passed to @p algorithm_callable. This function does not
//! validate advertised query metadata.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of @p d_temp_storage.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the runs_on guarantee.
//! @return The status returned by @p algorithm_callable.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env_and_runs_on(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, const AlgorithmCallable& algorithm_callable)
{
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, experimental::__get_runs_on_t>,
                "runs_on should be used inside guarantee to have an effect.");
  auto guarantees = ::cuda::__call_or(::cuda::execution::__get_guarantees, ::cuda::std::execution::env<>{}, env);
  auto runs_on    = ::cuda::__call_or(experimental::__get_runs_on, experimental::RunsOn{}, guarantees);

  // Preserve the separate storage query so callers can size storage without a device.
  return detail::dispatch_with_env(
    d_temp_storage, temp_storage_bytes, env, [&](auto&& tuning, void* storage, size_t& bytes, ::cudaStream_t stream) {
      return algorithm_callable(tuning, storage, bytes, stream, runs_on);
    });
}

//! @brief Validates an environment or object and dispatches with user-provided temporary storage.
//!
//! Advertised query expressions are validated before dispatch. Stream and tuning queries are also
//! used when callable without being advertised. Objects without metadata must provide at least a
//! stream or memory resource.
//!
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The status returned by @p algorithm_callable.
template <typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env_validation(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  detail::validate_env_or_object<EnvT>();
  return detail::dispatch_with_env(
    d_temp_storage, temp_storage_bytes, env, ::cuda::std::forward<AlgorithmCallable>(algorithm_callable));
}

//! @brief Dispatches with user-provided storage and selects a tuning policy from the environment.
//!
//! This interface uses callability-based property lookup and does not validate advertised query metadata.
//!
//! @tparam DefaultPolicySelector The policy selector used when the tuning environment does not provide one.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the selected policy.
//! @return The status returned by @p algorithm_callable.
template <typename DefaultPolicySelector, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env_and_tuning(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  return detail::dispatch_with_env(
    d_temp_storage,
    temp_storage_bytes,
    env,
    [&algorithm_callable](
      [[maybe_unused]] auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      using policy_t = decltype(DefaultPolicySelector{}(::cuda::compute_capability{}));
      using policy_selector =
        ::cuda::std::execution::__query_result_or_t<decltype(tuning_env), policy_t, DefaultPolicySelector>;
      return ::cuda::std::forward<AlgorithmCallable>(
        algorithm_callable)(policy_selector{}, d_temp_storage, temp_storage_bytes, stream);
    });
}

//! @brief Validates an environment or object, uses caller-provided storage, and selects a tuning policy.
//!
//! This interface applies the advertised-query validation and lookup rules of
//! @c dispatch_with_env_validation before selecting the tuning policy.
//!
//! @tparam DefaultPolicySelector The policy selector used when the tuning environment does not provide one.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable that receives the selected policy.
//! @return The status returned by @p algorithm_callable.
template <typename DefaultPolicySelector, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env_validation_and_tuning(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  return detail::dispatch_with_env_validation(
    d_temp_storage,
    temp_storage_bytes,
    env,
    [&algorithm_callable](
      [[maybe_unused]] auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
      using policy_t = decltype(DefaultPolicySelector{}(::cuda::compute_capability{}));
      using policy_selector =
        ::cuda::std::execution::__query_result_or_t<decltype(tuning_env), policy_t, DefaultPolicySelector>;
      return ::cuda::std::forward<AlgorithmCallable>(
        algorithm_callable)(policy_selector{}, d_temp_storage, temp_storage_bytes, stream);
    });
}
//! @endcond
} // namespace detail

CUB_NAMESPACE_END
