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

#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__utility/forward.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @cond

//! @brief Invokes a property query or returns a fallback value.
//!
//! When @c UseAdvertisedQueries is @c true and @p env exposes discoverable property-key metadata, the
//! advertised list is authoritative: @p query is invoked only when @p env advertises it with no
//! additional arguments. Otherwise, selection is based on whether @p query is callable with @p env.
//!
//! @tparam UseAdvertisedQueries Whether to honor an environment's advertised query list.
//! @tparam QueryT The query callable type.
//! @tparam FallbackT The fallback value type.
//! @tparam EnvT The queried environment or object type.
//! @param[in] query The property query to invoke.
//! @param[in] fallback The value returned when the query is not selected.
//! @param[in] env The environment or object to query.
//! @return The query result when selected; otherwise @p fallback.
template <bool UseAdvertisedQueries, typename QueryT, typename FallbackT, typename EnvT>
[[nodiscard]] CUB_RUNTIME_FUNCTION auto query_or([[maybe_unused]] QueryT query, FallbackT&& fallback, const EnvT& env)
{
  if constexpr (UseAdvertisedQueries && ::cuda::std::execution::__detail::__has_property_keys<EnvT>)
  {
    if constexpr (::cuda::std::execution::__detail::__advertises_query_v<EnvT, QueryT>)
    {
      return query(env);
    }
    else
    {
      return static_cast<FallbackT&&>(fallback);
    }
  }
  else
  {
    return ::cuda::__call_or(query, static_cast<FallbackT&&>(fallback), env);
  }
}

//! @brief Validates an advertised environment or a query-providing object.
//!
//! An environment with property-key metadata discoverable through
//! @c cuda::execution::property_keys_t must accept every advertised query expression on a const object.
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
                  "An object passed as a DeviceReduce environment must provide a stream or a memory resource.");
  }
}

//! @brief Dispatches an algorithm with automatically managed temporary storage.
//!
//! This function resolves the stream, memory resource, and tuning environment; queries the
//! required temporary-storage size; allocates the storage; invokes the algorithm; and then
//! releases the storage. When @c UseAdvertisedQueries is @c true and @p env exposes discoverable
//! property-key metadata, the advertised list is authoritative. Otherwise, properties are selected
//! by testing whether their customization point objects are callable with @p env. Missing or
//! unselected properties use their defaults. This function does not validate @p env.
//!
//! @tparam UseAdvertisedQueries Whether property lookup honors advertised query metadata. Defaults to @c false.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The temporary-storage sizing or allocation error, if any; otherwise the execution error, if any;
//! otherwise the deallocation status.
template <bool UseAdvertisedQueries = false, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env(const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  // Query stream from environment
  auto stream = detail::query_or<UseAdvertisedQueries>(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // Query memory resource from environment
  auto mr =
    detail::query_or<UseAdvertisedQueries>(::cuda::mr::get_memory_resource, detail::device_memory_resource{}, env);

  // Query tuning from environment
  const auto tuning =
    detail::query_or<UseAdvertisedQueries>(::cuda::execution::__get_tuning, ::cuda::std::execution::env<>{}, env);

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

//! @brief Validates an environment or object and dispatches with automatically managed temporary storage.
//!
//! Advertised query expressions are validated before dispatch. For environments with discoverable
//! metadata, the advertised list controls whether stream, memory-resource, and tuning queries are
//! used. Objects without metadata must provide at least a stream or memory resource and retain
//! callability-based property lookup.
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
  return detail::dispatch_with_env<true>(env, ::cuda::std::forward<AlgorithmCallable>(algorithm_callable));
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
//! the supplied temporary storage. When @c UseAdvertisedQueries is @c true and @p env exposes
//! discoverable property-key metadata, the advertised list is authoritative. Otherwise, properties
//! are selected by testing whether their customization point objects are callable with @p env.
//! Missing or unselected properties use their defaults. This function does not validate @p env.
//!
//! @tparam UseAdvertisedQueries Whether property lookup honors advertised query metadata. Defaults to @c false.
//! @tparam EnvT The execution environment or query-providing object type.
//! @tparam AlgorithmCallable The algorithm implementation callable type.
//! @param[in] d_temp_storage @devicestorage
//! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param[in] env The execution environment or query-providing object.
//! @param[in] algorithm_callable The algorithm implementation callable.
//! @return The status returned by @p algorithm_callable.
template <bool UseAdvertisedQueries = false, typename EnvT, typename AlgorithmCallable>
[[nodiscard]] CUB_RUNTIME_FUNCTION cudaError_t dispatch_with_env(
  void* d_temp_storage, size_t& temp_storage_bytes, const EnvT& env, AlgorithmCallable&& algorithm_callable)
{
  // Query stream from environment
  auto stream = detail::query_or<UseAdvertisedQueries>(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // Query tuning from environment
  const auto tuning =
    detail::query_or<UseAdvertisedQueries>(::cuda::execution::__get_tuning, ::cuda::std::execution::env<>{}, env);

  return algorithm_callable(tuning, d_temp_storage, temp_storage_bytes, stream.get());
}

//! @brief Validates an environment or object and dispatches with user-provided temporary storage.
//!
//! Advertised query expressions are validated before dispatch. For environments with discoverable
//! metadata, the advertised list controls whether stream and tuning queries are used. Objects without
//! metadata must provide at least a stream or memory resource and retain callability-based property
//! lookup.
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
  return detail::dispatch_with_env<true>(
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
