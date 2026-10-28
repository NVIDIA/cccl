//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief `zip_transform_reduce_into`: the two-input sibling of
 *        `transform_reduce_into` — fuses a binary combine over two
 *        co-partitioned sharded views with the MGMN reduce, so a residual
 *        like `r[i] = b[i] - Ax[i]` never gets materialized on the way to
 *        `sum(r[i]^2)`. Same engine, same three-form structure, same
 *        identity/init contract as `reduce.cuh` (see there for the
 *        rationale); this file only adds the per-shard input construction
 *        for TWO views zipped through a `zip_iterator`.
 *
 * Only the asynchronous, capture-legal `_into` form is provided (the
 * residual-norm/graph-conditional use case this exists for always wants a
 * device-resident result); add a synchronous form the same way
 * `transform_reduce` wraps `transform_reduce_into`'s engine call if needed.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__iterator/zip_iterator.h>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__sharded/algorithm/reduce/reduce.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>

#include <cstddef>
#include <utility>
#include <vector>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Adapts a two-argument zip_op(a, b) into the unary functor CUB's
//! transform-then-reduce path expects, called on a zip_iterator's
//! `tuple<a&, b&>` (mirrors thrust/cuda::std zip_iterator's reference type).
//!
//! Stores @p _ZipOp in an `optional` so this wrapper (and therefore the
//! `transform_iterator` built over it) stays default_initializable even
//! when @p _ZipOp is a capturing lambda closure — under C++17, a lambda
//! (captureless or not) has NO default constructor (that relaxation is
//! C++20-only), and `random_access_iterator`/`forward_iterator` require
//! their iterator type to be default_initializable. The default-constructed
//! state is never dereferenced in practice: `make_transform_iterator`
//! always builds this wrapper from a real, already-engaged instance, and
//! every copy downstream traces back to that one — same convention as a
//! default-constructed (singular, non-dereferenceable) raw-pointer or
//! `std::vector<T>::iterator`.
template <class _ZipOp>
struct __zip_unpack_fn
{
  ::cuda::std::optional<_ZipOp> __op;

  __zip_unpack_fn() = default;

  _CCCL_HOST_DEVICE_API explicit __zip_unpack_fn(_ZipOp __o)
      : __op(::cuda::std::move(__o))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tuple>
  _CCCL_HOST_DEVICE_API auto operator()(_Tuple&& __t) const
    -> decltype((*__op)(::cuda::std::get<0>(__t), ::cuda::std::get<1>(__t)))
  {
    return (*__op)(::cuda::std::get<0>(__t), ::cuda::std::get<1>(__t));
  }
};

//! @brief The engine call for the two-input (zip) form: same shape as
//! `__mgmn_transform_reduce_into_slots`, except the per-lane input is a
//! zip_iterator over BOTH views' shard pointers, wrapped exactly like
//! `__reduce_engine::__input`'s transform overload (one more iterator layer
//! for the lifted-identity path when the operator has none known).
template <class _S1,
          class _S2,
          class _Envs,
          class _DriveEnv,
          class _CallEnv,
          class _Stored,
          class _ZipOp,
          class _ReduceOp,
          class _Vp>
_CCCL_HOST_API void __mgmn_zip_transform_reduce_into_slots(
  const _S1& __data1,
  const _S2& __data2,
  const _Envs& __envs,
  const _DriveEnv& __drive_env,
  const _CallEnv& __call_env,
  const char* __what,
  const ::std::vector<_Stored*>& __outputs,
  _ZipOp __zip_op,
  _ReduceOp __op,
  const _Vp& __init)
{
  using __engine  = __reduce_engine<_ReduceOp, _Vp>;
  using __elem1_t = view_element_t<_S1>;
  using __elem2_t = view_element_t<_S2>;
  static_assert(::cuda::std::is_same_v<_Stored, typename __engine::__stored_t>);

  const auto __reqs = __mgmn_requirements<true>(__call_env);
  __mgmn_drive<true>(
    __data1,
    __envs,
    __drive_env,
    __what,
    [&](const auto& __env) {
      return __mgmn_alloc_env(__env, __reqs);
    },
    [&](const auto& __comms, const auto& __menvs, const auto& __lanes) {
      const auto __inputs = __mgmn_per_lane(__lanes, [&](::std::size_t __g) {
        auto __zit = ::cuda::make_zip_iterator(
          static_cast<const __elem1_t*>(__data1.shard(__g).data), static_cast<const __elem2_t*>(__data2.shard(__g).data));
        auto __zt = ::cuda::make_transform_iterator(__zit, __zip_unpack_fn<_ZipOp>{__zip_op});
        if constexpr (__engine::__direct)
        {
          return __zt;
        }
        else
        {
          return ::cuda::make_transform_iterator(__zt, __lift_fn<_Vp>{});
        }
      });
      ::cuda::experimental::mgmn::reduce(
        ::cuda::experimental::broadcasted,
        __comms,
        __menvs,
        __inputs,
        __mgmn_sizes(__data1, __lanes),
        __outputs,
        __engine::__lift_init(__init),
        __engine::__lift_op(__op),
        __engine::__identity());
    });
}
} // namespace reserved

/**
 * @brief Asynchronous zip transform-reduce over two co-partitioned sharded
 * views, writing the aggregate through an output iterator:
 * `result = init (+) fold(zip_op(data1[i], data2[i]))`. The two-input
 * sibling of `transform_reduce_into` — same contract (capture-legal, no host
 * synchronization, `identity` handled internally so no caller-supplied
 * identity is required), except the per-shard input is a `zip_iterator`
 * over BOTH views' pointers: no combined array is ever materialized (unlike
 * `sharded::transform(binary)` followed by `sharded::reduce`, which is two
 * passes through memory).
 *
 * @param data1, data2 Co-partitioned sharded views (same shard count, same
 *                      per-shard global regions — checked).
 * @param out Device-writable output iterator; written exactly once.
 *
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count, the two views are not co-partitioned, or on more
 *         than 64 shards.
 */
_CCCL_TEMPLATE(
  class _S1, class _S2, class _Envs, class _ZipOp, class _Vp, class _ReduceOp, class _OutIt, class _CallEnv)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S1>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_S2>>
                 _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND async_call_env<_CallEnv>)
_CCCL_HOST_API void zip_transform_reduce_into(
  const _S1& data1,
  const _S2& data2,
  const _Envs& envs,
  _OutIt out,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp init_value,
  const _CallEnv& call_env)
{
  using __stored_t               = typename reserved::__reduce_engine<_ReduceOp, _Vp>::__stored_t;
  const ::std::size_t num_shards = reserved::__shard_count(data1);
  reserved::__check_env_count(envs, num_shards, "sharded::zip_transform_reduce_into");
  reserved::__check_copartitioned(data1, data2, "sharded::zip_transform_reduce_into");
  if (num_shards > reserved::__max_fold_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::zip_transform_reduce_into: more than 64 shards not supported");
  }

  const ::cuda::stream_ref call_stream = ::cuda::get_stream(call_env);

  if (num_shards == 0)
  {
    stream_scope scope(call_stream.get());
    reserved::__mgmn_store_value_kernel<<<1, 1, 0, call_stream.get()>>>(init_value, out);
    cuda_safe_call(cudaGetLastError());
    return;
  }

  auto scratch_mr = ::cuda::mr::get_memory_resource(envs[0]);
  __stored_t* d_slots =
    static_cast<__stored_t*>(scratch_mr.allocate(call_stream, num_shards * sizeof(__stored_t), alignof(__stored_t)));
  ::std::vector<__stored_t*> outputs;
  outputs.reserve(num_shards);
  for (const auto g : each(num_shards))
  {
    outputs.push_back(d_slots + g);
    __detail::__wait_stream_on(::cuda::get_stream(envs[g]).get(), call_stream.get());
  }

  reserved::__mgmn_zip_transform_reduce_into_slots(
    data1,
    data2,
    envs,
    reserved::__lane_ordered_t{},
    call_env,
    "sharded::zip_transform_reduce_into",
    outputs,
    zip_op,
    reduce_op,
    init_value);

  for (const auto g : each(num_shards))
  {
    __detail::__wait_stream_on(call_stream.get(), ::cuda::get_stream(envs[g]).get());
  }

  {
    stream_scope scope(call_stream.get());
    reserved::__mgmn_store_kernel<<<1, 1, 0, call_stream.get()>>>(static_cast<const __stored_t*>(d_slots), out);
    cuda_safe_call(cudaGetLastError());
  }
  scratch_mr.deallocate(call_stream, d_slots, num_shards * sizeof(__stored_t), alignof(__stored_t));
}

/**
 * @brief Asynchronous zip transform-reduce over two self-bound sharded
 * structures: environments derived via `default_envs(data1)` (both views
 * must be co-partitioned over the same places).
 */
_CCCL_TEMPLATE(class _S1, class _S2, class _ZipOp, class _Vp, class _ReduceOp, class _OutIt, class _CallEnv)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S1>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_S2>>
                 _CCCL_AND async_call_env<_CallEnv>)
_CCCL_HOST_API void zip_transform_reduce_into(
  const _S1& data1,
  const _S2& data2,
  _OutIt out,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp init_value,
  const _CallEnv& call_env)
{
  const auto envs = default_envs(data1);
  sharded::zip_transform_reduce_into(data1, data2, envs, out, zip_op, reduce_op, init_value, call_env);
}
} // namespace cuda::experimental::sharded
