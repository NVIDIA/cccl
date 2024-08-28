//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_ENV_H
#define __CUDAX_ASYNC_DETAIL_ENV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/reference_wrapper.h>

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <functional>

#include <cuda/experimental/__async/prologue.cuh>

// warning #20012-D: __device__ annotation is ignored on a
// function("inplace_stop_source") that is explicitly defaulted on its first
// declaration
_CCCL_NV_DIAG_SUPPRESS(20012)

namespace cuda::experimental::__async
{
template <class Ty>
extern Ty _unwrap_ref;

template <class Ty>
extern Ty& _unwrap_ref<::std::reference_wrapper<Ty>>;

template <class Ty>
extern Ty& _unwrap_ref<_CUDA_VSTD::reference_wrapper<Ty>>;

template <class Ty>
using _unwrap_reference_t = decltype(_unwrap_ref<Ty>);

template <class Query, class Value>
struct prop
{
  _CCCL_NO_UNIQUE_ADDRESS Query _query;
  _CCCL_NO_UNIQUE_ADDRESS Value _value;

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(Query) const noexcept -> const Value&
  {
    return _value;
  }
};

template <class... Envs>
struct env
{
  _tuple<Envs...> _envs;

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) _get_1st(Query) const noexcept
  {
    constexpr bool _flags[] = {_queryable<Envs, Query>..., false};
    constexpr size_t _idx   = __async::_find_pos(_flags, _flags + sizeof...(Envs));
    if constexpr (_idx != _npos)
    {
      return __async::_cget<_idx>(_envs);
    }
  }

  template <class Query, class Env = env>
  using _1st_env_t = decltype(DECLVAL(const Env&)._get_1st(Query{}));

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(Query query) const
    noexcept(_nothrow_queryable<_1st_env_t<Query>, Query>) //
    -> _query_result_t<_1st_env_t<Query>, Query>
  {
    return _get_1st(query).query(query);
  }
};

// partial specialization for two environments
template <class Env0, class Env1>
struct env<Env0, Env1>
{
  _CCCL_NO_UNIQUE_ADDRESS Env0 _env0;
  _CCCL_NO_UNIQUE_ADDRESS Env1 _env1;

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr decltype(auto) _get_1st(Query) const noexcept
  {
    if constexpr (_queryable<Env0, Query>)
    {
      return (_env0);
    }
    else if constexpr (_queryable<Env1, Query>)
    {
      return (_env1);
    }
  }

  template <class Query, class Env = env>
  using _1st_env_t = decltype(DECLVAL(const Env&)._get_1st(Query{}));

  template <class Query>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto query(Query query) const
    noexcept(_nothrow_queryable<_1st_env_t<Query>, Query>) //
    -> _query_result_t<_1st_env_t<Query>, Query>
  {
    return _get_1st(query).query(query);
  }
};

template <class... Envs>
_CCCL_HOST_DEVICE env(Envs...) -> env<_unwrap_reference_t<Envs>...>;

using empty_env = env<>;

namespace _adl
{
template <class Ty>
_CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto get_env(Ty* ty) noexcept //
  -> decltype(ty->get_env())
{
  static_assert(noexcept(ty->get_env()));
  return ty->get_env();
}

struct _get_env_t
{
  template <class Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Ty* ty) const noexcept //
    -> decltype(get_env(ty))
  {
    static_assert(noexcept(get_env(ty)));
    return get_env(ty);
  }
};
} // namespace _adl

struct get_env_t
{
  template <class Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Ty&& ty) const noexcept //
    -> decltype(ty.get_env())
  {
    static_assert(noexcept(ty.get_env()));
    return ty.get_env();
  }

  template <class Ty>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE auto operator()(Ty* ty) const noexcept //
    -> _call_result_t<_adl::_get_env_t, Ty*>
  {
    return _adl::_get_env_t()(ty);
  }

  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE empty_env operator()(_ignore) const noexcept
  {
    return {};
  }
};

namespace _region
{
_CCCL_GLOBAL_CONSTANT get_env_t get_env{};
} // namespace _region

using namespace _region;

template <class Ty>
using env_of_t = decltype(get_env(DECLVAL(Ty)));
} // namespace cuda::experimental::__async

_CCCL_NV_DIAG_SUPPRESS(20012)

#include <cuda/experimental/__async/epilogue.cuh>

#endif
