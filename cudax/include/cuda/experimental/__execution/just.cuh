//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_JUST
#define __CUDAX_EXECUTION_JUST

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _JustTag, class _SetTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_t
{
private:
  friend struct just_t;
  friend struct just_error_t;
  friend struct just_stopped_t;

  using __just_tag_t = _JustTag;
  using __set_tag_t  = _SetTag;

  template <class _Rcvr, class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __tuple_t               = ::cuda::std::__tuple<_Ts...>;

    _CCCL_API constexpr explicit __opstate_t(_Rcvr&& __rcvr, __tuple_t __values)
        : __rcvr_{__rcvr}
        , __values_{static_cast<__tuple_t&&>(__values)}
    {}

#if !_CCCL_COMPILER(GCC)
    // Because of gcc#98995, making this operation state immovable will cause errors in
    // functions that return composite operation states by value. Fortunately, the `just`
    // operation state doesn't strictly need to be immovable, since its address never
    // escapes. So for gcc, we let this operation state be movable.
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
    _CCCL_IMMOVABLE(__opstate_t);
#endif // !_CCCL_COMPILER(GCC)

    _CCCL_API constexpr void start() noexcept
    {
      ::cuda::std::__apply(
        _SetTag{}, static_cast<::cuda::std::__tuple<_Ts...>&&>(__values_), static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    __tuple_t __values_;
  };

  template <class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_base_t;

public:
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Ts>
  _CCCL_NODEBUG_API constexpr auto operator()(_Ts... __ts) const;
};

struct just_t : __just_t<just_t, set_value_t>
{
  template <class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

struct just_error_t : __just_t<just_error_t, set_error_t>
{
  template <class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

struct just_stopped_t : __just_t<just_stopped_t, set_stopped_t>
{
  template <class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;
};

template <class _JustTag, class _SetTag>
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_t<_JustTag, _SetTag>::__sndr_base_t
{
  using sender_concept = sender_t;

  template <class>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return completion_signatures<__set_tag_t(_Ts...)>{};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && noexcept(__nothrow_decay_copyable<_Rcvr, _Ts...>)
    -> __opstate_t<_Rcvr, _Ts...>
  {
    return __opstate_t<_Rcvr, _Ts...>{
      static_cast<_Rcvr&&>(__rcvr), static_cast<::cuda::std::__tuple<_Ts...>&&>(__values_)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto
  connect(_Rcvr __rcvr) const& noexcept(__nothrow_decay_copyable<_Rcvr, _Ts const&...>) -> __opstate_t<_Rcvr, _Ts...>
  {
    return __opstate_t<_Rcvr, _Ts...>{static_cast<_Rcvr&&>(__rcvr), __values_};
  }

  [[nodiscard]] _CCCL_API static constexpr auto get_env() noexcept
  {
    return __inln_attrs_t<__set_tag_t>{};
  }

  _CCCL_NO_UNIQUE_ADDRESS __just_tag_t __tag_;
  ::cuda::std::__tuple<_Ts...> __values_;
};

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_t::__sndr_t : __just_t<just_t, set_value_t>::__sndr_base_t<_Ts...>
{};

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_error_t::__sndr_t : __just_t<just_error_t, set_error_t>::__sndr_base_t<_Ts...>
{
  static_assert(sizeof...(_Ts) == 1, "just_error_t must be called with exactly one error type.");
};

template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_stopped_t::__sndr_t
    : __just_t<just_stopped_t, set_stopped_t>::__sndr_base_t<_Ts...>
{
  static_assert(sizeof...(_Ts) == 0, "just_stopped_t must not be called with any types.");
};

_CCCL_EXEC_CHECK_DISABLE
template <class _JustTag, class _SetTag>
template <class... _Ts>
_CCCL_NODEBUG_API constexpr auto __just_t<_JustTag, _SetTag>::operator()(_Ts... __ts) const
{
  using __sndr_t = typename _JustTag::template __sndr_t<_Ts...>;
  return __sndr_t{{{}, {static_cast<_Ts&&>(__ts)...}}};
}

template <class... _Ts>
inline constexpr size_t structured_binding_size<just_t::__sndr_t<_Ts...>> = 2;
template <class... _Ts>
inline constexpr size_t structured_binding_size<just_error_t::__sndr_t<_Ts...>> = 2;
template <class... _Ts>
inline constexpr size_t structured_binding_size<just_stopped_t::__sndr_t<_Ts...>> = 2;

_CCCL_GLOBAL_CONSTANT auto just         = just_t{};
_CCCL_GLOBAL_CONSTANT auto just_error   = just_error_t{};
_CCCL_GLOBAL_CONSTANT auto just_stopped = just_stopped_t{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_JUST
