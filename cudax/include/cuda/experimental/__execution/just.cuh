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

#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// Map from a disposition to the corresponding tag types:
namespace __detail
{
template <__disposition_t, class _Void = void>
extern _CUDA_VSTD::__undefined<_Void> __just_tag;
template <class _Void>
extern __fn_t<just_t>* __just_tag<__value, _Void>;
template <class _Void>
extern __fn_t<just_error_t>* __just_tag<__error, _Void>;
template <class _Void>
extern __fn_t<just_stopped_t>* __just_tag<__stopped, _Void>;
} // namespace __detail

template <__disposition_t _Disposition>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_PREFERRED_NAME(just_t) _CCCL_PREFERRED_NAME(just_error_t)
  _CCCL_PREFERRED_NAME(just_stopped_t) __just_t
{
private:
  using _JustTag _CCCL_NODEBUG_ALIAS = decltype(__detail::__just_tag<_Disposition>());
  using _SetTag _CCCL_NODEBUG_ALIAS  = decltype(__detail::__set_tag<_Disposition>());

  template <class _Rcvr, class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;
    using __tuple_t _CCCL_NODEBUG_ALIAS               = _CUDA_VSTD::__tuple<_Ts...>;

    _CCCL_API __opstate_t(_Rcvr&& __rcvr, __tuple_t __values)
        : __rcvr_{__rcvr}
        , __values_{static_cast<__tuple_t&&>(__values)}
    {}

#if !_CCCL_COMPILER(GCC)
    // Because of gcc#98995, making this operation state immovable will cause errors in
    // functions that return composite operation states by value. Fortunately, the `just`
    // operation state doesn't strictly need to be immovable, since its address never
    // escapes. So for gcc, we let this operation state be movable.
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);
#endif // !_CCCL_COMPILER(GCC)

    _CCCL_API void start() & noexcept
    {
      _CUDA_VSTD::__apply(
        _SetTag{}, static_cast<_CUDA_VSTD::__tuple<_Ts...>&&>(__values_), static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    __tuple_t __values_;
  };

public:
  template <class... _Ts>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class... _Ts>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Ts... __ts) const;
};

template <__disposition_t _Disposition>
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __just_t<_Disposition>::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return completion_signatures<_SetTag(_Ts...)>{};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && noexcept(__nothrow_decay_copyable<_Rcvr, _Ts...>)
    -> __opstate_t<_Rcvr, _Ts...>
  {
    return __opstate_t<_Rcvr, _Ts...>{
      static_cast<_Rcvr&&>(__rcvr), static_cast<_CUDA_VSTD::__tuple<_Ts...>&&>(__values_)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& noexcept(__nothrow_decay_copyable<_Rcvr, _Ts const&...>)
    -> __opstate_t<_Rcvr, _Ts...>
  {
    return __opstate_t<_Rcvr, _Ts...>{static_cast<_Rcvr&&>(__rcvr), __values_};
  }

  _CCCL_NO_UNIQUE_ADDRESS _JustTag __tag_;
  _CUDA_VSTD::__tuple<_Ts...> __values_;
};

template <__disposition_t _Disposition>
template <class... _Ts>
_CCCL_TRIVIAL_API constexpr auto __just_t<_Disposition>::operator()(_Ts... __ts) const
{
  return __sndr_t<_Ts...>{{}, {static_cast<_Ts&&>(__ts)...}};
}

template <class _Fn>
inline constexpr size_t structured_binding_size<just_t::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<just_error_t::__sndr_t<_Fn>> = 2;
template <class _Fn>
inline constexpr size_t structured_binding_size<just_stopped_t::__sndr_t<_Fn>> = 2;

_CCCL_GLOBAL_CONSTANT auto just         = just_t{};
_CCCL_GLOBAL_CONSTANT auto just_error   = just_error_t{};
_CCCL_GLOBAL_CONSTANT auto just_stopped = just_stopped_t{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_JUST
