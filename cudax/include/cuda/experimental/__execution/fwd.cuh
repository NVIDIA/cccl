//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_FWD
#define __CUDAX_EXECUTION_FWD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_BEGIN_NV_DIAG_SUPPRESS(2642) // call through incomplete class "cuda::experimental::execution::schedule_t"
                                   // will always produce an error when instantiated.

namespace cuda::experimental
{
// so we can refer to the cuda::experimental::__detail namespace below
namespace __detail
{
}
namespace execution
{
namespace __detail
{
using namespace cuda::experimental::__detail; // NOLINT(misc-unused-using-decls)
} // namespace __detail

// NOLINTBEGIN(misc-unused-using-decls)
using ::cuda::std::execution::__forwarding_query;
using ::cuda::std::execution::__unwrap_reference_t;
using ::cuda::std::execution::env;
using ::cuda::std::execution::env_of_t;
using ::cuda::std::execution::forwarding_query;
using ::cuda::std::execution::forwarding_query_t;
using ::cuda::std::execution::get_env;
using ::cuda::std::execution::get_env_t;
using ::cuda::std::execution::prop;

using ::cuda::std::execution::__nothrow_queryable_with;
using ::cuda::std::execution::__query_result_t;
using ::cuda::std::execution::__queryable_with;

using ::cuda::std::execution::__query_or;
using ::cuda::std::execution::__query_result_or_t;
// NOLINTEND(misc-unused-using-decls)

template <class _Env, class _Query, bool _Default>
_CCCL_CONCEPT __nothrow_queryable_with_or =
  bool(__queryable_with<_Env, _Query> ? __nothrow_queryable_with<_Env, _Query> : _Default);

struct _CCCL_TYPE_VISIBILITY_DEFAULT receiver_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT operation_state_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT sender_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT scheduler_t
{};

template <class _Ty>
using __sender_concept_t _CCCL_NODEBUG_ALIAS = typename ::cuda::std::remove_reference_t<_Ty>::sender_concept;

template <class _Ty>
using __receiver_concept_t _CCCL_NODEBUG_ALIAS = typename ::cuda::std::remove_reference_t<_Ty>::receiver_concept;

template <class _Ty>
using __scheduler_concept_t _CCCL_NODEBUG_ALIAS = typename ::cuda::std::remove_reference_t<_Ty>::scheduler_concept;

template <class _Ty>
using __operation_state_concept_t _CCCL_NODEBUG_ALIAS =
  typename ::cuda::std::remove_reference_t<_Ty>::operation_state_concept;

template <class _Ty>
inline constexpr bool __is_sender = __is_instantiable_with<__sender_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_receiver = __is_instantiable_with<__receiver_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_scheduler = __is_instantiable_with<__scheduler_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_operation_state = __is_instantiable_with<__operation_state_concept_t, _Ty>;

struct _CCCL_TYPE_VISIBILITY_DEFAULT dependent_sender_error;

struct _CCCL_TYPE_VISIBILITY_DEFAULT default_domain;

template <class... _Sigs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT completion_signatures;

template <class _Sndr, class... _Env>
_CCCL_NODEBUG_API _CCCL_CONSTEVAL auto get_completion_signatures();

template <class _Sndr, class... _Env>
using completion_signatures_of_t _CCCL_NODEBUG_ALIAS = decltype(execution::get_completion_signatures<_Sndr, _Env...>());

// handy enumerations for keeping type names readable
enum class __disposition : int8_t
{
  __invalid = -1,
  __value,
  __error,
  __stopped
};

// customization point objects:
struct _CCCL_TYPE_VISIBILITY_DEFAULT set_value_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT set_error_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT set_stopped_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT start_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT connect_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT schedule_t;

template <class _Sch>
using schedule_result_t _CCCL_NODEBUG_ALIAS = decltype(declval<schedule_t>()(declval<_Sch>()));

template <class _Sndr, class _Rcvr>
using connect_result_t _CCCL_NODEBUG_ALIAS = decltype(declval<connect_t>()(declval<_Sndr>(), declval<_Rcvr>()));

#if _CCCL_HOST_COMPILATION()
template <class _Sndr, class _Rcvr>
inline constexpr bool __nothrow_connectable = noexcept(declval<connect_t>()(declval<_Sndr>(), declval<_Rcvr>()));
#else // ^^^ _CCCL_HOST_COMPILATION() ^^^ / vvv !_CCCL_HOST_COMPILATION() vvv
template <class _Sndr, class _Rcvr>
inline constexpr bool __nothrow_connectable = __is_instantiable_with<connect_result_t, _Sndr, _Rcvr>;
#endif // ^^^ !_CCCL_HOST_COMPILATION() ^^^

// sender factory algorithms:
struct _CCCL_TYPE_VISIBILITY_DEFAULT read_env_t;

struct _CCCL_TYPE_VISIBILITY_DEFAULT just_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_error_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_stopped_t;

struct _CCCL_TYPE_VISIBILITY_DEFAULT just_from_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_error_from_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT just_stopped_from_t;

// sender adaptor algorithms:
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_value_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_error_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_stopped_t;

struct _CCCL_TYPE_VISIBILITY_DEFAULT then_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_error_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT upon_stopped_t;

struct _CCCL_TYPE_VISIBILITY_DEFAULT when_all_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_env_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT starts_on_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT continues_on_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT schedule_from_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_chunked_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT bulk_unchunked_t;

// sender consumer algorithms:
struct _CCCL_TYPE_VISIBILITY_DEFAULT sync_wait_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT start_detached_t;

// queries:
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_allocator_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_stop_token_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_scheduler_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_previous_scheduler_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_delegation_scheduler_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_forward_progress_guarantee_t;
template <class _Tag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_completion_scheduler_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_domain_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_domain_override_t;
template <class _Tag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_completion_domain_t;
struct _CCCL_TYPE_VISIBILITY_DEFAULT get_completion_behavior_t;

template <class _Ty>
using stop_token_of_t _CCCL_NODEBUG_ALIAS = decay_t<__call_result_t<get_stop_token_t, _Ty>>;

template <class _Env>
using __scheduler_of_t _CCCL_NODEBUG_ALIAS = decay_t<__call_result_t<get_scheduler_t, _Env>>;

template <class _Env>
using __previous_scheduler_of_t _CCCL_NODEBUG_ALIAS = decay_t<__call_result_t<get_previous_scheduler_t, _Env>>;

// get_forward_progress_guarantee:
enum class forward_progress_guarantee
{
  concurrent,
  parallel,
  weakly_parallel
};

namespace __detail
{
struct __get_tag
{
  template <class _Tag, class... _Child>
  _CCCL_NODEBUG_API constexpr auto operator()(int, _Tag, ::cuda::std::__ignore_t, _Child&&...) const -> _Tag
  {
    return _Tag{};
  }
};

template <class _Sndr, class _Tag = __visit_result_t<__get_tag&, _Sndr, int&>>
extern __fn_ptr_t<_Tag> __tag_of_v;
} // namespace __detail

template <class _Sndr>
using tag_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__tag_of_v<_Sndr>());

template <class _Sndr, class... _Tag>
inline constexpr bool __sender_for_v = _CCCL_REQUIRES_EXPR((_Sndr, variadic _Tag))(tag_of_t<_Sndr>{});

template <class _Sndr, class _Tag>
inline constexpr bool __sender_for_v<_Sndr, _Tag> =
  _CCCL_REQUIRES_EXPR((_Sndr, _Tag))(_Same_as(_Tag) tag_of_t<_Sndr>{});

template <class _Sndr, class... _Tag>
_CCCL_CONCEPT sender_for = __sender_for_v<_Sndr, _Tag...>;

namespace __detail
{
template <class _Sig>
inline constexpr __disposition __signature_disposition = __disposition::__invalid;
template <class... _Ts>
inline constexpr __disposition __signature_disposition<set_value_t(_Ts...)> = __disposition::__value;
template <class _Ty>
inline constexpr __disposition __signature_disposition<set_error_t(_Ty)> = __disposition::__error;
template <>
inline constexpr __disposition __signature_disposition<set_stopped_t()> = __disposition::__stopped;

} // namespace __detail

struct inline_scheduler;

struct stream_domain;
struct stream_context;
struct stream_scheduler;

} // namespace execution

} // namespace cuda::experimental

_CCCL_END_NV_DIAG_SUPPRESS()

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_FWD
