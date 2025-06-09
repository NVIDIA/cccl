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
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__launch/configuration.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

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
using namespace cuda::experimental::__detail; // // NOLINT(misc-unused-using-decls)
} // namespace __detail

struct _CCCL_TYPE_VISIBILITY_DEFAULT receiver_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT operation_state_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT sender_t
{};

struct _CCCL_TYPE_VISIBILITY_DEFAULT scheduler_t
{};

template <class _Ty>
using __sender_concept_t _CCCL_NODEBUG_ALIAS = typename _CUDA_VSTD::remove_reference_t<_Ty>::sender_concept;

template <class _Ty>
using __receiver_concept_t _CCCL_NODEBUG_ALIAS = typename _CUDA_VSTD::remove_reference_t<_Ty>::receiver_concept;

template <class _Ty>
using __scheduler_concept_t _CCCL_NODEBUG_ALIAS = typename _CUDA_VSTD::remove_reference_t<_Ty>::scheduler_concept;

template <class _Ty>
using __operation_state_concept_t _CCCL_NODEBUG_ALIAS =
  typename _CUDA_VSTD::remove_reference_t<_Ty>::operation_state_concept;

template <class _Ty>
inline constexpr bool __is_sender = __is_instantiable_with_v<__sender_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_receiver = __is_instantiable_with_v<__receiver_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_scheduler = __is_instantiable_with_v<__scheduler_concept_t, _Ty>;

template <class _Ty>
inline constexpr bool __is_operation_state = __is_instantiable_with_v<__operation_state_concept_t, _Ty>;

struct stream_domain;
struct dependent_sender_error;

struct default_domain;

template <class... _Sigs>
struct completion_signatures;

template <class _Sndr, class... _Env>
_CCCL_TRIVIAL_API _CCCL_CONSTEVAL auto get_completion_signatures();

template <class _Sndr, class... _Env>
using completion_signatures_of_t _CCCL_NODEBUG_ALIAS = decltype(get_completion_signatures<_Sndr, _Env...>());

// handy enumerations for keeping type names readable
enum __disposition_t
{
  __value,
  __error,
  __stopped
};

// customization point objects:
struct set_value_t;
struct set_error_t;
struct set_stopped_t;
struct start_t;
struct connect_t;
struct schedule_t;

// sender factory algorithms:
template <__disposition_t>
struct __just_t;
using just_t         = __just_t<__value>;
using just_error_t   = __just_t<__error>;
using just_stopped_t = __just_t<__stopped>;

template <__disposition_t>
struct __just_from_t;
using just_from_t         = __just_from_t<__value>;
using just_error_from_t   = __just_from_t<__error>;
using just_stopped_from_t = __just_from_t<__stopped>;

struct read_env_t;

// sender adaptor algorithms:
template <__disposition_t>
struct __let_t;
using let_value_t   = __let_t<__value>;
using let_error_t   = __let_t<__error>;
using let_stopped_t = __let_t<__stopped>;

template <__disposition_t>
struct __upon_t;
using then_t         = __upon_t<__value>;
using upon_error_t   = __upon_t<__error>;
using upon_stopped_t = __upon_t<__stopped>;

struct when_all_t;
struct conditional_t;
struct sequence_t;
struct write_env_t;
struct starts_on_t;
struct continues_on_t;
struct schedule_from_t;
struct bulk_t;

// sender consumer algorithms:
struct sync_wait_t;
struct start_detached_t;

// queries:
struct get_allocator_t;
struct get_stop_token_t;
struct get_scheduler_t;
struct get_delegation_scheduler_t;
struct get_forward_progress_guarantee_t;
template <class _Tag>
struct get_completion_scheduler_t;
struct get_domain_t;

namespace __detail
{
struct __get_tag
{
  template <class _Tag, class... _Child>
  _CCCL_TRIVIAL_API constexpr auto operator()(int, _Tag, _CUDA_VSTD::__ignore_t, _Child&&...) const -> _Tag
  {
    return _Tag{};
  }
};

template <class _Sndr, class _Tag = __visit_result_t<__get_tag&, _Sndr, int&>>
extern __fn_ptr_t<_Tag> __tag_of_v;
} // namespace __detail

template <class _Sndr>
using tag_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__tag_of_v<_Sndr>());

template <class _Sndr, class _Tag>
_CCCL_CONCEPT __sender_for = _CCCL_REQUIRES_EXPR((_Sndr, _Tag))(_Same_as(_Tag) tag_of_t<_Sndr>{});

namespace __detail
{
template <__disposition_t, class _Void = void>
extern _CUDA_VSTD::__undefined<_Void> __set_tag;
template <class _Void>
extern __fn_t<set_value_t>* __set_tag<__value, _Void>;
template <class _Void>
extern __fn_t<set_error_t>* __set_tag<__error, _Void>;
template <class _Void>
extern __fn_t<set_stopped_t>* __set_tag<__stopped, _Void>;
} // namespace __detail

} // namespace execution

} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_FWD
