//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXECUTION_POLICY_H
#define _CUDA_STD___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream.h>
#  include <cuda/__stream/stream_ref.h>
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__fwd/execution_policy.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

[[nodiscard]] _CCCL_API constexpr bool __has_unique_backend(const __execution_backend __backends) noexcept
{
  return ::cuda::std::has_single_bit(static_cast<uint32_t>(__backends));
}

//! @brief Base class for our execution policies.
//! It takes an untagged uint32_t because we want to be able to store 3 different enumerations in it.
template <uint32_t _Policy, class... _Envs>
struct __execution_policy_base : env<__unwrap_reference_t<_Envs>...>
{
  //! @brief Tag that identifies this and all derived classes as a CCCL execution policy
  static constexpr uint32_t __cccl_policy_ = _Policy;

  //! @brief Extracts the execution policy from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_policy __get_policy() noexcept
  {
    return __policy_to_execution_policy<_Policy>;
  }

  //! @brief Extracts the execution backend from the stored _Policy
  [[nodiscard]] _CCCL_API static constexpr __execution_backend __get_backend() noexcept
  {
    return __policy_to_execution_backend<_Policy>;
  }

  //! Forwards queries to the env
  using env<__unwrap_reference_t<_Envs>...>::query;

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
  //! @brief create a new policy with additional environments attached
  template <class _Env, size_t... _Indices>
  [[nodiscard]] _CCCL_HOST_API constexpr __execution_policy_base<_Policy, _Env, _Envs...>
  __with(_Env&& __env, index_sequence<_Indices...>) const
  {
    if constexpr (sizeof...(_Envs) == 2)
    {
      return __execution_policy_base<_Policy, _Env, _Envs...>{
        ::cuda::std::forward<_Env>(__env), this->__env0_, this->__env1_};
    }
    else
    {
      return __execution_policy_base<_Policy, _Env, _Envs...>{
        ::cuda::std::forward<_Env>(__env), ::cuda::std::__get<_Indices>(this->__envs_)...};
    }
  }

  //! @brief Prepend an environment to the current ones
  template <class _Env>
  [[nodiscard]] _CCCL_HOST_API constexpr auto with(_Env&& __env) const
  {
    if constexpr (__convertible_to_stream_ref<_Env>
                  && (is_lvalue_reference_v<_Env> || is_same_v<remove_cvref_t<_Env>, ::cudaStream_t>) )
    { // streams are special in that they are their own environment, but we always want to store a stream_ref
      // We must reject prvalue cuda::stream because they are not copyable
      static_assert(!is_same_v<remove_cvref_t<_Env>, ::cuda::stream> || is_lvalue_reference_v<_Env>,
                    "cuda::stream is not copyable. It must be passed as a cuda::stream_ref");
      ::cuda::stream_ref __stream{__env};
      return __with(prop{::cuda::get_stream, __stream}, ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
    }
    else if constexpr (::cuda::mr::resource<_Env>)
    {
      static_assert(!is_const_v<_Env>, "A memory resource must be passed by non-const reference");
      if constexpr (!is_lvalue_reference_v<_Env> && !::cuda::mr::__is_resource_ref<remove_cvref_t<_Env>>)
      { // The user passed a prvalue, which indicates we should own the resource
        return __with(prop{::cuda::mr::get_memory_resource,
                           ::cuda::mr::any_resource<> {
                             ::cuda::std::move(__env)
                           }},
                      ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
      }
      else
      {
        return __with(prop{::cuda::mr::get_memory_resource,
                           ::cuda::mr::resource_ref<> {
                             __env
                           }},
                      ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
      }
    }
    else
    {
      return __with(::cuda::std::forward<_Env>(__env), ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
    }
  }

  //! @brief Create a new environment from a tag and a value and prepend
  template <class _Tag, class _Value>
  [[nodiscard]] _CCCL_HOST_API constexpr auto with(const _Tag& __tag, _Value&& __value) const
  {
    if constexpr (is_same_v<remove_cvref_t<_Tag>, ::cuda::get_stream_t>)
    { // We want to force the use of ::cuda::stream_ref
      // We must reject prvalue cuda::stream because they are not copyable
      static_assert(!is_same_v<remove_cvref_t<_Value>, ::cuda::stream> || is_lvalue_reference_v<_Value>,
                    "cuda::stream is not copyable. It must be passed as a cuda::stream_ref");
      ::cuda::stream_ref __stream{__value};
      return __with(prop{__tag, __stream}, ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
    }
    else if constexpr (is_same_v<remove_cvref_t<_Tag>, ::cuda::mr::get_memory_resource_t>)
    {
      static_assert(!is_const_v<_Value>, "A memory resource must be passed by non-const reference");
      if constexpr (!is_lvalue_reference_v<_Value> && !::cuda::mr::__is_resource_ref<remove_cvref_t<_Value>>)
      { // The user passed a prvalue, which indicates we should own the resource
        return __with(prop{__tag,
                           ::cuda::mr::any_resource<> {
                             ::cuda::std::move(__value)
                           }},
                      ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
      }
      else
      {
        return __with(prop{__tag,
                           ::cuda::mr::resource_ref<> {
                             __value
                           }},
                      ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
      }
    }
    else
    {
      return __with(prop{__tag, ::cuda::std::forward<_Value>(__value)},
                    ::cuda::std::make_index_sequence<sizeof...(_Envs)>());
    }
  }
#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
};

using sequenced_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__sequenced)>;
_CCCL_GLOBAL_CONSTANT sequenced_policy seq{};

using parallel_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__parallel)>;
_CCCL_GLOBAL_CONSTANT parallel_policy par{};

using parallel_unsequenced_policy =
  __execution_policy_base<static_cast<uint32_t>(__execution_policy::__parallel_unsequenced)>;
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy par_unseq{};

using unsequenced_policy = __execution_policy_base<static_cast<uint32_t>(__execution_policy::__unsequenced)>;
_CCCL_GLOBAL_CONSTANT unsequenced_policy unseq{};

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXECUTION_POLICY_H
