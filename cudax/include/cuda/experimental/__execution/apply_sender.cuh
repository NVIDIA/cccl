//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_APPLY_SENDER
#define __CUDAX_EXECUTION_APPLY_SENDER

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/domain.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! A callable object that implements the `std::execution::apply_sender` functionality.
//! This is used to apply a sender to a domain, tag, and arguments, as specified in the
//! C++ standard draft. The implementation ensures compatibility with CUDA C++ Core
//! Libraries.
//! @see https://eel.is/c++draft/exec.snd.apply
struct _CCCL_TYPE_VISIBILITY_DEFAULT apply_sender_t
{
private:
  //! A type alias that determines the domain to apply the sender to. If the expansion of
  //! `__apply_sender_result_t` is valid for the given domain and arguments, the domain is
  //! used; otherwise, the `default_domain` is used.
  //! @tparam _Domain The domain to check.
  //! @tparam _Args The arguments to validate against the domain.
  template <class _Domain, class... _Args>
  using __apply_domain_t _CCCL_NODEBUG_ALIAS = ::cuda::std::
    _If<::cuda::std::_IsValidExpansion<__apply_sender_result_t, _Domain, _Args...>::value, _Domain, default_domain>;

public:
  //! Applies a sender to a domain, tag, and arguments.
  //! @tparam _Domain The domain used to select the algorithm implementation.
  //! @tparam _Tag The tag associated with the algorithm.
  //! @tparam _Sndr The sender to be applied.
  //! @tparam _Args The arguments to pass to the algorithm.
  //! @param __sndr The sender object.
  //! @param __args The arguments to pass to the algorithm.
  //! @return `DOM{}.apply_sender(_Tag{}, __sndr, __args...)`, where `DOM` is the first of
  //! [`_Domain`, `default_domain`] to make the expression well-formed.
  //! @note This function is `constexpr` and `noexcept` if the underlying domain's
  //! `apply_sender` is `noexcept`.
  //! @throws Any exception thrown by the underlying domain's `apply_sender`.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Domain, class _Tag, class _Sndr, class... _Args>
  _CCCL_NODEBUG_API constexpr auto operator()(_Domain, _Tag, _Sndr&& __sndr, _Args&&... __args) const
    noexcept(noexcept(__apply_domain_t<_Domain, _Tag, _Sndr, _Args...>{}.apply_sender(
      _Tag{}, static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...)))
      -> __apply_sender_result_t<__apply_domain_t<_Domain, _Tag, _Sndr, _Args...>, _Tag, _Sndr, _Args...>
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = __apply_domain_t<_Domain, _Tag, _Sndr, _Args...>;
    //! Calls the algorithm specified by _Tag using the determined domain.
    return __dom_t{}.apply_sender(_Tag{}, static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }
};

//! A global constant instance of `apply_sender_t`.
//! This can be used directly to invoke the `apply_sender` functionality.
_CCCL_GLOBAL_CONSTANT apply_sender_t apply_sender{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_APPLY_SENDER
