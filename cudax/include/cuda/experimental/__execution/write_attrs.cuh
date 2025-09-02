//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_WRITE_ATTRS
#define __CUDAX_EXECUTION_WRITE_ATTRS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//! @brief Sender adaptor that adds attributes to the child sender's attributes.
struct write_attrs_t
{
  template <class _Sndr, class _Attrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Attrs, class _SndrAttrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t : env<__env_ref_t<_Attrs const&>, __fwd_env_t<_SndrAttrs>>
  {};

  template <class _Attrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_API auto operator()(_Sndr __sndr) &&
    {
      return __sndr_t<_Sndr, _Attrs>{{}, static_cast<_Attrs&&>(__attrs_), static_cast<_Sndr&&>(__sndr)};
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_API friend auto operator|(_Sndr __sndr, __closure_t _clsr)
    {
      return __sndr_t<_Sndr, _Attrs>{{}, static_cast<_Attrs&&>(_clsr.__attrs_), static_cast<_Sndr&&>(__sndr)};
    }

    _Attrs __attrs_;
  };

  //! @brief Applies the given attributes to the sender and returns a new sender with the
  //! attributes attached.
  //!
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Attrs The type of the attributes to be attached.
  //! @param __sndr The sender to which the attributes will be applied.
  //! @param __attrs The attributes to attach to the sender.
  //! @return A new sender type with the specified attributes attached.
  //!
  //! @note This function does not modify the original sender or attributes, but returns a new composed sender.
  //!
  //! **Example:**
  //! @rst
  //! .. code-block:: c++
  //!
  //!    auto sndr = execution::write_attrs(execution::just(),
  //                                        execution::prop{execution::get_domain, MyDomain{}});
  //!    auto domain = execution::get_domain(execution::get_env(sndr));
  //!    static_assert(std::is_same_v<decltype(domain), MyDomain>);
  //!
  //! @endrst
  template <class _Sndr, class _Attrs>
  [[nodiscard]] _CCCL_API auto operator()(_Sndr __sndr, _Attrs __attrs) const -> __sndr_t<_Sndr, _Attrs>
  {
    return __sndr_t<_Sndr, _Attrs>{{}, static_cast<_Attrs&&>(__attrs), static_cast<_Sndr&&>(__sndr)};
  }

  //! @brief Create a sender adaptor closure object that, when combined with a sender,
  //! will apply the specified attributes to that sender.
  //!
  //! @tparam _Attrs The type of the attribute object to be forwarded.
  //! @param __attrs The attribute object to be forwarded to the closure.
  //! @return An instance of `__closure_t<_Attrs>` constructed from the forwarded attributes.
  //!
  //! **Example:**
  //! @rst
  //! .. code-block:: c++
  //!
  //!    auto sndr = execution::just()
  //               | execution::write_attrs(execution::prop{execution::get_domain, MyDomain{}});
  //!    auto domain = execution::get_domain(execution::get_env(sndr));
  //!    static_assert(std::is_same_v<decltype(domain), MyDomain>);
  //!
  //! @endrst
  template <class _Attrs>
  [[nodiscard]] _CCCL_API auto operator()(_Attrs __attrs) const
  {
    return __closure_t<_Attrs>{static_cast<_Attrs&&>(__attrs)};
  }
};

template <class _Sndr, class _Attrs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_attrs_t::__sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return execution::get_child_completion_signatures<_Self, _Sndr, _Env...>();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr _rcvr) && -> connect_result_t<_Sndr, _Rcvr>
  {
    return execution::connect(static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(_rcvr));
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr _rcvr) const& -> connect_result_t<_Sndr, _Rcvr>
  {
    return execution::connect(__sndr_, static_cast<_Rcvr&&>(_rcvr));
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t<_Attrs, env_of_t<_Sndr>>
  {
    return {{__env_ref(__attrs_), __fwd_env(execution::get_env(__sndr_))}};
  }

  _CCCL_NO_UNIQUE_ADDRESS write_attrs_t __tag_;
  _Attrs __attrs_;
  _Sndr __sndr_;
};

_CCCL_GLOBAL_CONSTANT write_attrs_t write_attrs{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_WRITE_ATTRS
