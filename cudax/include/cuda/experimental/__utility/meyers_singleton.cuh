//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_MEYERS_SINGLETON
#define _CUDAX__UTILITY_MEYERS_SINGLETON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief A singleton template class implementing the Meyers Singleton design pattern.
//!
//! @tparam _Tp The type of the singleton object.
//!
//! Uses the "Construct On First Use Idiom" to prevent issues related to
//! the static initialization order fiasco.
//!
//! Usage rules:
//! - The default constructor of `_Tp` should be protected.
//! - The destructor of `_Tp` should be protected.
//! - The copy and move constructors of `_Tp` should be disabled (implicit if you follow the rules above).
//!
//! Example usage:
//! @code
//! class my_singleton : public meyers_singleton<my_singleton> {
//! protected:
//!   my_singleton() = default;
//!   ~my_singleton() = default;
//! };
//! @endcode
template <class _Tp>
class meyers_singleton
{
protected:
  template <class _Up>
  struct __wrapper
  {
    using type = _Up;
  };
  friend typename __wrapper<_Tp>::type;

  meyers_singleton()                        = default;
  ~meyers_singleton()                       = default;
  meyers_singleton(const meyers_singleton&) = delete;
  meyers_singleton(meyers_singleton&&)      = delete;

public:
  //! @brief Provides access to the single instance of the class.
  //!
  //! @return A reference to the singleton instance.
  //!
  //! If the instance hasn't been created yet, this function will create it.
  static _Tp& instance() noexcept
  {
    static_assert(!::cuda::std::is_default_constructible_v<_Tp>,
                  "Make the default constructor of your Meyers singleton protected.");
    static_assert(!::cuda::std::is_destructible_v<_Tp>, "Make the destructor of your Meyers singleton protected.");
    static_assert(!::cuda::std::is_copy_constructible_v<_Tp>, "Disable the copy constructor of your Meyers singleton.");
    static_assert(!::cuda::std::is_move_constructible_v<_Tp>, "Disable the move constructor of your Meyers singleton.");
    struct _Derived : _Tp
    {};
    static _Derived __instance;
    return __instance;
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__UTILITY_MEYERS_SINGLETON
