//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_COMPUTE_CAPABILITY_H
#define _CUDA___DEVICE_COMPUTE_CAPABILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/devices.h>
#include <cuda/std/__charconv/from_chars.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/to_underlying.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Type representing the CUDA compute capability.
class compute_capability
{
  int __cc_{}; //!< The stored compute capability in format 10 * major + minor.

public:
  _CCCL_HIDE_FROM_ABI constexpr compute_capability() noexcept = default;

  //! @brief Constructs the object from compute capability \c __cc. The expected format is 10 * major + minor.
  //!
  //! @param __cc Compute capability.
  _CCCL_API explicit constexpr compute_capability(int __cc) noexcept
      : __cc_{__cc}
  {}

  //! @brief Constructs the object by combining the \c __major and \c __minor compute capability.
  //!
  //! @param __major The major compute capability.
  //! @param __minor The minor compute capability. Must be less than 10.
  _CCCL_API constexpr compute_capability(int __major, int __minor) noexcept
      : __cc_{10 * __major + __minor}
  {
    _CCCL_ASSERT(__major < ::cuda::std::numeric_limits<int>::max() / 10, "invalid major compute capability");
    _CCCL_ASSERT(__minor < 10, "invalid minor compute capability");
  }

  //! @brief Constructs the object from the architecture id.
  //!
  //! @param __arch_id The architecture id.
  _CCCL_API explicit constexpr compute_capability(arch_id __arch_id) noexcept
  {
    const auto __val = ::cuda::std::to_underlying(__arch_id);
    if (__val > __arch_specific_id_multiplier)
    {
      __cc_ = __val / __arch_specific_id_multiplier;
    }
    else
    {
      __cc_ = __val;
    }
  }

  _CCCL_HIDE_FROM_ABI constexpr compute_capability(const compute_capability&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr compute_capability& operator=(const compute_capability& __other) noexcept = default;

  //! @brief Gets the stored compute capability.
  //!
  //! @return The stored compute capability in format 10 * major + minor.
  [[nodiscard]] _CCCL_API constexpr int get() const noexcept
  {
    return __cc_;
  }

  //! @brief Gets the major compute capability.
  //!
  //! @return Major compute capability.
  //!
  //! @deprecated This symbol is deprecated because it collides with major(...) macro defined in <sys/sysmacros.h> and
  //! will be removed in next major release. Use cc.major_cap() instead.
  [[nodiscard]]
  CCCL_DEPRECATED_BECAUSE("This symbol is deprecated because it collides with major(...) macro defined in "
                          "<sys/sysmacros.h> and will be removed in next major release. Use cc.major_cap() instead.")
  _CCCL_API constexpr int major() const noexcept
  {
    return major_cap();
  }

  //! @brief Gets the major compute capability.
  //!
  //! @return Major compute capability.
  [[nodiscard]] _CCCL_API constexpr int major_cap() const noexcept
  {
    return __cc_ / 10;
  }

  //! @brief Gets the minor compute capability.
  //!
  //! @return Minor compute capability. The value is always less than 10.
  //!
  //! @deprecated This symbol is deprecated because it collides with minor(...) macro defined in <sys/sysmacros.h> and
  //! will be removed in next major release. Use cc.minor_cap() instead.
  [[nodiscard]]
  CCCL_DEPRECATED_BECAUSE("This symbol is deprecated because it collides with minor(...) macro defined in "
                          "<sys/sysmacros.h> and will be removed in next major release. Use cc.minor_cap() instead.")
  _CCCL_API constexpr int minor() const noexcept
  {
    return minor_cap();
  }

  //! @brief Gets the minor compute capability.
  //!
  //! @return Minor compute capability. The value is always less than 10.
  [[nodiscard]] _CCCL_API constexpr int minor_cap() const noexcept
  {
    return __cc_ % 10;
  }

  //! @brief Conversion operator to \c int.
  //!
  //! @return The stored compute capability in format 10 * major + minor.
  _CCCL_API explicit constexpr operator int() const noexcept
  {
    return __cc_;
  }

  //! @brief Equality operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ == __rhs.__cc_;
  }

  //! @brief Inequality operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ != __rhs.__cc_;
  }

  //! @brief Less than operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ < __rhs.__cc_;
  }

  //! @brief Less than or equal to operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ <= __rhs.__cc_;
  }

  //! @brief Greater than operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator>(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ > __rhs.__cc_;
  }

  //! @brief Greater than or equal to operator.
  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(compute_capability __lhs, compute_capability __rhs) noexcept
  {
    return __lhs.__cc_ >= __rhs.__cc_;
  }
};

inline namespace literals
{
inline namespace compute_capability_literals
{
_CCCL_DIAG_PUSH

// In case of success, cuda::std::from_chars returns cuda::std::errc{}, which is not an enumerated type of errc, so we
// need to suppress the warnings.
_CCCL_DIAG_SUPPRESS_CLANG("-Wswitch")
_CCCL_DIAG_SUPPRESS_GCC("-Wswitch")
_CCCL_DIAG_SUPPRESS_MSVC(4063)

// When cudafe++ recreates the source file for the host compiler, it produces `operator "" _cc`, which is deprecated by
// CWG2521, so we need to suppress the warnings.
#if _CCCL_COMPILER(CLANG, >=, 20)
_CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated-literal-operator")
#endif // _CCCL_COMPILER(CLANG, >=, 20)

[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL ::cuda::compute_capability
__cc_literal_impl(const char* __start, const char* __end) noexcept
{
  int __major{};
  const auto __major_result = ::cuda::std::from_chars(__start, __end, __major);
  switch (__major_result.ec)
  {
    case ::cuda::std::errc{}:
      break;
    case ::cuda::std::errc::result_out_of_range:
      _CCCL_VERIFY(false, "cuda::compute_capability literal major version out of range");
    case ::cuda::std::errc::invalid_argument:
    default:
      _CCCL_VERIFY(false,
                   "invalid cuda::compute_capability literal, must have format of `M.m` where `M`is the major version "
                   "and `m` the minor version");
  }

  auto __it = __major_result.ptr;
  _CCCL_VERIFY(__it != __end && *__it++ == '.', "cuda::compute_capability literal is missing the minor version part");
  _CCCL_VERIFY(__it + 1 == __end, "cuda::compute_capability literal can have only 1 minor digit");
  int __minor{};
  const auto __minor_result = ::cuda::std::from_chars(__it, __end, __minor);
  switch (__minor_result.ec)
  {
    case ::cuda::std::errc{}:
      break;
    case ::cuda::std::errc::result_out_of_range:
      _CCCL_VERIFY(false, "cuda::compute_capability literal minor version out of range");
    case ::cuda::std::errc::invalid_argument:
    default:
      _CCCL_VERIFY(false,
                   "invalid cuda::compute_capability literal, must have format of `M.m` where `M`is the major version "
                   "and `m` the minor version");
  }

  return ::cuda::compute_capability{__major, __minor};
}

//! @brief \c cuda::compute_capability literal. The expected format is `M.m` where `M`is the major version and `m` the
//!        minor version.
//!
//! Examples:
//! @code
//! using namespace cuda::compute_capability_literals;
//! const auto cc1 = 1.0_cc;
//! const auto cc2 = 12.1_cc;
//! const auto invalid_cc1 = 1_cc; // fails because of missing `.m`
//! const auto invalid_cc2 = 0x8.0.cc; // fails, must be of base 10
//! const auto invalid_cc3 = 1.103_cc // fails, minor version can have only 1 digit
//! @endcode
template <char... _Cs>
_CCCL_API _CCCL_CONSTEVAL ::cuda::compute_capability operator""_cc() noexcept
{
  constexpr char __cs[]{_Cs...};
  constexpr auto __ret = ::cuda::__cc_literal_impl(__cs, __cs + sizeof...(_Cs));
  return __ret;
}

_CCCL_DIAG_POP
} // namespace compute_capability_literals
} // namespace literals

_CCCL_END_NAMESPACE_CUDA

#if __cpp_lib_format >= 201907L
_CCCL_BEGIN_NAMESPACE_STD

template <class _CharT>
struct formatter<::cuda::compute_capability, _CharT> : private formatter<int, _CharT>
{
  template <class _ParseCtx>
  _CCCL_HOST_API constexpr auto parse(_ParseCtx& __ctx)
  {
    return __ctx.begin();
  }

  template <class _FmtCtx>
  _CCCL_HOST_API auto format(const ::cuda::compute_capability& __cc, _FmtCtx& __ctx) const
  {
    return formatter<int, _CharT>::format(__cc.get(), __ctx);
  }
};

_CCCL_END_NAMESPACE_STD
#endif // __cpp_lib_format >= 201907L

// todo: specialize cuda::std::formatter for cuda::compute_capability

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief Returns the \c cuda::compute_capability that is currently being compiled.
//!
//! @note This API cannot be used in constexpr context when compiling with nvc++ in CUDA mode.
[[nodiscard]] _CCCL_DEVICE_API inline _CCCL_TARGET_CONSTEXPR ::cuda::compute_capability
current_compute_capability() noexcept
{
#  if _CCCL_CUDA_COMPILER(NVHPC)
  return ::cuda::compute_capability{__builtin_current_device_sm()};
#  elif _CCCL_DEVICE_COMPILATION()
  return ::cuda::compute_capability{__CUDA_ARCH__ / 10};
#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
  return {};
#  endif // ^^^ !_CCCL_DEVICE_COMPILATION() ^^^
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_COMPUTE_CAPABILITY_H
