//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ARCH_ID_H
#define _CUDA___DEVICE_ARCH_ID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/__fwd/devices.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Architecture identifier
//! This type identifies an architecture. It has more possible entries than just numeric values of the compute
//! capability. For example, sm_90 and sm_90a have the same compute capability, but the identifier is different.
enum class arch_id : int
{
#define _CCCL_DEFINE_ARCH_ID(_CC)          sm_##_CC = _CC,
#define _CCCL_DEFINE_ARCH_SPECIFIC_ID(_CC) sm_##_CC##a = _CC * __arch_specific_id_multiplier,
  _CCCL_PP_FOR_EACH(_CCCL_DEFINE_ARCH_ID, _CCCL_KNOWN_PTX_ARCH_LIST)
    _CCCL_PP_FOR_EACH(_CCCL_DEFINE_ARCH_SPECIFIC_ID, _CCCL_KNOWN_PTX_ARCH_SPECIFIC_LIST)
#undef _CCCL_DEFINE_ARCH_ID
#undef _CCCL_DEFINE_ARCH_SPECIFIC_ID
};

[[nodiscard]] _CCCL_API constexpr auto __all_arch_ids() noexcept
{
  return ::cuda::std::array{
#define _CCCL_MAKE_ARCH_ID(_CC)          arch_id::sm_##_CC,
#define _CCCL_MAKE_ARCH_SPECIFIC_ID(_CC) arch_id::sm_##_CC##a,
    _CCCL_PP_FOR_EACH(_CCCL_MAKE_ARCH_ID, _CCCL_KNOWN_PTX_ARCH_LIST)
      _CCCL_PP_FOR_EACH(_CCCL_MAKE_ARCH_SPECIFIC_ID, _CCCL_KNOWN_PTX_ARCH_SPECIFIC_LIST)
#undef _CCCL_MAKE_ARCH_ID
#undef _CCCL_MAKE_ARCH_SPECIFIC_ID
  };
}

[[nodiscard]] _CCCL_API constexpr bool __is_specific_arch(arch_id __arch) noexcept
{
  return ::cuda::std::to_underlying(__arch) > __arch_specific_id_multiplier;
}

[[nodiscard]] _CCCL_API constexpr bool __has_known_arch(compute_capability __cc) noexcept
{
  switch (__cc.get())
  {
#define _CCCL_HAS_KNOWN_ARCH_CASE(_CC) case _CC:
    _CCCL_PP_FOR_EACH(_CCCL_HAS_KNOWN_ARCH_CASE, _CCCL_KNOWN_PTX_ARCH_LIST)
#undef _CCCL_HAS_KNOWN_ARCH_CASE
    return true;
    default:
      return false;
  }
}

[[nodiscard]] _CCCL_API constexpr bool __has_known_specific_arch(compute_capability __cc) noexcept
{
  switch (__cc.get() * __arch_specific_id_multiplier)
  {
#define _CCCL_HAS_KNOWN_SPECFIC_ARCH_CASE(_CC) case _CC:
    _CCCL_PP_FOR_EACH(_CCCL_HAS_KNOWN_SPECFIC_ARCH_CASE, _CCCL_KNOWN_PTX_ARCH_SPECIFIC_LIST)
#undef _CCCL_HAS_KNOWN_SPECFIC_ARCH_CASE
    return true;
    default:
      return false;
  }
}

//! @brief Converts the compute capability to the architecture id.
//!
//! @param __cc The compute capability. Must have a corresponding architecture id.
//!
//! @returns The architecture id.
[[nodiscard]] _CCCL_API constexpr arch_id to_arch_id(compute_capability __cc) noexcept
{
  _CCCL_ASSERT(::cuda::__has_known_arch(__cc), "this compute capability cannot be converted to arch id");
  return static_cast<arch_id>(__cc.get());
}

//! @brief Converts the compute capability to the architecture specific id.
//!
//! @param __cc The compute capability. Must have a corresponding architecture specific id.
//!
//! @returns The architecture specific id.
[[nodiscard]] _CCCL_API constexpr arch_id to_arch_specific_id(compute_capability __cc) noexcept
{
  _CCCL_ASSERT(::cuda::__has_known_specific_arch(__cc),
               "this compute capability cannot be converted to arch specific id");
  return static_cast<arch_id>(__cc.get() * __arch_specific_id_multiplier);
}

_CCCL_END_NAMESPACE_CUDA

#if __cpp_lib_format >= 201907L
_CCCL_BEGIN_NAMESPACE_STD

template <class _CharT>
struct formatter<::cuda::arch_id, _CharT> : private formatter<::cuda::compute_capability, _CharT>
{
  template <class _ParseCtx>
  _CCCL_HOST_API constexpr auto parse(_ParseCtx& __ctx)
  {
    return __ctx.begin();
  }

  template <class _FmtCtx>
  _CCCL_HOST_API auto format(const ::cuda::arch_id& __arch, _FmtCtx& __ctx) const
  {
    auto __it = __ctx.out();
    *__it++   = _CharT{'s'};
    *__it++   = _CharT{'m'};
    *__it++   = _CharT{'_'};
    __ctx.advance_to(__it);
    __it = formatter<::cuda::compute_capability, _CharT>::format(::cuda::compute_capability{__arch}, __ctx);
    if (::cuda::__is_specific_arch(__arch))
    {
      *__it++ = _CharT{'a'};
    }
    return __it;
  }
};

_CCCL_END_NAMESPACE_STD
#endif // __cpp_lib_format >= 201907L

// todo: specialize cuda::std::formatter for cuda::arch_id

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief This function should cause a link error. If it happens, you are trying to compile the code for an unsupported
//!        architecture (too new/old).
_CCCL_DEVICE_API ::cuda::arch_id __unknown_cuda_architecture();

//! @brief Returns the \c cuda::arch_id that is currently being compiled.
//!
//!        If the current architecture is not a known architecture from \c cuda::arch_id enumeration, the compilation
//!        will fail.
//!
//! @note This API cannot be used in constexpr context when compiling with nvc++ in CUDA mode.
template <class _Dummy = void>
[[nodiscard]] _CCCL_DEVICE_API inline _CCCL_TARGET_CONSTEXPR ::cuda::arch_id current_arch_id() noexcept
{
#  if _CCCL_CUDA_COMPILER(NVHPC)
  const auto __cc = ::cuda::device::current_compute_capability();
  if (::cuda::__has_known_arch(__cc))
  {
    return ::cuda::to_arch_id(__cc);
  }
  else
  {
    return ::cuda::device::__unknown_cuda_architecture();
  }
#  elif _CCCL_DEVICE_COMPILATION()
  constexpr auto __cc = ::cuda::device::current_compute_capability();
#    if defined(__CUDA_ARCH_SPECIFIC__)
  constexpr auto __is_known_cc = ::cuda::std::__always_false_v<_Dummy> || ::cuda::__has_known_specific_arch(__cc);
  static_assert(__is_known_cc, "unknown CUDA specific architecture");
  return ::cuda::to_arch_specific_id(__cc);
#    else // ^^^ __CUDA_ARCH_SPECIFIC__ ^^^ / vvv !__CUDA_ARCH_SPECIFIC__ vvv
  constexpr auto __is_known_cc = ::cuda::std::__always_false_v<_Dummy> || ::cuda::__has_known_arch(__cc);
  static_assert(__is_known_cc, "unknown CUDA architecture");
  return ::cuda::to_arch_id(__cc);
#    endif // ^^^ __CUDA_ARCH_SPECIFIC__ ^^^
#  else
  return {};
#  endif // ^^^ single-pass cuda compiler ^^^
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_ARCH_ID_H
