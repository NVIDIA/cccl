// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/config.cuh>

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__utility/exchange.h>

#if __cpp_lib_format >= 201907L
#  include <format>
#  include <iterator>
#  include <string>
#endif // __cpp_lib_format >= 201907L

#define CUB_TUNING_POLICY_FORMATTER_MEMBER_1(_TYPE, _NAME) \
  ::std::format_to(                                        \
    ::std::back_inserter(str), "{}." #_NAME " = {}", (::cuda::std::exchange(first, false)) ? "" : ", ", p._NAME);
#define CUB_TUNING_POLICY_FORMATTER_MEMBER(_DESC) CUB_TUNING_POLICY_FORMATTER_MEMBER_1 _DESC

#if __cpp_lib_format >= 201907L
#  define CUB_DEFINE_TUNING_POLICY_FORMATTER(_NAME, ...)                             \
    template <::cuda::std::same_as<char> CharT>                                      \
    struct std::formatter<CUB_NS_QUALIFIER::_NAME, CharT> : formatter<string, CharT> \
    {                                                                                \
      template <class FmtCtx>                                                        \
      auto format(const CUB_NS_QUALIFIER::_NAME& p, FmtCtx& ctx) const               \
      {                                                                              \
        string str;                                                                  \
        str += #_NAME "{ ";                                                          \
        bool first = true;                                                           \
        _CCCL_PP_FOR_EACH(CUB_TUNING_POLICY_FORMATTER_MEMBER, __VA_ARGS__)           \
        str += " }";                                                                 \
        return formatter<string, CharT>::format(str, ctx);                           \
      }                                                                              \
    };
#else // ^^^ __cpp_lib_format >= 201907L ^^^ / vvv __cpp_lib_format < 201907L vvv
#  define CUB_DEFINE_TUNING_POLICY_FORMATTER(_NAME, ...)
#endif // ^^^ __cpp_lib_format < 201907L ^^^

#define CUB_DEFINE_TUNING_POLICY_MEMBER_1(_TYPE, _NAME) _TYPE _NAME;
#define CUB_DEFINE_TUNING_POLICY_MEMBER(_DESC)          CUB_DEFINE_TUNING_POLICY_MEMBER_1 _DESC

#define CUB_TUNING_POLICY_OPERATOR_EQ_1(_TYPE, _NAME) &&lhs._NAME == rhs._NAME
#define CUB_TUNING_POLICY_OPERATOR_EQ(_DESC)          CUB_TUNING_POLICY_OPERATOR_EQ_1 _DESC

#define CUB_TUNING_POLICY_OPERATOR_SHL_1(_TYPE, _NAME) \
  << ((::cuda::std::exchange(first, false)) ? "" : ", ") << "." #_NAME " = " << p._NAME
#define CUB_TUNING_POLICY_OPERATOR_SHL(_DESC) CUB_TUNING_POLICY_OPERATOR_SHL_1 _DESC

// Macro for tuning policy definition. Must be used in global namespace.
#define CUB_DEFINE_TUNING_POLICY(_NAME, ...)                                                                 \
  CUB_NAMESPACE_BEGIN                                                                                        \
                                                                                                             \
  struct _NAME                                                                                               \
  {                                                                                                          \
    _CCCL_PP_FOR_EACH(CUB_DEFINE_TUNING_POLICY_MEMBER, __VA_ARGS__)                                          \
                                                                                                             \
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const _NAME& lhs, const _NAME& rhs) \
    {                                                                                                        \
      return true _CCCL_PP_FOR_EACH(CUB_TUNING_POLICY_OPERATOR_EQ, __VA_ARGS__);                             \
    }                                                                                                        \
                                                                                                             \
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const _NAME& lhs, const _NAME& rhs) \
    {                                                                                                        \
      return !(lhs == rhs);                                                                                  \
    }                                                                                                        \
                                                                                                             \
    friend ::std::ostream& operator<<(::std::ostream& os, const _NAME& p)                                    \
    {                                                                                                        \
      bool first = true;                                                                                     \
      return os << #_NAME "{ " _CCCL_PP_FOR_EACH(CUB_TUNING_POLICY_OPERATOR_SHL, __VA_ARGS__) << " }";       \
    }                                                                                                        \
  };                                                                                                         \
                                                                                                             \
  CUB_NAMESPACE_END                                                                                          \
                                                                                                             \
  CUB_DEFINE_TUNING_POLICY_FORMATTER(_NAME, __VA_ARGS__)
