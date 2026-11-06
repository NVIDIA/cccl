// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-prefer-addressof
// CLANG_TIDY_CONFIG_END

#include <cuda/std/memory>

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <class T>
struct __wrapper
{
  T member;

  auto address()
  {
    return &*this;
  }
};

template <class T>
struct _Wrapper {};

template <class T>
auto reserved_template_address(__wrapper<T>& value)
{
  return &value;
}

template <class T>
auto const_reserved_template_address(const __wrapper<T>& value)
{
  return &value;
}

template <class T>
auto volatile_reserved_template_address(volatile __wrapper<T>& value)
{
  return &value;
}

template <class T>
auto uppercase_reserved_template_address(_Wrapper<T>& value)
{
  return &value;
}

template <class T>
using public_alias = __wrapper<T>;

template <class T>
auto reserved_template_alias_address(public_alias<T>& value)
{
  return &value;
}

template <class T>
struct public_wrapper {};

template <class T>
auto public_template_address(public_wrapper<T>& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: "#include <cuda/std/__memory/addressof.h>\n\n"
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

template <class T>
using __reserved_alias = public_wrapper<T>;

template <class T>
auto public_template_alias_address(__reserved_alias<T>& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

template <class T>
auto reserved_variable_name(T& __value)
{
  return &__value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 8
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(__value){{'?}}{{$}}
}

template <class T>
auto reserved_template_member_address(__wrapper<T>& value)
{
  return &value.member;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 13
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value.member){{'?}}{{$}}
}

// CHECK-FIXES-NOT: #include
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
