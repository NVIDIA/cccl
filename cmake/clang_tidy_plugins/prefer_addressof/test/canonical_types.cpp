// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-prefer-addressof
// CheckOptions: cccl-prefer-addressof.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#include <cuda/std/memory>

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <class T>
using pointer_alias = T*;

template <class T>
auto pointer_address(pointer_alias<T>& value)
{
  return &value;
}

template <class T>
using member_pointer_alias = int T::*;

template <class T>
auto member_pointer_address(member_pointer_alias<T>& value)
{
  return &value;
}

template <class T>
using array_alias = T[3];

template <class T>
auto array_address(array_alias<T>& value)
{
  return &value;
}

template <class T>
using function_alias = T(T);

template <class T>
auto function_alias_address(function_alias<T>& function)
{
  return &function;
}

template <class T>
auto function_reference_address(T (&function)(T))
{
  return &function;
}

template <class T>
auto function_pointer_dereference(T (*function)(T))
{
  return &*function;
}

template <class T>
auto local_function_address()
{
  T function(T);
  return &function;
}

template <class T>
struct wrapper
{
  static T function(T);

  auto address()
  {
    return &function;
  }
};

template <class T>
auto pointee_address(pointer_alias<T> value)
{
  return &*value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: "#include <cuda/std/__memory/addressof.h>\n\n"
  // CHECK-FIXES: Length: 7
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(*value){{'?}}{{$}}
}

// Model proxy declarations without requiring a Thrust device backend.
namespace thrust
{
template <class T>
struct reference
{
  T* operator&() const;
};

template <class T>
struct device_reference
{
  T* operator&() const;
};

inline namespace abi
{
template <class T>
struct tagged_reference
{
  T* operator&() const;
};
}

template <class T>
struct other {};

namespace detail
{
template <class T>
struct reference {};
}
}

template <class T>
auto thrust_reference_address(thrust::reference<T>& value)
{
  return &value;
}

template <class T>
auto thrust_device_reference_address(thrust::device_reference<T>& value)
{
  return &value;
}

template <class T>
auto thrust_inline_reference_address(thrust::tagged_reference<T>& value)
{
  return &value;
}

template <class T>
using proxy_alias = thrust::device_reference<T>;

template <class T>
auto thrust_proxy_alias_address(const proxy_alias<T>& value)
{
  return &value;
}

template <class T>
struct reference {};

template <class T>
auto global_reference_address(reference<T>& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

namespace unrelated
{
inline namespace abi
{
template <class T>
struct device_reference {};
}
}

template <class T>
auto unrelated_reference_address(unrelated::device_reference<T>& value)
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
auto thrust_other_address(thrust::other<T>& value)
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
auto thrust_detail_reference_address(thrust::detail::reference<T>& value)
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
struct derived_proxy : thrust::device_reference<T> {};

template <class T>
auto derived_proxy_address(derived_proxy<T>& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

// CHECK-FIXES-NOT: #include
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
