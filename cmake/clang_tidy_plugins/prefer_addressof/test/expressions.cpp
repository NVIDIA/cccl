// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-prefer-addressof
// CLANG_TIDY_CONFIG_END

#include <cuda/std/memory>

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <typename T>
auto reference(T& value)
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

template <typename T>
auto const_reference(const T& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

template <typename T>
auto parenthesized(T& value)
{
  return &(value);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 8
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

template <typename T>
auto dereference(T* pointer)
{
  return &*pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 9
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(*pointer){{'?}}{{$}}
}

template <typename T>
auto subscript(T* values)
{
  return &values[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 10
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(values[1]){{'?}}{{$}}
}

template <typename T>
auto member(T& object)
{
  return &object.value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 13
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(object.value){{'?}}{{$}}
}

template <typename T>
auto conditional(bool condition, T& left, T& right)
{
  return &(condition ? left : right);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 27
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(condition ? left : right){{'?}}{{$}}
}

template <typename T>
auto call(T& (*get)())
{
  return &get();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(get()){{'?}}{{$}}
}

struct overloaded_address
{
  overloaded_address* operator&();
};

inline void concrete_types(int& value, overloaded_address& object)
{
  (void) &value;
  (void) &object;
}

template <typename T>
void nondependent_operands(int& value, T* pointer, T (&array)[3])
{
  (void) &value;
  (void) &pointer;
  (void) &array;
}

template <typename T>
void other_operators(T& left, T& right)
{
  (void) (left & right);
  left &= right;
  (void) (left && right);
}

template <typename T>
auto member_pointer()
{
  return &T::value;
}

template <typename T>
auto member_function_pointer()
{
  return &T::function;
}

template <typename T>
auto already_addressof(T& value)
{
  return cuda::std::addressof(value);
}

template <typename T>
void asm_operands(T& value, T* values)
{
  asm volatile("" : : "r"(&value));
  asm volatile("" : "+m"(*&value));
  asm volatile("" : : "r"(*reinterpret_cast<const unsigned*>(&values[0])));
}

template <typename T>
auto address(T& value)
{
  asm volatile("" : : "r"(&value));
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

inline void instantiate(int& integer, double& floating, overloaded_address& object)
{
  (void) address(integer);
  (void) address(floating);
  (void) address(object);
}

namespace visible_overload
{
struct unrelated {};
unrelated* operator&(unrelated&);

template <typename T>
void asm_operand(T& value)
{
  asm volatile("" : : "r"(&value));
}

template <typename T>
auto address(T& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

template <typename T>
auto member_pointer()
{
  return &T::value;
}

inline void instantiate(int& integer, unrelated& object)
{
  (void) visible_overload::address(integer);
  (void) visible_overload::address(object);
}
} // namespace visible_overload

// CHECK-FIXES-NOT: #include
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
