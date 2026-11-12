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
void function();

template <class T>
auto function_template_address()
{
  return &function<T>;
}

template <class T>
using function_address_type = decltype(&function<T>);

template <class T>
void overloaded(T);

template <class T>
void overloaded(T*);

template <class T>
using overload_address_type = decltype(&overloaded<T>);

namespace functions
{
template <class T>
void imported();
}

template <class T>
auto using_declaration_address()
{
  using functions::imported;
  return &imported<T>;
}

struct requirements
{
  using self_type = requirements;

  template <class T>
  static auto well_formed(T value) -> decltype(value + 1);

  template <class... Args, class = decltype(&self_type::well_formed<Args...>)>
  static void satisfied();
};

template <class T>
T variable;

template <class T>
auto variable_template_address()
{
  return &variable<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: "#include <cuda/std/__memory/addressof.h>\n\n"
  // CHECK-FIXES: Length: 12
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(variable<T>){{'?}}{{$}}
}

struct member_functions
{
  template <class T>
  static void function();

  template <class T>
  static void overloaded(T);

  template <class T>
  static void overloaded(T*);
};

template <class T>
auto dot_function_address(member_functions& object)
{
  return &object.function<T>;
}

template <class T>
auto arrow_function_address(member_functions* object)
{
  return &object->function<T>;
}

template <class T>
auto overloaded_function_address(member_functions& object)
  -> decltype(&object.overloaded<T>);

struct derived_functions : member_functions
{
  using member_functions::function;

  template <class T>
  auto this_function_address()
  {
    return &this->function<T>;
  }
};

template <class T>
auto inherited_function_address(derived_functions& object)
{
  return &object.function<T>;
}

struct variables
{
  template <class T>
  static T variable;
};

template <class T>
auto variable_template_address(variables& object)
{
  return &object.variable<T>;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 19
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(object.variable<T>){{'?}}{{$}}
}

template <class T>
auto dependent_member_address(T& object)
{
  return &object.member;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 14
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(object.member){{'?}}{{$}}
}

// CHECK-FIXES-NOT: #include
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
