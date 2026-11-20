// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-avoid-adl
// CheckOptions: cccl-avoid-adl.Namespaces=cub;cuda
// CheckOptions: cccl-avoid-adl.NamespaceMacros=cub=_CUB::
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns
{
struct base
{
  void member();
  static void static_member();
  static void (*pointer)();
  template <class T>
  static void generic(T);
};

template <class T>
struct box
{
  static void call();
};
}

namespace cub
{
struct object : ::other_ns::base
{
  void member();
  static void static_member();

  void implicit_object()
  {
    member();
    this->member();
    static_member();
    base::member();
  }
};

struct function_object
{
  int operator()(int) const;
};

struct inherited_members : ::other_ns::base
{
  using ::other_ns::base::static_member;
  using ::other_ns::base::generic;
  using member_alias = ::other_ns::base;

  void resolved()
  {
    static_member();
    generic(0);
    pointer();
    member_alias::static_member();
  }

  template <class T>
  void unresolved(T value)
  {
    generic(value);
  }
};

template <class T>
struct dependent_base : ::other_ns::box<T>
{
  using ::other_ns::box<T>::call;
  void example() { call(); }
};

struct local_pointer
{
  void example()
  {
    static void (*pointer)();
    pointer();
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
    // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
    // CHECK-FIXES: Replacements: []
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  }
};

void qualified_static_pointer()
{
  other_ns::base::pointer();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
}

// We can find class members before resolving the overload or substituting the argument type.
template <class T>
struct result_probe
{
  static char test(...);
  template <class U>
  static long test(U*);

  using type = decltype(test((T*) nullptr));
  static constexpr bool value = sizeof(test((T*) nullptr)) == sizeof(long);
};

struct probe_base
{
  template <class U>
  static long test(U*);
};

template <class T>
struct inherited_probe : probe_base
{
  using probe_base::test;
  using type = decltype(test((T*) nullptr));
};

template <class T>
void dependent_members(T& object, T* pointer)
{
  object.member();
  pointer->member();
  object.template member_template<int>();
}

void explicit_objects(object& value, object* pointer, function_object function)
{
  value.member();
  pointer->member();
  value.static_member();
  pointer->static_member();
  (value.*&object::member)();
  function(0);
  function.operator()(0);
  auto lambda = [] {};
  lambda();
}

void local_indirect_calls(void (*pointer)(), void (&reference)())
{
  pointer();
  reference();
  (*pointer)();
}

template <void (*Function)()>
void non_type_parameter()
{
  Function();
}

template <class Function>
void function_parameter(Function function)
{
  function();
}

template <class Function>
void dependent_local_callable(Function function)
{
  auto dispatch = function;
  dispatch();
}

template <class Function>
struct callable_fields
{
  Function dispatch;
  void (*pointer)();

  void example()
  {
    dispatch();
    this->dispatch();
    pointer();
    this->pointer();
  }
};
}

namespace other_ns
{
struct number
{
  int operator[](int) const;
  int operator()(int) const;
  bool operator!() const;
};
number operator+(number, number);
bool operator==(number, number);
bool operator!=(number, number);
bool operator<(number, number);
bool operator<=(number, number);
bool operator>(number, number);
bool operator>=(number, number);
number make_number();
}

namespace cub
{
void operator_syntax(other_ns::number left, other_ns::number right)
{
  (void) (left + right);
  (void) (left == right);
  (void) (left != right);
  (void) (left < right);
  (void) (left <= right);
  (void) (left > right);
  (void) (left >= right);
  (void) !left;
  (void) left[0];
  (void) left(0);
  (void) left.operator[](0);
  (void) left.operator()(0);
}

void explicit_free_operators(other_ns::number left, other_ns::number right)
{
  (void) operator+(left, right);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::other_ns::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) other_ns::operator+(left, right);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) ::other_ns::operator+(left, right);
}

void operator_argument(other_ns::number value)
{
  (void) (value + other_ns::make_number());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-avoid-adl]
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-avoid-adl
  // CHECK-FIXES: FileOffset: [[#CALL:]]
  // CHECK-FIXES: Replacements:
  // CHECK-FIXES-NEXT: - FilePath:
  // CHECK-FIXES-NEXT: Offset: [[#CALL]]
  // CHECK-FIXES-NEXT: Length: 0
  // CHECK-FIXES-NEXT: ReplacementText: '::'
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) (value + ::other_ns::make_number());
}
}

// CHECK-FIXES-NOT: DiagnosticName:

// clang-format on
