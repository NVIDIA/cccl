// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-qualified-call
// CLANG_TIDY_CONFIG_END

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

namespace other_ns
{
struct base
{
  void member();
  static void static_member();
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
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) other_ns::operator+(left, right);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) ::other_ns::operator+(left, right);
}

void operator_argument(other_ns::number value)
{
  (void) (value + other_ns::make_number());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: {{.*}} [cccl-qualified-call]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  (void) (value + ::other_ns::make_number());
}
}

// CHECK-FIXES: DiagnosticName: cccl-qualified-call

// clang-format on
