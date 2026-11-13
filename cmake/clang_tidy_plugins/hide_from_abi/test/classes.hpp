// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-hide-from-abi
// CheckOptions: cccl-hide-from-abi.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#include <cuda/std/detail/__config>

#undef _CCCL_HOST
#undef _CCCL_DEVICE
#undef _CCCL_HOST_DEVICE
// Attribute expansions keep the annotation in the declaration's source range.
#define _CCCL_HOST __attribute__((annotate("host")))
#define _CCCL_DEVICE __attribute__((annotate("device")))
#define _CCCL_HOST_DEVICE _CCCL_HOST _CCCL_DEVICE

// CHECK-MESSAGES-NOT: warning:
struct members {
  _CCCL_HOST members() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST ~members() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_DEVICE operator bool() const { return true; }
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST_DEVICE members& operator=(const members&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_DEVICE void method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST static void static_method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST virtual void virtual_method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST friend void friend_function(members) {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  void declared();
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
_CCCL_HOST inline void members::declared() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:

// CHECK-FIXES: Replacements: []
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

struct implicit_members { int value; };
inline implicit_members object{};
inline auto copied_object = object;

template <class T> struct container {
  _CCCL_HOST container() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_DEVICE void method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
template struct container<int>;
extern template struct container<double>;
inline container<char> implicit_instantiation;

template <> struct container<bool> {
  _CCCL_DEVICE void specialized_method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
template <class T> struct container<T*> {
  _CCCL_DEVICE void partial_method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
inline container<int*> partial_instantiation;

// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_DEVICE_API
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_DEVICE_API
// CHECK-FIXES: Length: 12
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_DEVICE_API

struct __attribute__((visibility("hidden"))) hidden_class {
  _CCCL_HOST void unannotated_method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};

// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

namespace defaulted_special_members {
// CHECK-MESSAGES-NOT: warning:
struct members {
  members() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  members(const members&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  members(members&&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  members& operator=(const members&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  members& operator=(members&&) = default;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  ~members() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};

// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '

struct annotated_members {
  _CCCL_HIDE_FROM_ABI annotated_members() = default;
  _CCCL_HIDE_FROM_ABI annotated_members(const annotated_members&) = default;
  _CCCL_HIDE_FROM_ABI annotated_members(annotated_members&&) = default;
  _CCCL_HIDE_FROM_ABI annotated_members& operator=(const annotated_members&) = default;
  _CCCL_HIDE_FROM_ABI annotated_members& operator=(annotated_members&&) = default;
  _CCCL_HIDE_FROM_ABI ~annotated_members() = default;
};

template <class T> struct container {
  constexpr explicit container() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
template struct container<int>;
inline container<char> implicit_instantiation;

// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '

struct out_of_line {
  out_of_line();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '
};
inline out_of_line::out_of_line() = default;
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use _CCCL_HIDE_FROM_ABI to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES: Replacements:
// CHECK-FIXES: Length: 0
// CHECK-FIXES-NEXT: ReplacementText: '_CCCL_HIDE_FROM_ABI '

} // namespace defaulted_special_members

namespace member_redeclarations {
struct members;
struct members;
struct members {
  _CCCL_HOST members();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST ~members();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  _CCCL_HOST void method();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  __attribute__((visibility("hidden"))) void hidden_method();
};
_CCCL_HOST inline members::members() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST inline members::~members() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST inline void members::method() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
_CCCL_HOST inline void members::hidden_method() {}

// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API
// CHECK-FIXES: Length: 10
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_API

template <class T> struct container;
template <class T> struct container {
  template <class U>
  _CCCL_HOST_DEVICE U method(U value);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
};
template <class T>
template <class U>
_CCCL_HOST_DEVICE U container<T>::method(U value) { return value; }
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use _CCCL_HOST_DEVICE_API instead to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
inline auto instantiated = container<int>{}.method(1);
inline auto also_instantiated = container<double>{}.method(1.0);

// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
// CHECK-FIXES: Length: 17
// CHECK-FIXES-NEXT: ReplacementText: _CCCL_HOST_DEVICE_API
} // namespace member_redeclarations

// The enclosing function still needs an annotation; its local members do not.
// CHECK-MESSAGES-NOT: warning:
inline void enclosing_function() {
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function needs an API macro (one of _CCCL_API, _CCCL_DEVICE_API, or _CCCL_HOST_API) to limit ABI visibility [cccl-hide-from-abi]
// CHECK-MESSAGES-NOT: warning:
  struct local {
    local() = default;
    local(const local&) = default;
    local(local&&) = default;
    local& operator=(const local&) = default;
    local& operator=(local&&) = default;
    ~local() = default;
    explicit local(int) {}
    _CCCL_HOST void method() {}
    static void static_method() {}
    operator bool() const { return true; }
    struct nested {
      nested() = default;
      void method() {}
    };
  };
}

template <class T>
_CCCL_HOST_API inline auto enclosing_template(T value) {
  struct local {
    T value;
    explicit local(T value) : value(value) {}
    ~local() {}
    T get() const { return value; }
  };
  return local{value};
}
inline auto instantiated_local = enclosing_template(1);

inline auto enclosing_lambda = [] {
  struct local {
    local() = default;
    explicit local(int) {}
    ~local() {}
    void method() {}
    struct nested {
      nested() = default;
      void method() {}
    };
  };
  return local{1};
};
inline auto lambda_local = enclosing_lambda();

// CHECK-FIXES: DiagnosticName: cccl-hide-from-abi
// CHECK-FIXES: Replacements: []
// CHECK-FIXES-NOT: DiagnosticName:
// clang-format on
