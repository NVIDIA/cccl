//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/traits.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <memory> // for ::std::shared_ptr

namespace cuda::experimental::stf::reserved
{
/**
 * @brief Provides flags for instantiating the `handle` class (below).
 */
enum handle_flags : unsigned
{
  defaults, ///< The default `handle` flags; no special features.
  non_null, ///< Specifies that the pointer underlying the `handle` object cannot be null.
};

/// A tag for constructing `handle` objects with `static_cast`.
inline enum class use_static_cast {} use_static_cast = {};

/// A tag for constructing `handle` objects with `dynamic_cast`.
inline enum class use_dynamic_cast {} use_dynamic_cast = {};

/// @brief Performs bitwise OR operation on `handle_flags`.
/// @return The result of bitwise OR operation.
constexpr handle_flags operator|(handle_flags a, handle_flags b)
{
  return handle_flags(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

/// @brief Performs bitwise AND operation on `handle_flags`.
/// @return The result of bitwise AND operation.
constexpr handle_flags operator&(handle_flags a, handle_flags b)
{
  return handle_flags(static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

/**
 * @brief A handle to an object of type `T` with flags specified by `f`.
 *
 * The semantics of `handle` are similar to those of `std::shared_ptr` except as moted below.
 *
 * `handle` does not accept a `T*`argument for construction. It does accept another `handle` or a `shared_ptr`.
 * Additionally, `handle` offers a constructor that forwards all arguments to `T`'s constructor.
 *
 * If `f` includes `handle_flags::non_null`, the constructor eagerly initializes the underlying pointer with an
 * allocated and constructed object. Only moved-from objects will have a null pointer.
 *
 * For increased safety, `handle` enforces statically that `T` cannot be constructed outside of `handle`. This is
 * achieved on the user side either by making `T`'s constructors protected or by making them private and making `handle`
 * a friend of `T`.
 */
template <typename T, handle_flags f = handle_flags::defaults>
class handle
{
public:
  /// @name Defaulted constructors and assignment operators
  /// @{
  handle(handle&)                  = default;
  handle(const handle&)            = default;
  handle(handle&&)                 = default;
  handle& operator=(handle&)       = default;
  handle& operator=(const handle&) = default;
  handle& operator=(handle&&)      = default;
  /// @}

  /// @brief Default constructor.
  handle()
  {
    static_assert(!::std::is_constructible_v<T>, "T's default constructor must be protected.");
    if constexpr (f & handle_flags::non_null)
    {
      static_assert(!::std::is_abstract_v<T>,
                    "A non-nullable handle of an abstract type cannot have a default constructor.");
      impl = ::std::make_shared<Derived<T>>();
    }
  }

  /// @brief Constructs a handle from another handle.
  template <typename T1, handle_flags f1>
  handle(handle<T1, f1> rhs)
      : impl(mv(rhs.impl))
  {
    if constexpr (f & handle_flags::non_null)
    {
      static_assert(f1 & handle_flags::non_null, "Cannot initialize a non-nullable handle from a nullable one.");
    }
  }

  /// @brief Variadic template constructor for creating a handle.
  template <typename... Args>
  handle(Args&&... args)
      : impl(make(::std::forward<Args>(args)...))
  {
    static_assert(!::std::is_constructible_v<T, Args...>, "T's constructors must be protected.");
  }

  /// @brief Constructs a handle from another handle with static_cast.
  template <typename T1, handle_flags f1>
  handle(handle<T1, f1>& src, decltype(use_static_cast))
      : handle(const_cast<const handle<T1, f1>&>(src), use_static_cast)
  {}

  /// @brief Similar constructor as above but for const handle.
  template <typename T1, handle_flags f1>
  handle(const handle<T1, f1>& src, decltype(use_static_cast))
      : handle(::std::static_pointer_cast<T>(src.impl))
  {
    if constexpr (f & handle_flags::non_null)
    {
      EXPECT(src.impl, "Pointer of static type ", type_name<T1>, " was null upon construction of non-null handle.");
      assert(impl);
    }
  }

  /// @brief Constructs a handle from another handle with dynamic_cast.
  template <typename T1, handle_flags f1>
  handle(handle<T1, f1>& src, decltype(use_dynamic_cast))
      : handle(const_cast<const handle<T1, f1>&>(src), use_dynamic_cast)
  {}

  /// @brief Similar constructor as above but for const handle.
  template <typename T1, handle_flags f1>
  handle(const handle<T1, f1>& src, decltype(use_dynamic_cast))
      : handle(::std::dynamic_pointer_cast<T>(src.impl))
  {
    if constexpr (f & handle_flags::non_null)
    {
      EXPECT(src.impl, "Pointer of static type ", type_name<T1>, " was null upon construction of non-null handle.");
      EXPECT(impl, "dynamic_cast<", type_name<T>, "> failed for pointer of static type ", type_name<T1>);
    }
  }

  /// @brief Constructs a handle from a shared_ptr with dynamic_cast.
  template <typename T1>
  handle(const ::std::shared_ptr<T1>& src, decltype(use_dynamic_cast))
      : handle(::std::dynamic_pointer_cast<T>(src))
  {}

  /// @brief Assignment operator for assigning one handle to another.
  template <typename T1, handle_flags f1>
  handle& operator=(handle<T1, f1> rhs)
  {
    if constexpr (f & handle_flags::non_null)
    {
      static_assert(f1 & handle_flags::non_null, "Cannot assign a non-nullable handle from a nullable one.");
    }
    impl = mv(rhs.impl);
    return *this;
  }

  ~handle() = default;

  /// @brief Dereference operator.
  T* operator->() const
  {
    assert(*this);
    return impl.get();
  }

  /// @brief Dereference the handle to get the actual object.
  T& operator*()
  {
    return *operator->();
  }

  /// @brief Const version of the dereference operator.
  const T& operator*() const
  {
    return *operator->();
  }

  /// @brief Conversion operator to bool.
  explicit operator bool() const
  {
    return impl.get() != nullptr;
  }

  /// @brief Conversion operator to shared_ptr.
  template <typename T1>
  operator ::std::shared_ptr<T1>() const
  {
    return impl;
  }

  /// @brief Equality operator.
  template <typename T1, handle_flags f1>
  bool operator==(const handle<T1, f1>& rhs)
  {
    return impl == rhs.impl;
  }

  /// @brief Weak type corresponding to handle
  using weak_t = ::std::weak_ptr<T>;

  /// @brief Returns a weak pointer to the underlying object.
  weak_t weak() const
  {
    return impl;
  }

  /// @brief If weak_ptr is null, does nothing and returns false. Otherwise, calls the lambda and returns true.
  template <typename Fun>
  static bool if_valid(const weak_t& wp, Fun&& fun)
  {
    if (auto p = wp.lock())
    {
      handle h{mv(p)};
      ::std::forward<Fun>(fun)(mv(h));
      return true;
    }
    return false;
  }

private:
  // All instantiations of handle are friends with one another
  template <typename T1, handle_flags f1>
  friend class handle;

  // Define a derived class to access the protected ctor
  template <class U>
  struct Derived : public U
  {
    template <typename... Args>
    Derived(Args&&... args)
        : U(::std::forward<Args>(args)...)
    {}
  };

  template <typename Arg, typename... Args>
  static auto make(Arg&& arg, Args&&... args)
  {
    if constexpr (sizeof...(args) == 0 && ::std::is_convertible_v<Arg, ::std::shared_ptr<T>>)
    {
      return ::std::forward<Arg>(arg);
    }
    else
    {
      return ::std::make_shared<Derived<T>>(::std::forward<Arg>(arg), ::std::forward<Args>(args)...);
    }
  }

  ::std::shared_ptr<T> impl;
};

#ifdef UNITTESTED_FILE
UNITTEST("Weak handle")
{
  class test
  {
  protected:
    test(int x)
    {
      a = x;
    }

  public:
    int a;
  };
  handle<test> h(42);
  EXPECT(h->a == 42);
  auto w = h.weak();
  handle<test>::if_valid(w, [](handle<test> x) {
    x->a++;
  });
  EXPECT(h->a == 43);
};
#endif // UNITTESTED_FILE
} // namespace cuda::experimental::stf::reserved
