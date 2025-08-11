//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 * @brief Execution policy header
 */
#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <cassert>
#include <variant>

namespace cuda::experimental::stf
{

/**
 * @brief Typed numeric for representing memory allocation size (in bytes) for the `thread_hierarchy_spec` API below.
 * Use `mem(n)` to pass memory size requirements to the `par()` and `con()` family of functions.
 *
 */
enum class mem : size_t
{
};

/**
 * @brief Describes the hardware scope for a given synchronization operation.
 *
 * This `enum class` defines various hardware scopes like `none`, `device`, `block`,
 * `thread`, and `all`, which are useful for specifying the level of granularity
 * at which an operation should occur.
 *
 * Note that we use powers of two so that it is easy to implement | or & operators.
 */
enum class hw_scope : unsigned int
{
  none   = 0, ///< No hardware scope.
  thread = 1, ///< Thread-level scope.
  block  = 2, ///< Block-level scope.
  device = 4, ///< Device-level scope.
  all    = 7 ///< All levels of hardware scope.
};

/**
 * @brief Bitwise OR operator for combining two `hw_scope` values.
 *
 * This function performs bitwise OR on the underlying types of two `hw_scope`
 * values. It's useful for combining multiple hardware scopes into a single
 * `hw_scope` value.
 *
 * @param lhs The left-hand side hw_scope.
 * @param rhs The right-hand side hw_scope.
 * @return A new hw_scope that is the bitwise OR of lhs and rhs.
 *
 * @note The function includes assertions to ensure that lhs and rhs are
 * within the allowed range.
 */
inline hw_scope operator|(const hw_scope& lhs, const hw_scope& rhs)
{
  assert(as_underlying(lhs) <= as_underlying(hw_scope::all));
  assert(as_underlying(rhs) <= as_underlying(hw_scope::all));
  return hw_scope(as_underlying(lhs) | as_underlying(rhs));
}

/**
 * @brief Bitwise AND operator for combining two `hw_scope` values.
 *
 * This function performs bitwise AND on the underlying types of two `hw_scope`
 * values. It's useful for checking if a scope if included into another `hw_scope` value.
 *
 * @param lhs The left-hand side hw_scope.
 * @param rhs The right-hand side hw_scope.
 * @return A new hw_scope that is the bitwise AND of lhs and rhs.
 *
 * @note The function includes assertions to ensure that lhs and rhs are
 * within the allowed range.
 */
inline hw_scope operator&(const hw_scope& lhs, const hw_scope& rhs)
{
  assert(as_underlying(lhs) <= as_underlying(hw_scope::all));
  assert(as_underlying(rhs) <= as_underlying(hw_scope::all));
  return hw_scope(as_underlying(lhs) & as_underlying(rhs));
}

//! @brief Bitwise NOT operator for inverting a `hw_scope` value.
//!
//! This function performs a bitwise NOT operation on the underlying type of a `hw_scope` value.
//! It is useful for computing the complement of a given scope, ensuring that only valid bits
//! within the range of `hw_scope::all` are set in the result.
//!
//! @param s The hw_scope value to invert.
//! @return A new hw_scope representing the bitwise complement of s, masked to valid bits.
//!
//! @note The function includes an assertion to ensure that s is within the allowed range.
//!       Any bits beyond those represented by `hw_scope::all` are cleared in the result.
inline hw_scope operator~(const hw_scope& s)
{
  assert(as_underlying(s) <= as_underlying(hw_scope::all));
  // Keep the bits beyond `all` zero at all times, and
  auto x = ~as_underlying(s) & as_underlying(hw_scope::all);
  return hw_scope(x);
}

template <auto... level_spec>
class thread_hierarchy_spec;

namespace reserved
{

template <bool, size_t, typename...>
struct deduce_execution_policy;

template <bool b, size_t s>
struct deduce_execution_policy<b, s>
{
  using type = thread_hierarchy_spec<b, s>;
};

template <bool b, size_t s, typename T, typename... Ts>
struct deduce_execution_policy<b, s, T, Ts...>
{
  using type = typename deduce_execution_policy<b, s, Ts...>::type;
};

template <bool b, size_t s, auto... P, typename... Ts>
struct deduce_execution_policy<b, s, thread_hierarchy_spec<P...>, Ts...>
{
  static_assert(::std::is_same_v<typename deduce_execution_policy<b, s, Ts...>::type, thread_hierarchy_spec<b, s>>,
                "Only one argument of type deduce_execution_policy<...> is allowed.");
  using type = thread_hierarchy_spec<b, s, P...>;
};

} // namespace reserved

template <auto... spec>
class thread_hierarchy;

template <>
class thread_hierarchy_spec<>
{};

/**
 * @brief A template class for specifying a thread hierarchy.
 * @tparam can_sync A boolean indicating if synchronization is possible.
 * @tparam width The width of the thread group at this level (0 to set it dynamically).
 * @tparam lower_levels Further specifications for lower levels of the thread hierarchy.
 */
template <bool can_sync, size_t width, auto... lower_levels>
class thread_hierarchy_spec<can_sync, width, lower_levels...>
{
  static_assert(sizeof...(lower_levels) % 2 == 0, "Must specify level specifications as pairs of bool and size_t.");

public:
  using thread_hierarchy_t = thread_hierarchy<can_sync, width, lower_levels...>;

  ///@{ @name Usual constructors
  thread_hierarchy_spec()                             = default;
  thread_hierarchy_spec(const thread_hierarchy_spec&) = default;
  thread_hierarchy_spec(thread_hierarchy_spec&)       = default;
  thread_hierarchy_spec(thread_hierarchy_spec&&)      = default;
  ///@}

  /**
   * @brief Constructor with variadic parameters (usually no needed; use `par` and `con` below instead)
   *
   * Parameters are used for initializing `this` depending on their types as follows:
   * - If type is `thread_hierarchy_spec<lower_levels...>`, parameter is used to initialize the inner part
   * - If type is `size_t`, parameter is used to initialize the width of the object
   * - If type is `mem`, parameter is used to initialize the memory (bytes) attribute of the object
   * - If type is `hw_scope`, parameter is used to initialize the scope attribute of the object
   *
   * Parameters can be specified in any order. All are optional. It is illegal to pass the same parameter type more
   * than once.
   *
   * @param P... Types of parameters
   * @param p Values of parameters
   */
  template <typename... P>
  explicit thread_hierarchy_spec(const P&... p)
      : inner(reserved::only_convertible_or(thread_hierarchy_spec<lower_levels...>(), p...))
      , dynamic_width(reserved::only_convertible_or(decltype(dynamic_width)(), p...))
      , sync_scope(reserved::only_convertible_or(hw_scope::all, p...))
      , mem_bytes(reserved::only_convertible_or(mem(0), p...))
  {
    shuffled_args_check<P...>(inner, dynamic_width, sync_scope, mem_bytes);
    if constexpr (sizeof...(lower_levels) > 0)
    {
      // Unclear about the logic here.
      if (inner.get_scope(0) != hw_scope::all)
      {
        sync_scope = sync_scope & ~inner.get_scope(0);
      }
    }
  }

  constexpr bool operator==(const thread_hierarchy_spec& rhs) const noexcept
  {
    if (dynamic_width != rhs.dynamic_width || sync_scope != rhs.sync_scope || mem_bytes != rhs.mem_bytes)
    {
      return false;
    }
    if constexpr (sizeof...(lower_levels) > 0)
    {
      return inner == rhs.inner;
    }
    else
    {
      return true;
    }
  }

  // For other types ...
  template <auto... other,
            std::enable_if_t<!std::is_same_v<thread_hierarchy_spec, thread_hierarchy_spec<other...>>, int> = 0>
  constexpr bool operator==(const thread_hierarchy_spec<other...>&) const noexcept
  {
    return false;
  }

  constexpr bool operator!=(const thread_hierarchy_spec& rhs) const noexcept
  {
    return !(*this == rhs);
  }

  template <auto... other,
            std::enable_if_t<!std::is_same_v<thread_hierarchy_spec, thread_hierarchy_spec<other...>>, int> = 0>
  constexpr bool operator!=(const thread_hierarchy_spec<other...>&) const noexcept
  {
    return true;
  }

  /// @brief Compute the depth of the thread hierarchy.
  /// @return The depth.
  static constexpr size_t depth()
  {
    return 1 + sizeof...(lower_levels) / 2;
  }

  /// @brief Check if synchronization is possible.
  /// @return A boolean indicating if synchronization is possible.
  static constexpr bool synchronizable()
  {
    return can_sync;
  }

  /**
   * @brief Set the inner thread hierarchy.
   * @tparam P Template arguments of the inner thread_hierarchy_spec object.
   * @param inner The inner thread_hierarchy_spec object.
   */
  template <auto... P>
  void set_inner(const thread_hierarchy_spec<P...>& inner)
  {
    this->inner = inner;
  }

  /**
   * @brief Get the memory bytes at a specific level.
   * @param level The level.
   * @return The memory bytes.
   */
  constexpr mem get_mem(size_t level) const
  {
    if constexpr (depth() > 1)
    {
      if (level > 0)
      {
        return inner.get_mem(level - 1);
      }
    }
    return mem_bytes;
  }

  /**
   * @brief Set the memory bytes at a specific level.
   * @param level The level.
   * @param value The memory bytes.
   */
  constexpr void set_mem(size_t level, mem value)
  {
    if constexpr (depth() > 1)
    {
      if (level > 0)
      {
        return inner.set_mem(level - 1, value);
      }
    }
    mem_bytes = value;
  }

  /**
   * @brief Checks if the given `thread_hierarchy_spec` is synchronizable at the given `level`.
   *
   * @tparam level The level in the hierarchy to check for the `sync` property. Level starts from 0 (top-level).
   */
#ifndef _CCCL_DOXYGEN_INVOKED // doxygen fails to parse this
  template <size_t level>
  static inline constexpr bool is_synchronizable = [] {
    if constexpr (level > 0)
    {
      return thread_hierarchy_spec<lower_levels...>::template is_synchronizable<level - 1>;
    }
    else
    {
      return can_sync;
    }
  }();
#else
  template <size_t level>
  static inline constexpr bool is_synchronizable;
#endif

  /**
   * @brief Get the statically-specified width at a specific level
   * @param level The level
   * @return The width (0 if width is dynamic)
   */
  static inline constexpr size_t static_width(size_t level)
  {
    size_t data[] = {width, lower_levels...};
    return data[2 * level];
  }

  /**
   * @brief Get the width at a specific level.
   * @param level The level.
   * @return The width.
   */
  constexpr size_t get_width(size_t level) const
  {
    if constexpr (depth() > 1)
    {
      if (level > 0)
      {
        return inner.get_width(level - 1);
      }
    }
    return dynamic_width.get();
  }

  /**
   * @brief Set the width.
   * @param new_width The new width.
   */
  void set_width(size_t level, size_t new_width)
  {
    if constexpr (depth() > 1)
    {
      if (level > 0)
      {
        return inner.set_width(level - 1, new_width);
      }
    }
    if constexpr (width == 0)
    {
      dynamic_width = new_width;
    }
    else
    {
      assert(!"Cannot set width at this level.");
    }
  }

  /**
   * @brief Get the scope at a specific level
   */
  constexpr hw_scope get_scope(size_t level) const
  {
    if constexpr (depth() > 1)
    {
      if (level > 0)
      {
        return inner.get_scope(level - 1);
      }
    }
    return sync_scope;
  }

private:
  /// @brief The inner thread hierarchy.
  [[no_unique_address]] thread_hierarchy_spec<lower_levels...> inner;
  /// @brief The dynamic width, if applicable.
  [[no_unique_address]] optionally_static<width, 0> dynamic_width;
  /// @brief Synchronization level(s)
  hw_scope sync_scope = hw_scope::none;
  /// @brief The memory bytes.
  mem mem_bytes = mem(0);
};

#ifndef _CCCL_DOXYGEN_INVOKED // doxygen fails to parse this
/**
 * @brief Creates and returns a `thread_hierarchy_spec` object with no synchronization and dynamic width.
 *
 * Parameters are used for initializing the result depending on their types as follows:
 * - If type is `thread_hierarchy_spec<lower_levels...>`, parameter is used to initialize the inner part
 * - If type is `size_t`, parameter is used to initialize the width of the object
 * - If type is `mem`, parameter is used to initialize the memory (bytes) attribute of the object
 * - If type is `hw_scope`, parameter is used to initialize the scope attribute of the object
 *
 * @tparam P... Types of parameters
 * @param p Values of parameters
 * @return A `thread_hierarchy_spec` instantiation with the appropriate arguments.
 */
template <typename... P>
constexpr auto par(const P&... p)
{
  using R = typename reserved::deduce_execution_policy<false, size_t(0), P...>::type;
  return R(p...);
}

/**
 * @brief Creates and returns a `thread_hierarchy_spec` object with no synchronization and static width.
 *
 * @overload
 *
 * Parameters are used for initializing the result depending on their types as follows:
 * - If type is `thread_hierarchy_spec<lower_levels...>`, parameter is used to initialize the inner part
 * - If type is `mem`, parameter is used to initialize the memory (bytes) attribute of the object
 * - If type is `hw_scope`, parameter is used to initialize the scope attribute of the object
 *
 * @tparam width level width
 * @tparam P... Types of parameters
 * @param p Values of parameters
 * @return A `thread_hierarchy_spec` instantiation with the appropriate arguments.
 */
template <size_t width, typename... P>
constexpr auto par(const P&... p)
{
  using R = typename reserved::deduce_execution_policy<false, width, P...>::type;
  return R(p...);
}

/// @{
/**
 * @brief Creates and returns a `thread_hierarchy_spec` object with synchronization and dynamic width.
 *
 * Parameters are used for initializing the result depending on their types as follows:
 * - If type is `thread_hierarchy_spec<lower_levels...>`, parameter is used to initialize the inner part
 * - If type is `size_t`, parameter is used to initialize the width of the object
 * - If type is `mem`, parameter is used to initialize the memory (bytes) attribute of the object
 * - If type is `hw_scope`, parameter is used to initialize the scope attribute of the object
 *
 * @param P... Types of parameters
 * @param p Values of parameters
 * @return A `thread_hierarchy_spec` instantiation with the appropriate arguments.
 */
template <typename... P>
constexpr auto con(const P&... p)
{
  using R = typename reserved::deduce_execution_policy<true, size_t(0), P...>::type;
  return R(p...);
}

/**
 * @brief Creates and returns a `thread_hierarchy_spec` object with synchronization and static width.
 *
 * @overload
 *
 * Parameters are used for initializing the result depending on their types as follows:
 * - If type is `thread_hierarchy_spec<lower_levels...>`, parameter is used to initialize the inner part
 * - If type is `mem`, parameter is used to initialize the memory (bytes) attribute of the object
 * - If type is `hw_scope`, parameter is used to initialize the scope attribute of the object
 *
 * @param P... Types of parameters
 * @param p Values of parameters
 * @return A `thread_hierarchy_spec` instantiation with the appropriate arguments.
 */
template <size_t width, typename... P>
constexpr auto con(const P&... p)
{
  using R = typename reserved::deduce_execution_policy<true, width, P...>::type;
  return R(p...);
}
/// @}
#endif // _CCCL_DOXYGEN_INVOKED

#ifdef UNITTESTED_FILE

// clang-format off
UNITTEST("par") {
    {
        auto x = par();
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0)>>);
        static_assert(x.depth() == 1);
        static_assert(!thread_hierarchy_spec<false, size_t(0)>::template is_synchronizable<0>);
        static_assert(x.static_width(0) == 0);
        ::std::ignore = x;
    }

    {
        auto x = par<1024>();
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(1024)>>);
        static_assert(x.depth() == 1);
        ::std::ignore = x;
    }

    {
        auto x = par(1024);
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0)>>);
        static_assert(x.depth() == 1);
        ::std::ignore = x;
        assert(x.get_width(1) == 1024);
    }

    {
        auto x = par(par());
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0), false, size_t(0)>>);
        static_assert(x.depth() == 2);
        static_assert(!thread_hierarchy_spec<false, size_t(0), false, size_t(0)>::template is_synchronizable<0>);
        static_assert(!thread_hierarchy_spec<false, size_t(0), false, size_t(0)>::template is_synchronizable<1>);
        ::std::ignore = x;
    }

    {
        auto x = par<1024>(par());
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(1024), false, size_t(0)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 1024);
        ::std::ignore = x;
    }

    {
        auto x = par(par(), 512);
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0), false, size_t(0)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 512);
        ::std::ignore = x;
    }

    {
        auto x = par(par(), mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0), false, size_t(0)>>);
        static_assert(x.depth() == 2);
        ::std::ignore = x;
    }

    {
        auto x = par<256>(par(), mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(256), false, size_t(0)>>);
        static_assert(x.depth() == 2);
        ::std::ignore = x;
    }

    {
        auto x = par(par<256>(), 1024, mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<false, size_t(0), false, size_t(256)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 1024);
        assert(x.get_width(1) == 256);
        assert(x.get_mem(0) == mem(512));
        ::std::ignore = x;
    }
};

UNITTEST("con") {
    {
        auto x = con();
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0)>>);
        static_assert(x.depth() == 1);
        static_assert(thread_hierarchy_spec<true, size_t(0)>::template is_synchronizable<0>);
        ::std::ignore = x;
    }

    {
        auto x = con<1024>();
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(1024)>>);
        static_assert(x.depth() == 1);
        ::std::ignore = x;
    }

    {
        auto x = con(1024);
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0)>>);
        static_assert(x.depth() == 1);
        ::std::ignore = x;
        assert(x.get_width(1) == 1024);
    }

    {
        auto x = con(con());
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0), true, size_t(0)>>);
        static_assert(x.depth() == 2);
        ::std::ignore = x;
    }

    {
        auto x = con<1024>(con());
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(1024), true, size_t(0)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 1024);
        ::std::ignore = x;
    }

    {
        auto x = con(con(), 512);
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0), true, size_t(0)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 512);
        ::std::ignore = x;
    }

    {
        auto x = con(con(), mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0), true, size_t(0)>>);
        static_assert(x.depth() == 2);
        ::std::ignore = x;
    }

    {
        auto x = con<256>(con(), mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(256), true, size_t(0)>>);
        static_assert(x.depth() == 2);
        ::std::ignore = x;
    }

    {
        auto x = con(con<256>(), 1024, mem(512));
        static_assert(::std::is_same_v<decltype(x), thread_hierarchy_spec<true, size_t(0), true, size_t(256)>>);
        static_assert(x.depth() == 2);
        assert(x.get_width(0) == 1024);
        assert(x.get_width(1) == 256);
        assert(x.get_mem(0) == mem(512));
        ::std::ignore = x;
    }
};
// clang-format on

// These trigger a segfault in nvcc 12.9. Temporarily disabling until they can be investigated.
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 9)
// unittest for core.h that can't be there
UNITTEST("optionally_static")
{
  optionally_static<size_t(42), 0> v1;
  static_assert(v1.get() == 42);
  static_assert(v1 == v1);
  static_assert(v1 == 42UL);

  optionally_static<size_t(43), 0> v2;
  static_assert(v2.get() == 43UL);

  optionally_static<size_t(0), 0> v3;
  EXPECT(v3.get() == 0);
  v3 = 44;
  EXPECT(v3.get() == 44UL);

#    if 0
  // TODO clarify these tests !

  // Make sure the size is optimized properly
  struct S1
  {
    [[no_unique_address]] optionally_static<size_t(42)> x;
    int y;
  };
  static_assert(sizeof(S1) == sizeof(int));
  struct S2
  {
    int y;
    [[no_unique_address]] optionally_static<size_t(42)> x;
  };
  static_assert(sizeof(S1) == sizeof(int));
#    endif

  // Multiplication
  static_assert(v1 * v1 == 42UL * 42UL);
  static_assert(v1 * v2 == 42UL * 43UL);
  static_assert(v1 * 44 == 42UL * 44UL);
  static_assert(44 * v1 == 42UL * 44UL);
  EXPECT(v1 * v3 == 42 * 44);

  // Odds and ends
  optionally_static<3, 18> v4;
  optionally_static<6, 18> v5;
  static_assert(v4 * v5 == 18UL);
  static_assert(v4 * v5 == (optionally_static<18, 18>(18)));

// TODO solve these there are some ambiguities !
#    if 0
  // Mutating operators
  optionally_static<1, 1> v6;
  v6 += v6;
  EXPECT(v6 == 0);
  v6 += 2;
  EXPECT(v6 == 2);
  v6 -= 1;
  EXPECT(v6 == 1);
  v6++;
  ++v6;
  EXPECT(v6 == 3);
  --v6;
  v6--;
  EXPECT(v6 == 1);
  EXPECT(-v6 == -1);
#    endif
};
#  endif

UNITTEST("thread hierarchy spec equality")
{
  EXPECT(par() == par());
  EXPECT(con() == con());
  EXPECT(con<128>() == con<128>());
  EXPECT(con(128) == con(128));

  static_assert(con<128>() == con<128>());
  static_assert(con<128>() != con<64>());
  static_assert(con() != par());

  EXPECT(par() != con());

  EXPECT(par(par<256>(), 1024, mem(512)) == par(par<256>(), 1024, mem(512)));

  // Change one of the par to con
  EXPECT(par(par<256>(), 1024, mem(512)) != par(con<256>(), 1024, mem(512)));

  EXPECT(con(mem(512)) != con());
  EXPECT(con(mem(512)) != con(mem(128)));
};

UNITTEST("spec with scope")
{
  auto spec = par(hw_scope::device | hw_scope::block, par<128>(hw_scope::thread));

  EXPECT(spec.get_scope(0) == (hw_scope::device | hw_scope::block));
  EXPECT(spec.get_scope(1) == hw_scope::thread);
};

UNITTEST("spec with scope and implicit scopes")
{
  auto spec = par(par<128>(hw_scope::thread));

  EXPECT(spec.get_scope(1) == hw_scope::thread);

  // Since the second level is bound to the thread scope, level 0 should be
  // bound to blocks and devices instead.
  EXPECT((spec.get_scope(0) & hw_scope::thread) == hw_scope::none);
};

UNITTEST("thread hierarchy spec equality")
{
  EXPECT(par() == par());
  EXPECT(con() == con());
  EXPECT(con<128>() == con<128>());
  EXPECT(con(128) == con(128));

  static_assert(con<128>() == con<128>());
  static_assert(con<128>() != con<64>());
  static_assert(con() != par());

  EXPECT(par() != con());

  EXPECT(par(par<256>(), 1024, mem(512)) == par(par<256>(), 1024, mem(512)));

  // Change one of the par to con
  EXPECT(par(par<256>(), 1024, mem(512)) != par(con<256>(), 1024, mem(512)));

  EXPECT(con(mem(512)) != con());
  EXPECT(con(mem(512)) != con(mem(128)));
};

UNITTEST("spec with scope")
{
  auto spec = par(hw_scope::device | hw_scope::block, par<128>(hw_scope::thread));

  EXPECT(spec.get_scope(0) == (hw_scope::device | hw_scope::block));
  EXPECT(spec.get_scope(1) == hw_scope::thread);
};

#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
