//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Unit testing framework used in CUDASTF
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

#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/traits.cuh>

#include <filesystem>

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
// One level of macro indirection is required in order to resolve __COUNTER__,
// and get varname1 instead of varname__COUNTER__.
#  define _55f56f4e3b45c8cf3fa50b28fed72e2a(a, b) _a56ec7069122ad2e0888a508ecdc4639(a, b)
#  define _a56ec7069122ad2e0888a508ecdc4639(a, b) a##b

/**
 * @brief A macro that creates a unique identifier based on a given base name.
 *
 * This macro generates a unique identifier by concatenating the base name with the value of the
 * `__COUNTER__` macro. The `__COUNTER__` macro is a non-standard extension that is supported by
 * some compilers, such as GCC and Clang, and it expands to a unique integer value within the
 * translation unit.
 *
 * @code
 * // Example usage:
 * int CUDASTF_UNIQUE_NAME(my_variable) = 42;  // Creates a unique variable name like my_variable0
 * int CUDASTF_UNIQUE_NAME(my_variable) = 43;  // Creates a unique variable name like my_variable1
 * @endcode
 *
 * @note This macro is not portable as the `__COUNTER__` macro is not part of the C++ standard.
 *
 * @param base The base name for the unique identifier.
 */
#  define CUDASTF_UNIQUE_NAME(base) _55f56f4e3b45c8cf3fa50b28fed72e2a(base, __COUNTER__)

/**
 * @brief Macro to check a variety of conditions, throws if the condition is false
 *
 * @param ... condition to check, followed by zero or more messages to be made part of the text of the exception thrown
 * if the condition is false. Messages are concatenated.
 *
 * @return If condition is a comparison expression, returns the left-hand side operand, otherwise returns a copy of the
 * condition
 *
 * The condition and all optional error messages are evaluated exactly once.
 *
 * The file and line of the invocation are made part of the exception message, in a format similar to compiler error
 * messages (recognizable by tools such as emacs, Visual Studio Code, etc.). In case of a failing comparison, the
 * exception thrown will contain information about the left-hand side and right-hand side values. For example,
 * `EXPECT(a == b)` will throw an exception that makes the values of `a` and `b` part of the textual content of the
 * exception object (accessible by calling `what()`).
 *
 * @paragraph example Example
 * @snippet this EXPECT
 */
#  define EXPECT(...)                              \
    ::cuda::experimental::stf::expecter::validate( \
      ::cuda::std::source_location::current(), ::cuda::experimental::stf::expecter()->*__VA_ARGS__)

namespace cuda::experimental::stf
{

/*
 * @brief Entry point for all comparisons (equality, inequality, and ordering) tested.
 *
 */
struct expecter
{
  /*
   * @brief Exception embedding location information.
   *
   */
  struct failure : ::std::runtime_error
  {
    template <typename... Msgs>
    failure(::cuda::std::source_location loc, const Msgs&... msgs)
        : ::std::runtime_error(text(loc, msgs...))
    {}

    template <typename... Msgs>
    static ::std::string text(::cuda::std::source_location loc, const Msgs&... msgs)
    {
      ::std::stringstream s;
      s << loc.file_name() << '(' << loc.line() << "): ";
      (stream(s, msgs), ...);
      auto result = s.str();
      if (!result.empty() && result.back() != '\n')
      {
        result += '\n';
      }
      return result;
    }

  private:
    template <typename T>
    static void stream(::std::stringstream& s, const T& x)
    {
      if constexpr (reserved::has_ostream_operator<T>::value)
      {
        s << x;
      }
      else
      {
        s << type_name<T> << "{?}";
      }
    }
  };

  template <typename... Msgs>
  [[noreturn]] static _CCCL_API inline void
  __throw_stf_failure([[maybe_unused]] ::cuda::std::source_location loc, [[maybe_unused]] const Msgs&... msgs)
  {
#  if _CCCL_HAS_EXCEPTIONS()
    NV_IF_ELSE_TARGET(NV_IS_HOST, (throw failure(loc, msgs...);), (::cuda::std::terminate();))
#  else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
    ::cuda::std::terminate();
#  endif // !_CCCL_HAS_EXCEPTIONS()
  }

  /*
   * @brief Wrapper for a comparison operation using one of `==`, `!=`, `<`, `>`, `<=`, `>=`. Includes the result and
   * the operands.
   *
   * @tparam T Type of the left-hand side operator (qualifiers and reference included)
   * @tparam U Type of the right-hand side operator (qualifiers and reference included)
   */
  template <typename L, typename R>
  struct comparison_expression
  {
    bool value;
    L lhs;
    R rhs;
    const char* op;
    comparison_expression(bool value, L lhs, R rhs, const char* op)
        : value(value)
        , lhs(::std::forward<L>(lhs))
        , rhs(::std::forward<R>(rhs))
        , op(op)
    {}
  };

  /*
   * @brief Wraps a single term (the left-hand side one) in a comparison expression.
   *
   * @tparam T type of the term, may include a reference or a const reference etc.
   */
  template <typename T>
  struct term
  {
    T value;

    template <typename U>
    term(U&& value)
        : value(::std::forward<U>(value))
    {}

    /*
     * All comparison operators are defined assuming `const&` inputs. They are evaluated eagerly and return a
     * `compariso_expression` object with information necessary for constructing an error message, if any.
     */
#  define _9d10c7e37932af3c4f39a5ce7ff00b5a(op)                                            \
    template <typename U>                                                                  \
    auto operator op(const U& rhs) &&                                                      \
    {                                                                                      \
      using T_noref = std::remove_reference_t<T>;                                          \
      using U_noref = std::remove_reference_t<U>;                                          \
      if constexpr (std::is_integral_v<T_noref> && std::is_integral_v<U_noref>             \
                    && std::is_signed_v<T_noref> != std::is_signed_v<U_noref>)             \
      {                                                                                    \
        using CommonType = std::common_type_t<T_noref, U_noref>;                           \
        const bool r     = static_cast<CommonType>(value) op static_cast<CommonType>(rhs); \
        return comparison_expression<T, const U&>(r, std::forward<T>(value), rhs, #op);    \
      }                                                                                    \
      else                                                                                 \
      {                                                                                    \
        /* Direct comparison for other cases */                                            \
        const bool r = value op rhs;                                                       \
        return comparison_expression<T, const U&>(r, std::forward<T>(value), rhs, #op);    \
      }                                                                                    \
    }

    _9d10c7e37932af3c4f39a5ce7ff00b5a(==);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(!=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(<);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(>);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(<=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(>=);

#  undef _9d10c7e37932af3c4f39a5ce7ff00b5a

    /*
     * Operators `^`, `|`, `*`, `/`, `%`, `+`, `-`, `>>`, and `<<` are defined only for carrying the actual
     * computation alongside with the `term` wrapper. For example, in the expression `EXPECT(a+b==c)`, which expands
     * to `expecter()->*a+b==c`, the operator+ defined below will make sure `a+b` is correctly computed before
     * evaluating
     * `==`.
     */
#  define _9d10c7e37932af3c4f39a5ce7ff00b5a(op)                                \
    template <typename U>                                                      \
    auto operator op(U&& rhs) &&                                               \
    {                                                                          \
      using Result = decltype(value op ::std::forward<U>(rhs));                \
      return term<Result>(::std::forward<T>(value) op ::std::forward<U>(rhs)); \
    }

    _9d10c7e37932af3c4f39a5ce7ff00b5a(*);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(/);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(%);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(+);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(-);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(>>);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(<<);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(&);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(^);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(|);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(+=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(-=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(*=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(/=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(%=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(<<=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(>>=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(&=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(^=);
    _9d10c7e37932af3c4f39a5ce7ff00b5a(|=);

#  undef _9d10c7e37932af3c4f39a5ce7ff00b5a

    /*
     * Operators `&&`, `||`, `,` do not obey order of evaluation and are disallowed at top level. For example you
     * cannot write `EXPECT(a && b)` - instead, you could write `EXPECT((a && b))` but you lose information about
     * which side of the conjunction failed.
     */
    template <typename U>
    auto operator&&(U&&) = delete;
    template <typename U>
    auto operator||(U&&) = delete;
    template <typename U>
    auto operator,(U&&) = delete;
  };

  /**
   * @brief Initiates a test, use like this: `test ->* a == b;`
   *
   * @tparam T The type of the initial left-hand side operand
   * @param lhs The laft-hand operand, will be saved.
   * @return auto
   */
  template <typename T>
  term<T> operator->*(T&& lhs)
  {
    return term<T>(::std::forward<T>(lhs));
  }

  /*
   * @brief Validates a single-term expression, as in `EXPECT(v.empty())`.
   *
   * @tparam T the term's type including qualifiers
   * @param loc source location for error reporting
   * @param t term value, an exception will be thrown if non-true
   */
  template <typename T, typename... Msgs>
  static decltype(auto) validate(::cuda::std::source_location loc, term<T>&& t, const Msgs&... msgs)
  {
    if (t.value)
    {
      return ::std::forward<T>(t.value);
    }
    if constexpr (sizeof...(msgs) == 0)
    {
      using U = ::std::remove_reference_t<T>;
      __throw_stf_failure(
        loc,
        "Tested expression of type " + ::std::string(type_name<T>) + " is "
          + (std::is_same_v<const U, const bool> ? "false"
             : ::std::is_arithmetic_v<U>         ? "zero"
                                                 : "null")
          + ".\n");
    }
    else
    {
      __throw_stf_failure(loc, msgs...);
    }
  }

  /*
   * @brief Validates a comparison using `==`, `!=` etc.
   *
   * @tparam L type of left-hand side operand
   * @tparam R type of left-hand side operand
   * @param loc source code location
   * @param e expression, an exception will be thrown is `e.value` is false
   */
  template <typename L, typename R, typename... Msgs>
  static decltype(auto) validate(::cuda::std::source_location loc, comparison_expression<L, R>&& e, const Msgs&... msgs)
  {
    if (e.value)
    {
      return ::std::forward<L>(e.lhs);
    }
    __throw_stf_failure(loc, e.lhs, ' ', e.op, ' ', e.rhs, " is false.\n", msgs...);
  }
};

/*
 * @brief Small object that runs a unittest
 *
 * Typically not used directly; just use `UNITTEST` (see below) to define and run tests.
 *
 * Also includes static globals for test assertions and for counting pass/fail unittests.
 */
template <typename...>
class unittest;

/*
 * @brief Parameterless unittest implementation
 *
 * Typically not used directly; just use `UNITTEST` (see below) to define and run tests.
 *
 * Most unittests are not parameterized. In addition, all parameterized unittests ultimately inherit `unittest<>` so
 * this class has most of the unittest paraphernalia within.
 */
template <>
class unittest<>
{
public:
  /*
   * @brief Constructor (to be used with the `UNITTEST` macro) taking a name and the current location.
   *
   * @param name name of the unittest, e.g. `UNITTEST("name goes here") { ... };`
   * @param loc current location of the use of `UNITTEST`, automatically provided by the `UNITTEST` macro.
   */
  unittest(const char* name, ::cuda::std::source_location loc)
      : name(name)
      , loc(mv(loc))
  {}

  /*
   * @brief Factory method that creates a (potentially parameterized) unittest from variadic arguments.
   *
   * @tparam Params types of unittest parameters, if any (multiple types are allowed)
   * @param name name of the unittest
   * @param loc code location
   * @param params unittest parameters, if any
   * @return unittest<Params...> the unittest object with inferred type and properly initialized
   *
   * This function is not used directly. Instead, the invocation `UNITTEST("name", 1, 2.2)` calls
   * `unittest<>::make("name", cuda::std::source_location::current(), 1, 2,2)`.
   */
  template <typename... Params>
  static unittest<Params...> make(const char* name, ::cuda::std::source_location loc, Params&&... params)
  {
    return unittest<Params...>(name, loc, ::std::forward<Params>(params)...);
  }

public:
  /*
   * @brief Unittest entry point, to be used with the `UNITTEST` macro.
   *
   * @tparam Fun type of lambda containing the unittest
   * @param fun lambda containing the unittest
   * @return unittest& `*this`
   */
  template <typename Fun>
  unittest& operator->*([[maybe_unused]] Fun&& fun)
  {
#  ifdef UNITTESTED_FILE
    if (!::std::filesystem::equivalent(loc.file_name(), UNITTESTED_FILE))
    {
      return *this;
    }
    ++total();
    try
    {
      fun(name);
      ++passed();
      // fprintf(stdout, "%s(%zu): PASS(%s)\n", filename.c_str(), line, name);
    }
    catch (const expecter::failure& e)
    {
      fprintf(stderr, "%sUNITTEST FAILURE: %s\n", e.what(), name);
    }
    catch (const ::std::exception& e)
    {
      fprintf(stderr, "%s(%u): %s\nUNITTEST FAILURE: %s\n", loc.file_name(), loc.line(), e.what(), name);
    }
#  endif
    return *this;
  }

public:
  static size_t& passed()
  {
    static size_t result;
    return result;
  }

  static size_t& total()
  {
    static size_t result;
    return result;
  }

protected:
  const char* const name;
  const ::cuda::std::source_location loc;
};

/*
 * @brief Implementation of `unittest` for one or more parameters.
 *
 * @tparam Param first parameter type (may include qualifiers and adornments)
 * @tparam Params other parameters, if any (may include qualifiers and adornments) (may be empty)
 *
 * The hierarchy is set up in a linear fashion: `unittest<int, double>` inherits `unittest<double>` which in turn
 * inherits `unittest<>`.
 */
template <typename Param, typename... Params>
class unittest<Param, Params...> : public unittest<Params...>
{
public:
  /**
   * @brief Constructor invoked by the `UNITTEST` macro.
   *
   * @param name name of the unittest
   * @param loc code location of the invocation
   * @param param first parameter, is saved in this object and forwarded to the unittest lambda
   * @param params other parameters, if any
   */
  unittest(const char* name, ::cuda::std::source_location loc, Param param, Params... params)
      : unittest<Params...>(name, loc, ::std::forward<Params>(params)...)
      , param(::std::forward<Param>(param))
  {}

  /**
   * @brief Launch the given unittest lambda
   *
   * @tparam Fun Lambda type
   * @param fun lambda object
   * @return unittest& `*this`
   */
  template <typename Fun>
  unittest& operator->*(Fun&& fun)
  {
    unittest<>::operator->*([&](const char* name) {
      fun(name, ::std::forward<Param>(param));
    });
    if constexpr (sizeof...(Params) > 0)
    {
      unittest<Params...>::operator->*(::std::forward<Fun>(fun));
    }
    return *this;
  }

private:
  Param param;
};

} // namespace cuda::experimental::stf

// Try to detect when __VA_OPT__ is available
#  if defined(__cplusplus) && __cplusplus >= 202002L
#    define STF_HAS_UNITTEST_WITH_ARGS 1
#  endif

#  ifdef UNITTESTED_FILE

int main()
{
  using namespace cuda::experimental::stf;
  if (unittest<>::total() != unittest<>::passed())
  {
    fprintf(stderr, "FAIL: %zu of %zu\n", unittest<>::total() - unittest<>::passed(), unittest<>::total());
    return 1;
  }
  // fprintf(stdout, "PASS: %zu of %zu\n", passed(), total());
}

#    ifdef STF_HAS_UNITTEST_WITH_ARGS
#      define UNITTEST(name, ...)                                                    \
        [[maybe_unused]] static const auto CUDASTF_UNIQUE_NAME(unittest) =           \
          ::cuda::experimental::stf::unittest<>::make(                               \
            name, ::cuda::std::source_location::current() __VA_OPT__(, __VA_ARGS__)) \
            ->*[]([[maybe_unused]] const char* unittest_name __VA_OPT__(, [[maybe_unused]] auto&& unittest_param))
#    else
#      define UNITTEST(name)                                                                             \
        [[maybe_unused]] static const auto CUDASTF_UNIQUE_NAME(unittest) =                               \
          ::cuda::experimental::stf::unittest<>::make(name, ::cuda::std::source_location::current())->*[ \
          ]([[maybe_unused]] const char* unittest_name)
#    endif

#  else

#    ifdef STF_HAS_UNITTEST_WITH_ARGS
#      define UNITTEST(name, ...)                                                 \
        [[maybe_unused]] static const auto CUDASTF_UNIQUE_NAME(unused_unittest) = \
          []([[maybe_unused]] auto&& unittest_name __VA_OPT__(, [[maybe_unused]] auto&& unittest_param))
#    else
#      define UNITTEST(name)                                                      \
        [[maybe_unused]] static const auto CUDASTF_UNIQUE_NAME(unused_unittest) = \
          []([[maybe_unused]] auto&& unittest_name)
#    endif

#  endif

// Just making sure that core functionality works
#  ifdef STF_HAS_UNITTEST_WITH_ARGS
UNITTEST("Numeric NOP test.", int(), float(), double())
{
  using T = ::std::remove_reference_t<decltype(unittest_param)>;
  auto x  = EXPECT(T(1) + T(1) == T(2));
  static_assert(::std::is_same_v<decltype(x), T>);
};
#  endif // STF_HAS_UNITTEST_WITH_ARGS

UNITTEST("EXPECT")
{
  //! [EXPECT]
  EXPECT(1 == 1); // pass
  EXPECT(1 + 1 == 2, "All we know about mathematics has been upended!"); // pass
  auto p = EXPECT(malloc(100) != nullptr); // pass, put the pointer to allocated memory in p
  free(p);

  // This test is specifically about how we handle exceptions, so we disable it
  // when exceptions are not allowed
#  if _CCCL_HAS_EXCEPTIONS()
  try
  {
    EXPECT(41 == 102); // fail, will throw exception
  }
  catch (const ::std::exception& e)
  {
    // The exception message will contain the actual numbers 41 and 102.
    EXPECT(::std::string_view(e.what()).find("41 == 102 is false") != ::std::string_view::npos);
  }
#  endif // !_CCCL_HAS_EXCEPTIONS()
  //! [EXPECT]
};

/* Warning! All occurrences of `babe699bbb083d2b0aac2460e75bca96` below will be replaced with `UNITTEST` in
 * postprocessing. This is because we instructed doxygen to NOT generate documentation for the symbol UNITTEST, but we
 * still want to have the definition of the macro show up in the documentation. So we define
 * `babe699bbb083d2b0aac2460e75bca96`, document it, then replace it with `UNITTEST` in all doxygen-generated files.
 */

/**
 * @brief Macro to introduce unit tests. Use `UNITTEST("description") { ... code ... };` at namespace level or inside a
 * function
 *
 * @param name description of the unittest, e.g. `"Widgets are copyable and moveable"`
 * @param ... optional parameters in case multiple invocations of the unittest are necessary. Each parameters must be a
 * C++ value of any type and is passed to the unittest code with the name `unittest_param`.
 *
 * The `UNITTEST` invocation must be either at namespace level or in a function body as a declaration. The invocation is
 * followed by a braced compound statement ended with a semicolon. Example:
 * @snippet this UNITTEST simple
 *
 * `UNITTEST` accepts optional arguments, which are available inside the unittest body as `unittest_param`. The body of
 * the unittest will be executed once per argument. Arguments may have different types, effectively allowing a given
 * unittest to run with any number of types. Example:
 * @snippet this UNITTEST parameterized
 *
 * Unittest code will be executed if and only of the macro `UNITTESTED_FILE` is defined as a string that is equal to the
 * name of the current file. A project can be unittested one source file at a time by using the
 * @c -DUNITTESTED_FILE='"my_file.cpp"' flag in the project's build tool.
 *
 * If the unittest code throws an exception, it counts as a failure. Exception information will be output to `stderr`
 * and testing will continue with the next unittest.
 */
#  define babe699bbb083d2b0aac2460e75bca96(name, ...)

#  undef babe699bbb083d2b0aac2460e75bca96

//! [UNITTEST simple]
UNITTEST("Simple unittest")
{
  EXPECT(1 != 2);
};
//! [UNITTEST simple]

#  ifdef STF_HAS_UNITTEST_WITH_ARGS
//! [UNITTEST parameterized]
UNITTEST("Parameterized unittest", 42, 42L, 42.0f, 42.0)
{
  // In case the type of the parameter is needed, it can be accessed as follows:
  // using T = ::std::remove_reference_t<decltype(unittest_param)>;
  EXPECT(unittest_param == 42);
};
//! [UNITTEST parameterized]
#  endif // STF_HAS_UNITTEST_WITH_ARGS

// Documentation unittest for traits.h
UNITTEST("type_name")
{
  using namespace cuda::experimental::stf;
  //! [type_name]
  EXPECT(type_name<int> == "int");
  EXPECT(type_name<double> == "double");
  //! [type_name]
};

// Another documentation unittest for traits.h
UNITTEST("tuple2tuple")
{
  using namespace cuda::experimental::stf;
  //! [tuple2tuple]
  auto t  = ::std::make_tuple(1, 2, 3);
  auto t1 = tuple2tuple(t, [](auto x) {
    return x + 1.0;
  });
  static_assert(::std::is_same_v<decltype(t1), ::std::tuple<double, double, double>>);
  EXPECT(t1 == ::std::make_tuple(2.0, 3.0, 4.0));
  //! [tuple2tuple]
};

UNITTEST("only_convertible")
{
  using namespace cuda::experimental::stf;
  auto a = only_convertible<int>(1, "hello");
  EXPECT(a == 1);
};

UNITTEST("all_convertible")
{
  using namespace cuda::experimental::stf;
  auto a = all_convertible<int>(1, 2, "hello", 3);
  EXPECT((a == ::std::array<int, 3>{1, 2, 3}));
};

UNITTEST("shuffled arguments")
{
  using namespace cuda::experimental::stf;
  auto x0 = reserved::only_convertible_or(42);
  EXPECT(x0 == 42);
  auto x1 = reserved::only_convertible_or(42, 23);
  EXPECT(x1 == 23);
  auto x2 = reserved::only_convertible_or(42, "23", ::std::tuple{1});
  EXPECT(x2 == 42);
  // Note that a needs to be iniliazed because shuffled_args_check takes a const ref
  int a = 0;
  shuffled_args_check<int>(a);
  shuffled_args_check<double>(a);
  // shuffled_args_check<int, double>(a);  // "Duplicate argument: ..."
  // shuffled_args_check<double>(a, a);  // "Ambiguous argument: ..."
  // shuffled_args_check<char*>(a);  // "Incompatible argument: ..."
  auto t2       = shuffled_tuple<int, ::std::string>("hello", 1);
  auto [x, str] = shuffled_tuple<int, ::std::string>("hello", 1);
  EXPECT(str == "hello");
  EXPECT(x == 1);

  struct A
  {
    int value;
  };
  struct B
  {
    ::std::string text;
  };
  struct C
  {
    float number;
  };

  A aa{10};
  B bb{"example"};
  C cc{3.14f};
  // This will create an `::std::tuple<A, B, C>` from the provided arguments,
  // automatically assigning them based on their types.
  shuffled_tuple<A, B, C>(bb, cc, aa);
  shuffled_tuple<A, B, C>(cc, aa);
  shuffled_tuple<A, B, C>();
  // This will not compile because there are two arguments that can fill the same tuple slot.
  // auto my_tuple = shuffled_tuple<A, B, C, double>(bb, cc, aa, aa);
  // This will not compile because the argument '5' can convert to both 'int' and 'float',
  // causing an ambiguity.
  // auto my_tuple = shuffled_tuple<A, B, C>(b, c, 5);
};

UNITTEST("shuffled_array_tuple")
{
  using namespace cuda::experimental::stf;
  auto a = shuffled_array_tuple<int, ::std::string>(1, 2, "hello", 3, "world");
  EXPECT((a == ::std::tuple<::std::array<int, 3>, ::std::array<::std::string, 2>>{{1, 2, 3}, {"hello", "world"}}));
};

UNITTEST("cuda::std::source_location")
{
  auto test_func = [](const ::cuda::std::source_location loc = ::cuda::std::source_location::current()) {
    // Check the source location metadata
    EXPECT(loc.line() > 0); // The line number should be positive
    EXPECT(loc.file_name() != nullptr); // File name should not be null
    EXPECT(loc.function_name() != nullptr); // Function name should not be null
  };

  // Call the test function and validate the source location information
  test_func();
};

#else // _CCCL_DOXYGEN_INVOKED  Do not document
// Ensure these are ignored by Doxygen
#  define UNITTEST(name, ...)
#endif // _CCCL_DOXYGEN_INVOKED  Do not document
