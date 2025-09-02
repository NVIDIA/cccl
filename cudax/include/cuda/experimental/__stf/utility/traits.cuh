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
 * @brief Utilities related to trait classes
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

#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/mdspan>

#include <cuda/experimental/__stf/utility/core.cuh>

#include <array>
#include <cassert>
#include <string_view>
#include <tuple>

namespace cuda::experimental::stf
{

namespace reserved
{

// We use this function as a detector for what __PRETTY_FUNCTION__ looks like
template <typename T>
constexpr ::std::string_view type_name_IMPL()
{
#if _CCCL_COMPILER(MSVC)
  return __FUNCSIG__;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  return __PRETTY_FUNCTION__;
#endif // !_CCCL_COMPILER(MSVC)
}

// Length of prefix and suffix in __PRETTY_FUNCTION__ when used with `type_name`.
inline constexpr ::std::pair<size_t, size_t> type_name_affixes = [] {
  const auto p      = type_name_IMPL<double>();
  const auto target = ::std::string_view("double");
  const auto len    = target.size();
  // Simulate p.find() by hand because clang can't do it.
  size_t i = target.npos;
  for (std::size_t start = 0; start <= p.size() - len; ++start)
  {
    if (p.substr(start, len) == target)
    {
      i = start; // Found the substring, set i to the starting position
      break; // Exit loop after finding the first match
    }
  }
  auto j = p.size() - i - len;
  return ::std::pair{i, j};
}();

template <class T>
constexpr ::std::string_view type_name_impl()
{
#if _CCCL_COMPILER(MSVC)
  constexpr ::std::string_view p = __FUNCSIG__;
  // MSVC does not provide constexpr methods so we make this utility much simpler and return __FUNCSIG__ directly
  return p;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  ::std::string_view p = __PRETTY_FUNCTION__;
  return p.substr(type_name_affixes.first, p.size() - type_name_affixes.first - type_name_affixes.second);
#endif // !_CCCL_COMPILER(MSVC)
}

} // namespace reserved

/**
 * @brief Yields a string form of type name. Exact spelling not guaranteed (e.g. `type_name<int*>` may be `"int*"`,
 * `"int *"` etc).
 *
 * @tparam T The type to show.
 *
 * @paragraph example Example
 * @snippet unittest.h type_name
 */
template <class T>
inline constexpr ::std::string_view type_name = reserved::type_name_impl<T>();

/**
 * @brief Converts each element in `t` to a new value by calling `f`, then returns a tuple collecting the values thus
 * obtained.
 *
 * @tparam Tuple Type of the tuple to convert
 * @tparam Fun Type of mapping function to apply
 * @param t Object to convert, must support `std::apply`
 * @param f function to convert each element of the tuple, must take a single parameter
 * @return constexpr auto The tuple resulting from the mapping
 *
 * @paragraph example Example
 * @snippet unittest.h tuple2tuple
 */
template <typename Tuple, typename Fun>
constexpr auto tuple2tuple(const Tuple& t, Fun&& f)
{
  return ::std::apply(
    [&](auto&&... x) {
      return ::std::tuple(f(::std::forward<decltype(x)>(x))...);
    },
    t);
}

/*
 * @brief A function that will fail to compile, and result in an error message
 * with type T. Used internally for debugging. Since this uses a static_assert,
 * it will break compilation even if the function is called in a path that is
 * supposed to be unreachable !
 *
 * @tparam T A type which we want to display.
 */
template <typename T>
class print_type_name_and_fail
{
  static_assert(::std::integral_constant<T*, nullptr>::value, "Type name is: ");
};

namespace reserved
{

/**
 * @brief A singleton template class implementing the Meyers Singleton design pattern.
 *
 * @tparam T The type of the singleton object.
 *
 * It uses the "Construct On First Use Idiom" to prevent issues related to
 * the static initialization order fiasco.
 *
 * Usage rules:
 * - The default constructor of `T` should be protected.
 * - The destructor of `T` should be protected.
 * - The copy and move constructors of `T` should be disabled (implicit if you follow the rules above).
 *
 * Example usage:
 * ```cpp
 * class my_singleton : public meyers_singleton<my_singleton> {
 * protected:
 *   my_singleton() = default;
 *   ~my_singleton() = default;
 * };
 * ```
 */
template <class T>
class meyers_singleton
{
protected:
  template <class U>
  struct wrapper
  {
    using type = U;
  };
  friend typename wrapper<T>::type;

  meyers_singleton()                        = default;
  ~meyers_singleton()                       = default;
  meyers_singleton(const meyers_singleton&) = delete;
  meyers_singleton(meyers_singleton&&)      = delete;

public:
  /**
   * @brief Provides access to the single instance of the class.
   *
   * @return T& A reference to the singleton instance.
   *
   * If the instance hasn't been created yet, this function will create it.
   */
  static T& instance()
  {
    static_assert(!::std::is_default_constructible_v<T>,
                  "Make the default constructor of your Meyers singleton protected.");
    static_assert(!::std::is_destructible_v<T>, "Make the destructor of your Meyers singleton protected.");
    static_assert(!::std::is_copy_constructible_v<T>, "Disable the copy constructor of your Meyers singleton.");
    static_assert(!::std::is_move_constructible_v<T>, "Disable the move constructor of your Meyers singleton.");
    struct U : T
    {};
    static U instance;
    return instance;
  }
};

} // end namespace reserved

/**
 * @brief Converts an array-like object (such as an `std::array`) to an `std::tuple`.
 *
 * This function template takes an array-like object and returns a tuple containing the same elements.
 * If the input array has a size of zero, an empty tuple is returned.
 *
 * @tparam Array Type of the array-like object. Must have `std::tuple_size_v<Array>` specialization.
 * @param array The array-like object to be converted to a tuple.
 * @return A tuple containing the elements of the input array.
 *
 * Example usage:
 * @code
 * std::array<int, 3> arr = {1, 2, 3};
 * auto t = to_tuple(arr); // t is a std::tuple<int, int, int>
 * @endcode
 */
template <typename Array>
auto to_tuple(Array&& array)
{
  return tuple2tuple(::std::forward<Array>(array), [](auto&& e) {
    return ::std::forward<decltype(e)>(e);
  });
}

/**
 * @brief Array-like tuple with a single element type repeated `n` times.
 *
 * The `array_tuple` template generates a `std::tuple` with a single type `T` repeated `n` times.
 * This can be used to create a tuple with consistent types and a fixed size.
 *
 * @tparam T The type of the elements that the tuple will contain.
 * @tparam n The number of elements that the tuple will contain.
 *
 * ### Example
 *
 * ```cpp
 * using my_tuple = array_tuple<int, 5>; // Results in std::tuple<int, int, int, int, int>
 * ```
 *
 * @note The specialization `array_tuple<T, 0>` will result in an empty tuple (`std::tuple<>`).
 */
template <typename T, size_t n>
using array_tuple = decltype(to_tuple(::std::array<T, n>{}));

// Mini-unittest
static_assert(::std::is_same_v<array_tuple<size_t, 3>, ::std::tuple<size_t, size_t, size_t>>);

namespace reserved
{

/**
 * @brief Converts an `std::tuple` into a `cuda::std::array`.
 *
 * This function takes an `std::tuple` with elements of potentially differing types, provided the first type, `T0`, is
 * assignable from the rest. The resulting `std::array` will contain the elements from the tuple, where all the
 * elements are of the type `T0`.
 *
 * @tparam T0 The type of the first element in the tuple, and the type of the elements in the resulting array.
 * @tparam Ts... The types of the other elements in the tuple. They must all be convertible to `T0`.
 * @param obj The `std::tuple` object to convert.
 * @return An `std::array` containing the elements from the tuple, converted to type `T0`.
 */
template <typename T0, typename... Ts>
::cuda::std::array<T0, 1 + sizeof...(Ts)> to_cuda_array(const ::std::tuple<T0, Ts...>& obj)
{
  ::cuda::std::array<T0, 1 + sizeof...(Ts)> result;
  each_in_tuple(obj, [&](auto index, const auto& value) {
    result[index] = value;
  });
  return result;
}

/**
 * @brief Converts an `std::array` to a `cuda::std::array`
 */
template <typename T, size_t N>
::cuda::std::array<T, N> convert_to_cuda_array(const ::std::array<T, N>& std_array)
{
  ::cuda::std::array<T, N> result;
  for (size_t i = 0; i < N; i++)
  {
    result[i] = std_array[i];
  }
  return result;
}

} // end namespace reserved

/**
 * @brief Extracts and returns the first argument from a parameter pack that is convertible to type `T`.
 *
 * This function template recursively inspects each argument in a parameter pack until it finds the first
 * argument that can be converted to type `T`. It statically asserts that only one such convertible argument exists
 * in the parameter pack to ensure uniqueness.
 *
 * @tparam T The target type to which the argument should be convertible.
 * @tparam P0 The type of the first argument in the parameter pack.
 * @tparam P Variadic template representing the rest of the arguments in the parameter pack.
 * @param p0 The first argument in the parameter pack.
 * @param p The remaining arguments in the parameter pack.
 * @return `T` A copy or reference to the first argument that is convertible to type `T`.
 *
 * @note This function will cause a compile-time error if more than one argument in the parameter pack is convertible to
 * type `T`.
 *
 * \code{.cpp}
 *   int i = 42;
 *   double d = 3.14;
 *   std::string s = "hello";
 *   // The following call will return 's'.
 *   auto result = only_convertible<std::string>(i, d, s);
 * \endcode
 *
 */
template <typename T, typename P0, typename... P>
T only_convertible(P0&& p0, [[maybe_unused]] P&&... p)
{
  if constexpr (::std::is_convertible_v<P0, T>)
  {
    static_assert(!(::std::is_convertible_v<P, T> || ...), "Duplicate argument type found");
    return ::std::forward<P0>(p0);
  }
  else
  {
    // Ignore current head and recurse to tail
    return only_convertible<T>(::std::forward<P>(p)...);
  }
}

/**
 * @brief Creates an `std::array` of type `T` from all convertible elements in a parameter pack.
 *
 * This template function inspects each element in a parameter pack and adds it to an `std::array` if it is
 * convertible to the type `T`. It returns this array containing all convertible elements. The function ensures that
 * only convertible elements are added by using `std::is_convertible`. It also handles exceptions and properly
 * destructs already constructed elements in case of an exception during array construction.
 *
 * @tparam T The target type to which the elements of the parameter pack should be convertible.
 * @tparam P Variadic template representing the types in the parameter pack.
 * @param p The parameter pack containing elements to be checked for convertibility and potentially added to the array.
 * @return `std::array<T, N>` An array of type `T` containing all elements from the parameter pack that are
 * convertible to `T`.
 *
 * @note The size of the returned array, `N`, is determined at compile time based on the number of convertible elements
 * in the parameter pack.
 * @throws Any exception thrown during the construction of elements in the array will be propagated after destructing
 * already constructed elements.
 *
 * \code{.cpp}
 *   int i = 42;
 *   double d = 3.14;
 *   std::string s = "hello";
 *   // The following call will create an array of strings from all arguments that can be converted to a string.
 *   // In this case, only 's' is convertible, so the array will contain two copies of 's'.
 *   auto result = all_convertible<std::string>(i, s, d, s);
 * \endcode
 */
template <typename T, typename... P>
auto all_convertible(P&&... p)
{
  // We use a union here to prevent the compiler from calling the destructor of the array.
  // All construction/destruction will be done manually for efficiency purposes.
  static constexpr size_t size = (::std::is_convertible_v<P, T> + ...);
  unsigned char buffer[size * sizeof(T)];
  auto& result = *reinterpret_cast<::std::array<T, size>*>(&buffer[0]);
  size_t i     = 0; // marks the already-constructed portion of the array

  auto rollback = [&result, &i]() {
    for (size_t j = 0; j < i; ++j)
    {
      result[j].~T();
    }
  };

  auto __guard = ::cuda::std::__make_exception_guard(rollback);
  each_in_pack(
    [&](auto&& e) {
      if constexpr (::std::is_convertible_v<decltype(e), T>)
      {
        new (result.data() + i) T(::std::forward<decltype(e)>(e));
        ++i;
      }
    },
    ::std::forward<P>(p)...);
  __guard.__complete();
  return mv(result);
}

namespace reserved
{
/**
 * @brief Chooses a parameter from `P...` of a type convertible to `T`. If found, it is returned. If no such parameter
 * is found, returns `default_v`.
 *
 * For now only value semantics are supported.
 *
 * @tparam T Result type
 * @tparam P Variadic parameter types
 * @param default_v Default value
 * @param p Variadic parameter values
 * @return T Either the first convertible parameter, or `default_v` if no such parameter is found
 */
template <typename T, typename... P>
T only_convertible_or([[maybe_unused]] T default_v, [[maybe_unused]] P&&... p)
{
  if constexpr (!(::std::is_convertible_v<P, T> || ...))
  {
    return default_v;
  }
  else
  {
    return only_convertible<T>(::std::forward<P>(p)...);
  }
}

/* Checks whether a collection of `DataTypes` objects can be unambiguously initialized (in some order)
 from a collection of `ArgTypes` objects. Not all objects must be initialized,
 e.g. `check_initialization<int, int*>(1)` passes. */
template <typename... DataTypes>
struct check_initialization
{
  /* Yields the number of types in `Ts` to which `T` can be converted. */
  template <typename T>
  static constexpr int count_convertibilty = (::std::is_convertible_v<T, DataTypes> + ... + 0);

  template <typename... ArgTypes>
  static constexpr void from()
  {
    (
      [] {
        using T = ArgTypes;
        static_assert(count_convertibilty<T> > 0,
                      "Incompatible argument: argument type doesn't match any member type.");
        static_assert(count_convertibilty<T> == 1,
                      "Ambiguous argument: argument type converts to more than one member type.");
      }(),
      ...); // This expands ArgTypes
  }
};
} // namespace reserved

/**
 * @brief Checks the convertibility of argument types to a set of data types.
 *
 * This function checks if each type in `ArgTypes` is convertible to exactly one type in DataTypes.
 * If a type is not convertible to exactly one type, a static assertion will fail at compile time.
 *
 * @tparam ArgTypes The types to check the convertibility of.
 * @tparam DataTypes The types to check the convertibility to.
 * @param ... The data of the types to check the convertibility to.
 *
 * @note A static_assert error occurs if a type is not convertible to exactly one type.
 *
 * \code{.cpp}
 *     struct A {};
 *     struct B {};
 *     struct C {};
 *     struct D {};
 *
 *     A a;
 *     B b;
 *     C c;
 *     D d;
 *
 *     // This will compile successfully because each type A, B, C is convertible to itself and only itself.
 *     shuffled_args_check<A, B, C>(c, a, b);
 *
 *     // This will fail to compile because type D is not provided in the DataTypes.
 *     // shuffled_args_check<A, B, C>(a, b, c, d);
 *
 *     // This will fail to compile because int is convertible to both float and double, causing ambiguity.
 *     // shuffled_args_check<int, float, double>(5);
 * \endcode
 */
template <typename... ArgTypes, typename... DataTypes>
void shuffled_args_check(const DataTypes&...)
{
  reserved::check_initialization<DataTypes...>::template from<ArgTypes...>();
}

/**
 * @brief Creates a tuple with arguments in any order.
 *
 * This function creates a tuple where each element is a value from the variadic argument list `args` that is
 * convertible to the corresponding type in `DataTypes`. The function checks the convertibility of each argument type to
 * the corresponding type in `DataTypes` and throws a static assertion if any argument type is not convertible to
 * exactly one type in `DataTypes`.
 *
 * @tparam DataTypes The types to use for the tuple elements.
 * @tparam ArgTypes The types of the arguments to shuffle.
 * @param args The arguments to shuffle.
 *
 * @return std::tuple<DataTypes...> A tuple where each element is a value from `args` that is convertible to the
 * corresponding type in `DataTypes`.
 *
 * @note A static_assert error is issued if a type is not convertible to exactly one type.
 *
 * \code{.cpp}
 *     struct A { int value; };
 *     struct B { std::string text; };
 *     struct C { float number; };
 *
 *     A a{10};
 *     B b{"example"};
 *     C c{3.14f};
 *
 *     // This will create an `std::tuple<A, B, C>` from the provided arguments,
 *     // automatically assigning them based on their types.
 *     auto my_tuple = shuffled_tuple<A, B, C>(b, c, a);
 *     // Now my_tuple is of type std::tuple<A, B, C>, and contains a, b, c in that order.
 *
 *     // This will not compile because there are two arguments that can fill the same tuple slot.
 *     // auto my_tuple = shuffled_tuple<A, B, C, double>(b, c, a, a);
 *
 *     // This will not compile because the argument '5' can convert to both 'int' and 'float',
 *     // causing an ambiguity.
 *     // auto my_tuple = shuffled_tuple<A, B, C>(b, c, 5);
 *  \endcode
 */
template <typename... DataTypes, typename... ArgTypes>
::std::tuple<DataTypes...> shuffled_tuple(ArgTypes... args)
{
  reserved::check_initialization<DataTypes...>::template from<ArgTypes...>();
  return ::std::tuple<DataTypes...>{reserved::only_convertible_or(DataTypes(), mv(args)...)...};
}

/**
 * @brief Creates a tuple where each element is an `std::array`, constructed from the arguments provided.
 *
 * This function constructs a tuple where each element is an array of a specific type. Each array in the tuple
 * contains all the arguments of the corresponding type. The function checks at compile-time to ensure that each
 * argument type is convertible to exactly one of the tuple's element types, avoiding ambiguity or incompatible types.
 *
 * @tparam DataTypes The types of the arrays in the tuple. Each type corresponds to an array in the tuple.
 * @tparam ArgTypes Variadic template representing the types of the arguments.
 * @param args The arguments from which the arrays in the tuple are constructed. These can be in any order.
 * @return A tuple where each element is an array of one of the `DataTypes`, containing the respective convertible
 * arguments.
 *
 * @note The function uses `all_convertible` internally to construct arrays for each data type in `DataTypes`.
 *
 * Example usage:
 * @code
 * auto result = shuffled_array_tuple<int, double, char>('a', 5, 6.0, 'b', 7);
 * // result is a tuple containing:
 * // std::array<int, 2> {5, 7},
 * // std::array<double, 1> {6.0},
 * // std::array<char, 2> {'a', 'b'}
 * @endcode
 */
template <typename... DataTypes, typename... ArgTypes>
auto shuffled_array_tuple(ArgTypes... args)
{
  reserved::check_initialization<DataTypes...>::template from<ArgTypes...>();
  return ::std::tuple{all_convertible<DataTypes>(mv(args)...)...};
}

namespace reserved
{

/**
 * @brief Trait class to check if a function can be invoked with `std::apply` using a tuple type
 */
template <typename F, typename Tuple>
inline constexpr bool is_applicable_v = false;

template <typename F, typename... Args>
inline constexpr bool is_applicable_v<F, ::std::tuple<Args...>> = ::std::is_invocable_v<F, Args...>;

/**
 * @brief A compile-time boolean that checks if a type supports streaming with std::ostream <<.
 *
 * This trait is true if the type T can be streamed using std::ostream <<, and false otherwise.
 *
 * @tparam T The type to check for streaming support with std::ostream <<.
 */
template <typename T, typename = void>
struct has_ostream_operator : ::std::false_type
{};

template <typename T>
struct has_ostream_operator<T, decltype(void(::std::declval<::std::ostream&>() << ::std::declval<const T&>()), void())>
    : ::std::true_type
{};

} // end namespace reserved

} // namespace cuda::experimental::stf
