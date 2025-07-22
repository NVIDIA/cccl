/*! \file meta.h
 *  \brief Defines template classes
 *         for metaprogramming in the
 *         unit tests.
 */

#pragma once

namespace unittest
{

// mark the absence of a type
struct null_type
{};

// this type encapsulates a list of
// types
template <typename... Ts>
struct type_list
{};

// this type provides a way of indexing
// into a type_list
template <typename List, unsigned int i>
struct get_type
{
  using type = null_type;
};

template <typename T, typename... Ts>
struct get_type<type_list<T, Ts...>, 0>
{
  using type = T;
};

template <typename T, typename... Ts, unsigned int i>
struct get_type<type_list<T, Ts...>, i>
{
  using type = typename get_type<type_list<Ts...>, i - 1>::type;
};

template <typename T, unsigned int i>
using get_type_t = typename get_type<T, i>::type;

// this type and its specialization provides a way to
// iterate over a type_list, and
// applying a unary function to each type
template <typename TypeList, template <typename> class Function>
struct for_each_type;

template <template <typename...> typename L, typename... Ts, template <typename> class Function>
struct for_each_type<L<Ts...>, Function>
{
  template <typename... Us>
  void operator()(Us... args)
  {
    (..., Function<Ts>{}(::cuda::std::forward<Us>(args)...));
  }
};

// this type and its specialization instantiates
// a template by applying T to Template.
// if T == null_type, then its result is also null_type
template <template <typename> class Template, typename T>
struct ApplyTemplate1
{
  using type = Template<T>;
};

template <template <typename> class Template>
struct ApplyTemplate1<Template, null_type>
{
  using type = null_type;
};

// this type and its specializations instantiates
// a template by applying T1 & T2 to Template.
// if either T1 or T2 == null_type, then its result
// is also null_type
template <template <typename, typename> class Template, typename T1, typename T2>
struct ApplyTemplate2
{
  using type = Template<T1, T2>;
};

template <template <typename, typename> class Template, typename T>
struct ApplyTemplate2<Template, T, null_type>
{
  using type = null_type;
};

template <template <typename, typename> class Template, typename T>
struct ApplyTemplate2<Template, null_type, T>
{
  using type = null_type;
};

template <template <typename, typename> class Template>
struct ApplyTemplate2<Template, null_type, null_type>
{
  using type = null_type;
};

// this type creates a new type_list by applying a Template to each of
// the Type_list's types
template <typename TypeList, template <typename> class Template>
struct transform1;

template <typename... Ts, template <typename> class Template>
struct transform1<type_list<Ts...>, Template>
{
  using type = type_list<typename ApplyTemplate1<Template, Ts>::type...>;
};

template <typename TypeList1, typename TypeList2, template <typename, typename> class Template>
struct transform2;

template <typename... T1s, typename... T2s, template <typename, typename> class Template>
struct transform2<type_list<T1s...>, type_list<T2s...>, Template>
{
  using type = type_list<typename ApplyTemplate2<Template, T1s, T2s>::type...>;
};

template <typename... Ls>
struct concat;

template <typename L>
struct concat<L>
{
  using type = L;
};

template <template <typename...> class L, typename... T1s, typename... T2s, typename... Ls>
struct concat<L<T1s...>, L<T2s...>, Ls...>
{
  using type = concat<L<T1s..., T2s...>, Ls...>;
};

} // namespace unittest
