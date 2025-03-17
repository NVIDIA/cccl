//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// metafunction header that provides some vocabulary for manipulating type packs
// exports:
//   rotl<T...>
//   for_n<Count, T, Fn>
//   bind_last<Fn<...>, B>
//   append<List, Tx...>
//   append_n<Count, List, T>

#ifndef HETEROGENEOUS_META_H
#define HETEROGENEOUS_META_H

template <typename... Tx>
struct type_list
{};

// Rotates the typelist by removing the head and moving it to the tail
template <typename... Tx>
struct rotl_impl
{
  // Empty or 1 element case
  using type = type_list<Tx...>;
};
template <typename T, typename... Tx>
struct rotl_impl<T, Tx...>
{
  using type = type_list<Tx..., T>;
};
template <typename... Tx>
struct rotl_impl<type_list<Tx...>>
{
  using type = typename rotl_impl<Tx...>::type;
};

template <typename... Tx>
using rotl = typename rotl_impl<Tx...>::type;

// static_assert(std::is_same<rotl<int, char>, type_list<char,int>>(), "");
// static_assert(std::is_same<rotl<int, char, short>, type_list<char,short,int>>(), "");

template <size_t Idx, typename T, template <typename> class Fn>
struct for_n_impl
{
  using type = typename for_n_impl<Idx - 1, Fn<T>, Fn>::type;
};
template <typename T, template <typename> class Fn>
struct for_n_impl<1, T, Fn>
{
  using type = Fn<T>;
};
template <typename T, template <typename> class Fn>
struct for_n_impl<0, T, Fn>
{
  using type = T;
};

template <size_t Idx, typename T, template <typename> class Fn>
using for_n = typename for_n_impl<Idx, T, Fn>::type;

// static_assert(std::is_same<for_n<2, type_list<int, char, short>, rotl>, type_list<short, int, char>>(), "");
// static_assert(std::is_same<for_n<3, type_list<int, char, short>, rotl>, type_list<int, char, short>>(), "");

template <template <typename...> class Fn, typename B>
struct bind_last_impl
{
  template <typename T>
  struct bound
  {
    using type = Fn<T, B>;
  };

  template <typename T>
  using type = typename bound<T>::type;
};

template <template <typename...> class Fn, typename B>
using bind_last = bind_last_impl<Fn, B>;

template <typename TypeList, typename... Tx>
struct append_impl;
template <typename... Old, typename... New>
struct append_impl<type_list<Old...>, New...>
{
  using type = type_list<Old..., New...>;
};
template <typename... Old, typename... New>
struct append_impl<type_list<Old...>, type_list<New...>>
{
  using type = type_list<Old..., New...>;
};

template <typename TypeList, typename... Tx>
using append = typename append_impl<TypeList, Tx...>::type;

// static_assert(std::is_same<append<type_list<>, int>, type_list<int>>(), "");
// static_assert(std::is_same<append<type_list<int>, int>, type_list<int,int>>(), "");

template <size_t Idx, typename TypeList, typename T>
using append_n = for_n<Idx, TypeList, bind_last<append, T>::template type>;

// static_assert(std::is_same<append_n<3, type_list<char>, int>, type_list<char, int, int, int>>(), "");
// static_assert(std::is_same<append_n<5, type_list<char>, int>, type_list<char, int, int, int, int, int>>(), "");

#endif // HETEROGENEOUS_META_H
