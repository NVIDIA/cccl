// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace static_map_detail
{
template <unsigned int K, unsigned int V>
struct key_value
{
  static const unsigned int key   = K;
  static const unsigned int value = V;
};

template <typename Head, typename Tail = void>
struct cons
{
  template <unsigned int Key, unsigned int DefaultValue>
  struct static_get
  {
    static const unsigned int value =
      (Key == Head::key) ? (Head::value) : Tail::template static_get<Key, DefaultValue>::value;
  };

  template <unsigned int DefaultValue>
  _CCCL_HOST_DEVICE static unsigned int get(unsigned int key)
  {
    return (key == Head::key) ? (Head::value) : Tail::template get<DefaultValue>(key);
  }
};

template <typename Head>
struct cons<Head, void>
{
  template <unsigned int Key, unsigned int DefaultValue>
  struct static_get
  {
    static const unsigned int value = (Key == Head::key) ? (Head::value) : DefaultValue;
  };

  template <unsigned int DefaultValue>
  _CCCL_HOST_DEVICE static unsigned int get(unsigned int key)
  {
    return (key == Head::key) ? (Head::value) : DefaultValue;
  }
};

template <unsigned int DefaultValue,
          unsigned int Key0   = 0,
          unsigned int Value0 = DefaultValue,
          unsigned int Key1   = 0,
          unsigned int Value1 = DefaultValue,
          unsigned int Key2   = 0,
          unsigned int Value2 = DefaultValue,
          unsigned int Key3   = 0,
          unsigned int Value3 = DefaultValue,
          unsigned int Key4   = 0,
          unsigned int Value4 = DefaultValue,
          unsigned int Key5   = 0,
          unsigned int Value5 = DefaultValue,
          unsigned int Key6   = 0,
          unsigned int Value6 = DefaultValue,
          unsigned int Key7   = 0,
          unsigned int Value7 = DefaultValue>
struct static_map
{
  using impl = cons<
    key_value<Key0, Value0>,
    cons<key_value<Key1, Value1>,
         cons<key_value<Key2, Value2>,
              cons<key_value<Key3, Value3>,
                   cons<key_value<Key4, Value4>,
                        cons<key_value<Key5, Value5>, cons<key_value<Key6, Value6>, cons<key_value<Key7, Value7>>>>>>>>>;

  template <unsigned int Key>
  struct static_get
  {
    static const unsigned int value = impl::template static_get<Key, DefaultValue>::value;
  };

  _CCCL_HOST_DEVICE static unsigned int get(unsigned int key)
  {
    return impl::template get<DefaultValue>(key);
  }
};
} // end namespace static_map_detail

template <unsigned int DefaultValue,
          unsigned int Key0   = 0,
          unsigned int Value0 = DefaultValue,
          unsigned int Key1   = 0,
          unsigned int Value1 = DefaultValue,
          unsigned int Key2   = 0,
          unsigned int Value2 = DefaultValue,
          unsigned int Key3   = 0,
          unsigned int Value3 = DefaultValue,
          unsigned int Key4   = 0,
          unsigned int Value4 = DefaultValue,
          unsigned int Key5   = 0,
          unsigned int Value5 = DefaultValue,
          unsigned int Key6   = 0,
          unsigned int Value6 = DefaultValue,
          unsigned int Key7   = 0,
          unsigned int Value7 = DefaultValue>
struct static_map
    : static_map_detail::static_map<
        DefaultValue,
        Key0,
        Value0,
        Key1,
        Value1,
        Key2,
        Value2,
        Key3,
        Value3,
        Key4,
        Value4,
        Key5,
        Value5,
        Key6,
        Value6,
        Key7,
        Value7>
{};

template <unsigned int Key, typename StaticMap>
struct static_lookup
{
  static const unsigned int value = StaticMap::template static_get<Key>::value;
};

template <typename StaticMap>
_CCCL_HOST_DEVICE unsigned int lookup(unsigned int key)
{
  return StaticMap::get(key);
}
} // end namespace detail
THRUST_NAMESPACE_END
