//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/std/__string/char_traits.h>
#include <cuda/std/__type_traits/copy_cvref.h>

#include <cuda/experimental/execution.cuh>

#include <iostream>

#include "testing.cuh" // IWYU pragma: keep

// Workaround for https://github.com/llvm/llvm-project/issues/113087
#if defined(__clang__) && defined(__cpp_lib_tuple_like)
#  define C2H_CHECK_TUPLE(...) CHECK((__VA_ARGS__))
#else
#  define C2H_CHECK_TUPLE(...) CHECK(__VA_ARGS__)
#endif

//! A move-only type
struct movable
{
  _CCCL_HOST_DEVICE movable(int value)
      : value_(value)
  {}

  movable(movable&&) = default;

  _CCCL_HOST_DEVICE friend bool operator==(const movable& a, const movable& b) noexcept
  {
    return a.value_ == b.value_;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(const movable& a, const movable& b) noexcept
  {
    return a.value_ != b.value_;
  }

  friend std::ostream& operator<<(std::ostream& os, const movable& self)
  {
    os << "movable{" << self.value_ << "}";
    return os;
  }

  _CCCL_HOST_DEVICE int value() const
  {
    return value_;
  } // silence warning of unused private field

private:
  int value_;
};

//! A type with potentially throwing move/copy constructors
struct potentially_throwing
{
  potentially_throwing() = default;

  _CCCL_HOST_DEVICE potentially_throwing(potentially_throwing&&) noexcept(false) {}

  _CCCL_HOST_DEVICE potentially_throwing(const potentially_throwing&) noexcept(false) {}

  _CCCL_HOST_DEVICE potentially_throwing& operator=(potentially_throwing&&) noexcept(false)
  {
    return *this;
  }

  _CCCL_HOST_DEVICE potentially_throwing& operator=(const potentially_throwing&) noexcept(false)
  {
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os, const potentially_throwing&)
  {
    os << "potentially_throwing{}";
    return os;
  }
};

struct non_default_constructible
{
  _CCCL_HOST_DEVICE constexpr explicit non_default_constructible(int value) noexcept
      : value_(value)
  {}

  _CCCL_HOST_DEVICE friend constexpr bool
  operator==(const non_default_constructible& a, const non_default_constructible& b) noexcept
  {
    return a.value_ == b.value_;
  }

  _CCCL_HOST_DEVICE friend constexpr bool
  operator!=(const non_default_constructible& a, const non_default_constructible& b) noexcept
  {
    return a.value_ != b.value_;
  }

  friend std::ostream& operator<<(std::ostream& os, const non_default_constructible& self)
  {
    os << "non_default_constructible{" << self.value_ << "}";
    return os;
  }

  int value_;
};

struct string
{
  string() = default;

  _CCCL_HOST_DEVICE /*implicit*/ string(char const* c)
      : len(cuda::std::char_traits<char>::length(c))
      , str(cuda::std::char_traits<char>::copy(new char[len + 1], c, len + 1))
  {}

  _CCCL_HOST_DEVICE string(string&& other) noexcept
      : len(cuda::std::exchange(other.len, 0))
      , str(cuda::std::exchange(other.str, nullptr))
  {}

  _CCCL_HOST_DEVICE string(const string& other)
      : string(string(other.c_str()))
  {}

  _CCCL_HOST_DEVICE ~string()
  {
    delete[] str;
  }

  _CCCL_HOST_DEVICE void swap(string& other) noexcept
  {
    cuda::std::swap(len, other.len);
    cuda::std::swap(str, other.str);
  }

  _CCCL_HOST_DEVICE string& operator=(string other) noexcept
  {
    swap(other);
    return *this;
  }

  _CCCL_HOST_DEVICE string operator+(string const& other) const
  {
    string result;
    result.len = len + other.len;
    result.str = new char[result.len + 1];
    cuda::std::char_traits<char>::copy(result.str, str, len);
    cuda::std::char_traits<char>::copy(result.str + len, other.str, other.len + 1);
    return result;
  }

  _CCCL_HOST_DEVICE friend bool operator==(const string& left, const string& right) noexcept
  {
    return left.size() == right.size() && cuda::std::char_traits<char>::compare(left.str, right.str, left.size()) == 0;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(const string& left, const string& right) noexcept
  {
    return !(left == right);
  }

  friend std::ostream& operator<<(std::ostream& os, const string& self)
  {
    os << "string{" << self.str << "}";
    return os;
  }

  _CCCL_HOST_DEVICE cuda::std::size_t size() const
  {
    return len;
  }

  _CCCL_HOST_DEVICE const char* c_str() const
  {
    return str;
  }

private:
  cuda::std::size_t len{};
  char* str{};
};

struct error_code
{
  _CCCL_HOST_DEVICE friend bool operator==(const error_code& left, const error_code& right) noexcept
  {
    return left.ec == right.ec;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(const error_code& left, const error_code& right) noexcept
  {
    return !(left == right);
  }

  friend std::ostream& operator<<(std::ostream& os, const error_code& self)
  {
    os << "error_code{" << static_cast<int>(self.ec) << "}";
    return os;
  }

  std::errc ec;
};

template <class Sndr, class... Values>
void check_values(Sndr&& sndr, const Values&... values) noexcept
{
  try
  {
    auto opt = cudax_async::sync_wait(static_cast<Sndr&&>(sndr));
    if (!opt)
    {
      CUDAX_FAIL("Expected value completion; got stopped instead.");
    }
    else
    {
      auto&& vals = *opt;
      CUDAX_CHECK(vals == ::cuda::std::tie(values...));
    }
  }
  catch (...)
  {
    CUDAX_FAIL("Expected value completion; got error instead.");
  }
}

template <class... Ts>
using types = ::cuda::std::__type_list<Ts...>;

template <class... Values, class Sndr>
_CCCL_HOST_DEVICE void check_value_types(Sndr&&) noexcept
{
  using actual_t = cudax_async::value_types_of_t<Sndr, cudax_async::env<>, types, ::cuda::std::__make_type_set>;
  if constexpr (!::cuda::std::__type_set_eq_v<actual_t, Values...>)
  {
    ::cuda::std::__type_list<Values...> hard_error = actual_t{}; // Force the compiler to tell us the types involved.
    static_assert(::cuda::std::__type_set_eq_v<actual_t, Values...>, "value_types_of_t does not match expected types");
  }
}

template <class... Errors, class Sndr>
_CCCL_HOST_DEVICE void check_error_types(Sndr&&) noexcept
{
  using actual_t = cudax_async::error_types_of_t<Sndr, cudax_async::env<>, ::cuda::std::__make_type_set>;
  if constexpr (!::cuda::std::__type_set_eq_v<actual_t, Errors...>)
  {
    ::cuda::std::__type_list<Errors...> hard_error = actual_t{}; // Force the compiler to tell us the types involved.
    static_assert(::cuda::std::__type_set_eq_v<actual_t, Errors...>, "error_types_of_t does not match expected types");
  }
}

template <bool SendsStopped, class Sndr>
_CCCL_HOST_DEVICE void check_sends_stopped(Sndr&&) noexcept
{
  static_assert(cudax_async::sends_stopped<Sndr> == SendsStopped, "sends_stopped does not match expected value");
}

template <class Sndr, class... Ts>
inline void wait_for_value(Sndr&& snd, Ts&&... val)
{
  ::cuda::std::optional<::cuda::std::tuple<Ts...>> res = cudax_async::sync_wait(static_cast<Sndr&&>(snd));
  CHECK(res.has_value());
  ::cuda::std::tuple<Ts...> expected(static_cast<Ts&&>(val)...);
  if constexpr (::cuda::std::tuple_size_v<::cuda::std::tuple<Ts...>> == 1)
  {
    C2H_CHECK_TUPLE(::cuda::std::get<0>(res.value()) == ::cuda::std::get<0>(expected));
  }
  else
  {
    C2H_CHECK_TUPLE(res.value() == expected);
  }
}
