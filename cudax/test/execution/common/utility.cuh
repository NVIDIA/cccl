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

struct string
{
  string() = default;

  _CCCL_HOST_DEVICE explicit string(char const* c)
  {
    std::size_t len = 0;
    while (c[len++])
      ;
    char* tmp = str = new char[len];
    while ((*tmp++ = *c++))
      ;
  }

  _CCCL_HOST_DEVICE string(string&& other) noexcept
      : str(other.str)
  {
    other.str = nullptr;
  }

  _CCCL_HOST_DEVICE string(const string& other)
      : string(string(other.str))
  {}

  _CCCL_HOST_DEVICE ~string()
  {
    delete[] str;
  }

  _CCCL_HOST_DEVICE friend bool operator==(const string& left, const string& right) noexcept
  {
    char const* l = left.str;
    char const* r = right.str;
    while (*l && *r)
    {
      if (*l++ != *r++)
      {
        return false;
      }
    }
    return *l == *r;
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

private:
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

// run_loop isn't supported on-device yet, so neither can sync_wait be.
#if !defined(__CUDA_ARCH__)

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

#else // !defined(__CUDA_ARCH__)

template <class Sndr, class... Values>
void check_values(Sndr&& sndr, const Values&... values) noexcept
{}

#endif // !defined(__CUDA_ARCH__)

template <class... Ts>
using types = _CUDA_VSTD::__type_list<Ts...>;

template <class... Values, class Sndr>
_CCCL_HOST_DEVICE void check_value_types(Sndr&&) noexcept
{
  using actual_t = cudax_async::value_types_of_t<Sndr, cudax_async::env<>, types, _CUDA_VSTD::__make_type_set>;
  static_assert(_CUDA_VSTD::__type_set_eq_v<actual_t, Values...>, "value_types_of_t does not match expected types");
}

template <class... Errors, class Sndr>
_CCCL_HOST_DEVICE void check_error_types(Sndr&&) noexcept
{
  using actual_t = cudax_async::error_types_of_t<Sndr, cudax_async::env<>, _CUDA_VSTD::__make_type_set>;
  static_assert(_CUDA_VSTD::__type_set_eq_v<actual_t, Errors...>, "error_types_of_t does not match expected types");
}

template <bool SendsStopped, class Sndr>
_CCCL_HOST_DEVICE void check_sends_stopped(Sndr&&) noexcept
{
  static_assert(cudax_async::sends_stopped<Sndr> == SendsStopped, "sends_stopped does not match expected value");
}

template <class Sndr, class... Ts>
inline void wait_for_value(Sndr&& snd, Ts&&... val)
{
  _CUDA_VSTD::optional<_CUDA_VSTD::tuple<Ts...>> res = cudax_async::sync_wait(static_cast<Sndr&&>(snd));
  CHECK(res.has_value());
  _CUDA_VSTD::tuple<Ts...> expected(static_cast<Ts&&>(val)...);
  if constexpr (_CUDA_VSTD::tuple_size_v<_CUDA_VSTD::tuple<Ts...>> == 1)
  {
    C2H_CHECK_TUPLE(_CUDA_VSTD::get<0>(res.value()) == _CUDA_VSTD::get<0>(expected));
  }
  else
  {
    C2H_CHECK_TUPLE(res.value() == expected);
  }
}

// A sender adapter that adds attributes to the child sender's attributes.
struct write_attrs_t
{
  template <class Sndr, class Attrs>
  struct _sndr_t;

  template <class Attrs>
  struct __closure
  {
    Attrs _attrs_;

    template <class Sndr>
    [[nodiscard]] _CCCL_API friend auto operator|(Sndr _sndr, __closure _clsr)
    {
      return _sndr_t<Sndr, Attrs>{{}, static_cast<Attrs&&>(_clsr._attrs_), static_cast<Sndr&&>(_sndr)};
    }
  };

  template <class Sndr, class Attrs>
  [[nodiscard]] _CCCL_API auto operator()(Sndr _sndr, Attrs _attrs) const -> _sndr_t<Sndr, Attrs>
  {
    return _sndr_t<Sndr, Attrs>{{}, static_cast<Attrs&&>(_attrs), static_cast<Sndr&&>(_sndr)};
  }

  template <class Attrs>
  [[nodiscard]] _CCCL_API auto operator()(Attrs _attrs) const
  {
    return __closure<Attrs>{static_cast<Attrs&&>(_attrs)};
  }
};

template <class Sndr, class Attrs>
struct write_attrs_t::_sndr_t
{
  using sender_concept = cuda::experimental::execution::sender_t;
  using _attrs_t       = _CUDA_STD_EXEC::env<const Attrs&, _CUDA_STD_EXEC::env_of_t<Sndr>>;

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> _attrs_t
  {
    return {_attrs_, _CUDA_STD_EXEC::get_env(_sndr_)};
  }

  template <class Self, class... Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return cuda::experimental::execution::get_child_completion_signatures<Self, Sndr, Env...>();
  }

  template <class Rcvr>
  [[nodiscard]] _CCCL_API auto connect(Rcvr _rcvr) && -> cuda::experimental::execution::connect_result_t<Sndr, Rcvr>
  {
    return cuda::experimental::execution::connect(_CUDA_VSTD::move(_sndr_), _CUDA_VSTD::move(_rcvr));
  }

  template <class Rcvr>
  [[nodiscard]] _CCCL_API auto connect(Rcvr _rcvr) const& -> cuda::experimental::execution::connect_result_t<Sndr, Rcvr>
  {
    return cuda::experimental::execution::connect(_sndr_, _CUDA_VSTD::move(_rcvr));
  }

  _CCCL_NO_UNIQUE_ADDRESS write_attrs_t __tag_;
  Attrs _attrs_;
  Sndr _sndr_;
};

inline constexpr write_attrs_t write_attrs{};
