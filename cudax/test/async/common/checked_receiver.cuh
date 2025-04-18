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

#include <cuda/experimental/__async/sender.cuh>

#include "testing.cuh"

namespace
{
template <class... Values>
struct checked_value_receiver
{
  using receiver_concept = cudax_async::receiver_t;

  checked_value_receiver(Values... values)
      : _values{values...}
  {}

  // This overload is needed to avoid an nvcc compiler bug where a variadic
  // pack is not visible within the scope of a lambda.
  _CCCL_HOST_DEVICE void set_value() && noexcept
  {
    if constexpr (!_CUDA_VSTD::is_same_v<_CUDA_VSTD::__type_list<Values...>, _CUDA_VSTD::__type_list<>>)
    {
      CUDAX_FAIL("expected a value completion; got a different value");
    }
  }

  template <class... As>
  _CCCL_HOST_DEVICE void set_value(As... as) && noexcept
  {
    if constexpr (_CUDA_VSTD::is_same_v<_CUDA_VSTD::__type_list<Values...>, _CUDA_VSTD::__type_list<As...>>)
    {
      _values.__apply(
        [&](auto const&... vs) {
          CUDAX_CHECK(((vs == as) && ...));
        },
        _values);
    }
    else
    {
      CUDAX_FAIL("expected a value completion; got a different value");
    }
  }

  template <class Error>
  _CCCL_HOST_DEVICE void set_error(Error) && noexcept
  {
    CUDAX_FAIL("expected a value completion; got an error");
  }

  _CCCL_HOST_DEVICE void set_stopped() && noexcept
  {
    CUDAX_FAIL("expected a value completion; got stopped");
  }

  cudax_async::__tuple<Values...> _values;
};

template <class... Values>
checked_value_receiver(Values...) -> checked_value_receiver<Values...>;

template <class Error>
struct checked_error_receiver
{
  using receiver_concept = cudax_async::receiver_t;

  template <class... As>
  _CCCL_HOST_DEVICE void set_value(As...) && noexcept
  {
    CUDAX_FAIL("expected an error completion; got a value");
  }

  template <class Ty>
  _CCCL_HOST_DEVICE void set_error(Ty ty) && noexcept
  {
    if constexpr (_CUDA_VSTD::is_same_v<Error, Ty>)
    {
      CUDAX_CHECK(ty == _error);
    }
    else
    {
      CUDAX_FAIL("expected an error completion; got a different error");
    }
  }

  _CCCL_HOST_DEVICE void set_stopped() && noexcept
  {
    CUDAX_FAIL("expected a value completion; got stopped");
  }

  Error _error;
};

template <class Error>
checked_error_receiver(Error) -> checked_error_receiver<Error>;

struct checked_stopped_receiver
{
  using receiver_concept = cudax_async::receiver_t;

  template <class... As>
  _CCCL_HOST_DEVICE void set_value(As...) && noexcept
  {
    CUDAX_FAIL("expected a stopped completion; got a value");
  }

  template <class Ty>
  _CCCL_HOST_DEVICE void set_error(Ty) && noexcept
  {
    CUDAX_FAIL("expected an stopped completion; got an error");
  }

  _CCCL_HOST_DEVICE void set_stopped() && noexcept {}
};

} // namespace
