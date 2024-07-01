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

#include <cuda/std/__exception/terminate.h>

#include "config.cuh"
#include "cpos.cuh"

// Must be the last include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
struct start_detached_t
{
#ifndef __CUDACC__

private:
#endif
  struct _opstate_base_t : _immovable
  {};

  struct _rcvr_t
  {
    using receiver_concept = receiver_t;

    _opstate_base_t* _opstate;
    void (*_destroy)(_opstate_base_t*) noexcept;

    template <class... As>
    void set_value(As&&...) && noexcept
    {
      _destroy(_opstate);
    }

    template <class Error>
    void set_error(Error&&) && noexcept
    {
      ::cuda::std::terminate();
    }

    void set_stopped() && noexcept
    {
      _destroy(_opstate);
    }
  };

  template <class Sndr>
  struct _opstate_t : _opstate_base_t
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = __async::completion_signatures_of_t<Sndr, _rcvr_t>;
    connect_result_t<Sndr, _rcvr_t> _op;

    static void _destroy(_opstate_base_t* ptr) noexcept
    {
      delete static_cast<_opstate_t*>(ptr);
    }

    _CCCL_HOST_DEVICE explicit _opstate_t(Sndr&& sndr)
        : _op(__async::connect(static_cast<Sndr&&>(sndr), _rcvr_t{this, &_destroy}))
    {}

    _CCCL_HOST_DEVICE void start() & noexcept
    {
      __async::start(_op);
    }
  };

public:
  /// @brief Eagerly connects and starts a sender and lets it
  /// run detached.
  template <class Sndr>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void operator()(Sndr sndr) const
  {
    __async::start(*new _opstate_t<Sndr>{static_cast<Sndr&&>(sndr)});
  }
};

_CCCL_GLOBAL_CONSTANT start_detached_t start_detached{};
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
