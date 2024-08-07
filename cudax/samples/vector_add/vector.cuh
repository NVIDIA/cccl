//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CONTAINER_VECTOR
#define _CUDAX__CONTAINER_VECTOR

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/utility.cuh>

#if 1 //_CCCL_STD_VER >= 2017
namespace cuda::experimental
{
using ::cuda::std::span;
using ::thrust::device_vector;
using ::thrust::host_vector;

namespace detail
{
template <typename _Ty>
struct __in_box
{
  const _Ty& __val;
};

template <typename _Ty>
struct __out_box
{
  _Ty& __val;
};
} // namespace detail

template <typename _Ty>
class vector
{
public:
  vector() = default;
  explicit vector(size_t n)
      : __h_(n)
  {}

  _Ty& operator[](size_t i) noexcept
  {
    __dirty_ = true;
    return __h_[i];
  }

  const _Ty& operator[](size_t i) const noexcept
  {
    return __h_[i];
  }

private:
  enum class __param : unsigned
  {
    _in    = 1,
    _out   = 2,
    _inout = 3
  };

  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr __param operator&(__param __a, __param __b) noexcept
  {
    return __param(unsigned(__a) & unsigned(__b));
  }

  void sync_host_to_device() const
  {
    if (__dirty_)
    {
      printf("sync_host_to_device\n");
      __d_     = __h_;
      __dirty_ = false;
    }
  }

  void sync_device_to_host()
  {
    printf("sync_device_to_host\n");
    __h_ = __d_;
  }

  template <__param _Param>
  struct __action : detail::__immovable
  {
    static constexpr bool __mut = ((_Param & __param::_out) == __param::_out);
    using __cv_vector           = ::cuda::std::__maybe_const<!__mut, vector>;

    explicit __action(stream_ref __str, __cv_vector& __v) noexcept
        : __str_(__str)
        , __v_(__v)
    {
      printf("action()\n");
      if constexpr ((_Param & __param::_in) == __param::_in)
      {
        __v_.sync_host_to_device();
      }
    }

    ~__action()
    {
      printf("~action()\n");
      if constexpr ((_Param & __param::_out) == __param::_out)
      {
        printf("about to synchronize the stream\n");
        fflush(stdout);
        __str_.wait(); // wait for the kernel to finish
        printf("done synchronizing the stream\n");
        fflush(stdout);
        __v_.sync_device_to_host();
      }
    }

    using __as_kernel_arg = ::cuda::std::span<_Ty>;

    operator ::cuda::std::span<_Ty>()
    {
      printf("to span\n");
      return {__v_.__d_.data().get(), __v_.__d_.size()};
    }

  public:
    stream_ref __str_;
    __cv_vector& __v_;
  };

  _CCCL_NODISCARD_FRIEND __action<__param::_inout> __cudax_launch_transform(stream_ref __str, const vector& __v) noexcept
  {
    return __action<__param::_inout>{__str, __v};
  }

  _CCCL_NODISCARD_FRIEND __action<__param::_in>
  __cudax_launch_transform(stream_ref __str, detail::__in_box<vector> __b) noexcept
  {
    return __action<__param::_in>{__str, __b.__val};
  }

  _CCCL_NODISCARD_FRIEND __action<__param::_out>
  __cudax_launch_transform(stream_ref __str, detail::__out_box<vector> __b) noexcept
  {
    return __action<__param::_out>{__str, __b.__val};
  }

  host_vector<_Ty> __h_;
  mutable device_vector<_Ty> __d_{};
  mutable bool __dirty_ = true;
};

template <class _Ty>
detail::__in_box<_Ty> in(const _Ty& __v) noexcept
{
  return {__v};
}

template <class _Ty>
detail::__out_box<_Ty> out(_Ty& __v) noexcept
{
  return {__v};
}

} // namespace cuda::experimental

#endif
#endif
