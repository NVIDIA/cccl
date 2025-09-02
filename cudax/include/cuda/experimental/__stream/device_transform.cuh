//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_DEVICE_TRANSFORM_CUH
#define _CUDAX__STREAM_DEVICE_TRANSFORM_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
namespace __detail
{
// This function turns rvalues into prvalues and leaves lvalues as is.
template <typename _Tp>
_CCCL_API constexpr auto __ixnay_xvalue(_Tp&& __value) noexcept(__nothrow_movable<_Tp>) -> _Tp
{
  return ::cuda::std::forward<_Tp>(__value);
}
} // namespace __detail

template <typename _Tp>
using __remove_rvalue_reference_t = decltype(__detail::__ixnay_xvalue(::cuda::std::declval<_Tp>()));

namespace __tfx
{
// Device transform:
//
// The device transform is a mechanism to transform arguments passed to the
// algorithms prior to actually enqueueing work on a stream. This is useful for
// example, to automatically convert contiguous ranges into spans. It is also
// useful for executing per-argument actions before and after the kernel launch.
// A host_vector might want a pre-launch action to copy data from host to device
// and a post-launch action to copy data back from device to host.
//
// The expression `device_transform(stream, arg)` is expression-equivalent to
// the first of the following expressions that is valid:
//
// 1. `transform_device_argument(stream, arg).transformed_argument()`
// 2. `transform_device_argument(stream, arg)`
// 3. `arg.transformed_argument()`
// 4. `arg`
_CCCL_HOST_API void transform_device_argument();

struct _CCCL_TYPE_VISIBILITY_DEFAULT __device_transform_t
{
  // Types that want to customize `device_transform` should define overloads of
  // transform_device_argument that are find-able by ADL.
  template <typename _Arg>
  using __transform_result_t = __remove_rvalue_reference_t<decltype(transform_device_argument(
    ::cuda::stream_ref{::cudaStream_t{}}, ::cuda::std::declval<_Arg>()))>;

  template <typename _Arg>
  using __transformed_argument_t =
    __remove_rvalue_reference_t<decltype(::cuda::std::declval<_Arg>().transformed_argument())>;

#if _CCCL_COMPILER(MSVC)
  // MSVC has a bug where it rejects the use of an immovable type in a defaulted function
  // argument.
  // https://developercommunity.visualstudio.com/t/rejects-valid-cannot-use-an-immovable/10932844
  template <typename _Arg>
  struct __optional : execution::__variant<_Arg>
  {
    __optional() = default;
    __optional(__optional&&); // declared but not defined
  };
#else
  template <typename _Arg>
  using __optional = execution::__variant<_Arg>;
#endif

  // The use of `__variant` here is to move the destruction of the object returned from
  // transform_device_argument into the caller's stack frame. Objects created for default arguments
  // are located in the caller's stack frame. This is so that a use of `device_transform`
  // such as:
  //
  //   kernel<<<grid, block, 0, stream>>>(device_transform(stream, arg));
  //
  // is equivalent to:
  //
  //   kernel<<<grid, block, 0, stream>>>(transform_device_argument(stream, arg).transformed_argument());
  //
  // where the object returned from `transform_device_argument` is destroyed *after* the kernel
  // launch.
  //
  // What I really wanted to do was:
  //
  //   template <typename Arg>
  //   auto operator()(::cuda::stream_ref stream, Arg&& arg, auto&& action = transform_device_argument(arg))
  //
  // but sadly that is not valid C++.
  _CCCL_TEMPLATE(typename _Stream, typename _Arg)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Stream, ::cuda::stream_ref> _CCCL_AND(
    !::cuda::std::is_reference_v<__transform_result_t<_Arg>>))
  [[nodiscard]] _CCCL_HOST_API auto
  operator()(_Stream&& __stream, _Arg&& __arg, __optional<__transform_result_t<_Arg>> __storage = {}) const
    -> decltype(auto)
  {
    // Calls to transform_device_argument are intentionally unqualified so as to use ADL.
    if constexpr (__is_instantiable_with<__transformed_argument_t, __transform_result_t<_Arg>>)
    {
      return _CCCL_MOVE(__storage.__emplace_from([&] {
               return transform_device_argument(__stream, ::cuda::std::forward<_Arg>(__arg));
             }))
        .transformed_argument();
    }
    else
    {
      return _CCCL_MOVE(__storage.__emplace_from([&] {
        return transform_device_argument(__stream, ::cuda::std::forward<_Arg>(__arg));
      }));
    }
  }

  // If transform_device_argument returns a reference type, then there are no pre- and
  // post-launch actions. (References types don't have ctors/dtors.) There is no need to
  // store the result of transform_device_argument.
  _CCCL_TEMPLATE(typename _Stream, typename _Arg)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Stream, ::cuda::stream_ref>
                   _CCCL_AND ::cuda::std::is_reference_v<__transform_result_t<_Arg>>)
  [[nodiscard]] _CCCL_HOST_API auto operator()(_Stream&& __stream, _Arg&& __arg) const -> decltype(auto)
  {
    // Calls to transform_device_argument are intentionally unqualified so as to use ADL.
    if constexpr (__is_instantiable_with<__transformed_argument_t, __transform_result_t<_Arg>>)
    {
      return transform_device_argument(__stream, ::cuda::std::forward<_Arg>(__arg)).transformed_argument();
    }
    else
    {
      return transform_device_argument(__stream, ::cuda::std::forward<_Arg>(__arg));
    }
  }

  template <typename _Arg>
  [[nodiscard]] _CCCL_HOST_API auto operator()(::cuda::std::__ignore_t, _Arg&& __arg) const -> decltype(auto)
  {
    if constexpr (__is_instantiable_with<__transformed_argument_t, _Arg>)
    {
      return ::cuda::std::forward<_Arg>(__arg).transformed_argument();
    }
    else
    {
      return static_cast<_Arg>(::cuda::std::forward<_Arg>(__arg));
    }
  }
};
} // namespace __tfx

_CCCL_GLOBAL_CONSTANT auto device_transform = __tfx::__device_transform_t{};

template <typename _Arg>
using transformed_device_argument_t _CCCL_NODEBUG_ALIAS =
  __remove_rvalue_reference_t<::cuda::std::__call_result_t<__tfx::__device_transform_t, ::cuda::stream_ref, _Arg>>;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // !_CUDAX__STREAM_DEVICE_TRANSFORM_CUH
