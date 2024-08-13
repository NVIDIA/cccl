/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CUDAX__CONTAIERS_ASYNC_UNINITIALIZED_BUFFER
#define __CUDAX__CONTAIERS_ASYNC_UNINITIALIZED_BUFFER

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/stream.cuh>

namespace cuda::experimental
{
template <typename _Tp, typename... _Properties>
class async_uninitialized_buffer
{
private:
  _CUDA_VMR::async_resource_ref<_Properties...> __mr_;
  size_t __count_ = 0;
  _Tp* __buf_     = nullptr;
  ::cuda::stream_ref __alloc_stream_{};

  _CCCL_NODISCARD_FRIEND ::cuda::std::span<_Tp>
  __cudax_launch_transform(::cuda::stream_ref __launch_stream, const async_uninitialized_buffer& __self)
  {
    if (__self.__alloc_stream_ != __launch_stream)
    {
      // Record an event on the allocation stream and wait on it in the launch
      // stream. TODO: The need to create an owning stream here is a temporary
      // work-around until stream_ref has a way to wait on an event.
      auto __launch_stream2 = stream::from_native_handle(__launch_stream.get());
      __launch_stream2.wait(__self.__alloc_stream_); // BUGBUG potentially throwing
      (void) __launch_stream2.release();
    }
    return {__self.__buf_, __self.__count_};
  }

  //! @brief Determines the properly aligned start of the buffer given the alignment and size of  `T`
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _Tp* __get_data() const noexcept
  {
    return _CUDA_VSTD::launder(__buf_);
    // constexpr size_t __alignment = alignof(_Tp);
    // size_t __space               = __get_allocation_size(__count_);
    // void* __ptr                  = __buf_;
    // return _CUDA_VSTD::launder(
    //   reinterpret_cast<_Tp*>(_CUDA_VSTD::align(__alignment, __count_ * sizeof(_Tp), __ptr, __space)));
  }

public:
  using value_type = _Tp;
  using reference  = _Tp&;
  using pointer    = _Tp*;
  using size_type  = size_t;

  async_uninitialized_buffer(
    _CUDA_VMR::async_resource_ref<_Properties...> __mr, const size_t __count, ::cuda::stream_ref __stream)
      : __mr_(__mr)
      , __count_(__count)
      , __buf_(__count == 0
                 ? nullptr
                 : static_cast<_Tp*>(__mr.allocate_async(__count * sizeof(_Tp), alignof(max_align_t), __stream)))
      , __alloc_stream_(__stream)
  {}

  async_uninitialized_buffer(
    _Tp, _CUDA_VMR::async_resource_ref<_Properties...> __mr, const size_t __count, ::cuda::stream_ref __stream)
      : async_uninitialized_buffer(__mr, __count, __stream)
  {}

  async_uninitialized_buffer(const async_uninitialized_buffer&)            = delete;
  async_uninitialized_buffer& operator=(const async_uninitialized_buffer&) = delete;

  //! @brief Move construction
  //! @param __other Another \c async_uninitialized_buffer
  async_uninitialized_buffer(async_uninitialized_buffer&& __other) noexcept
      : __mr_(__other.__mr_)
      , __count_(_CUDA_VSTD::exchange(__other.__count_, {}))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, {}))
      , __alloc_stream_(_CUDA_VSTD::exchange(__other.__alloc_stream_, {}))
  {}

  async_uninitialized_buffer& operator=(async_uninitialized_buffer&& __other) noexcept
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      async_uninitialized_buffer(_CUDA_VSTD::move(__other)).swap(*this);
    }
    return *this;
  }

  ~async_uninitialized_buffer()
  {
    if (__buf_)
    {
      __mr_.deallocate_async(__buf_, __count_ * sizeof(_Tp), alignof(max_align_t), __alloc_stream_);
    }
  }

  //! @brief Returns an aligned pointer to the buffer
  _CCCL_NODISCARD _CCCL_HOST_DEVICE pointer begin() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns an aligned pointer to end of the buffer
  _CCCL_NODISCARD _CCCL_HOST_DEVICE pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  //! @brief Returns an aligned pointer to the buffer
  _CCCL_NODISCARD _CCCL_HOST_DEVICE pointer data() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns the size of the buffer
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr size_type size() const noexcept
  {
    return __count_;
  }

  //! @rst
  //! Returns the :ref:`async_resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` used to allocate
  //! the buffer
  //! @endrst
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CUDA_VMR::async_resource_ref<_Properties...> resource() const noexcept
  {
    return __mr_;
  }

  //! @brief Swaps the contents with those of another \c uninitialized_buffer
  //! @param __other The other \c uninitialized_buffer.
  _CCCL_HOST_DEVICE constexpr void swap(async_uninitialized_buffer& __other) noexcept
  {
    _CUDA_VSTD::swap(__mr_, __other.__mr_);
    _CUDA_VSTD::swap(__count_, __other.__count_);
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
    _CUDA_VSTD::swap(__alloc_stream_, __other.__alloc_stream_);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed Properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(
    (!::cuda::property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend constexpr void get_property(const async_uninitialized_buffer&, _Property) noexcept {}
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

_LIBCUDACXX_TEMPLATE(typename _Tp, typename _AsyncResource)
_LIBCUDACXX_REQUIRES(
  _CUDA_VMR::async_resource_with<_AsyncResource, _CUDA_VMR::device_accessible, _CUDA_VMR::host_accessible>)
async_uninitialized_buffer(_Tp, _AsyncResource&, size_t, ::cuda::stream_ref)
  -> async_uninitialized_buffer<_Tp, _CUDA_VMR::device_accessible, _CUDA_VMR::host_accessible>;

_LIBCUDACXX_TEMPLATE(typename _Tp, typename _AsyncResource)
_LIBCUDACXX_REQUIRES(_CUDA_VMR::async_resource_with<_AsyncResource, _CUDA_VMR::device_accessible> _LIBCUDACXX_AND(
  !_CUDA_VMR::async_resource_with<_AsyncResource, _CUDA_VMR::host_accessible>))
async_uninitialized_buffer(_Tp, _AsyncResource&, size_t, ::cuda::stream_ref)
  -> async_uninitialized_buffer<_Tp, _CUDA_VMR::device_accessible>;

_LIBCUDACXX_TEMPLATE(typename _Tp, typename _AsyncResource)
_LIBCUDACXX_REQUIRES(_CUDA_VMR::async_resource_with<_AsyncResource, _CUDA_VMR::host_accessible> _LIBCUDACXX_AND(
  !_CUDA_VMR::async_resource_with<_AsyncResource, _CUDA_VMR::device_accessible>))
async_uninitialized_buffer(_Tp, _AsyncResource&, size_t, ::cuda::stream_ref)
  -> async_uninitialized_buffer<_Tp, _CUDA_VMR::host_accessible>;

} // namespace cuda::experimental

#endif // __CUDAX__CONTAIERS_ASYNC_UNINITIALIZED_BUFFER
