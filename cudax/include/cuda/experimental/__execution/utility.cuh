//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_UTILITY
#define __CUDAX_EXECUTION_UTILITY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/pod_tuple.h>
#include <cuda/std/initializer_list>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
_CCCL_GLOBAL_CONSTANT size_t __npos = static_cast<size_t>(-1);

struct __empty
{};

struct [[deprecated]] __deprecated
{};

struct __nil
{};

_CCCL_API constexpr auto __maximum(::cuda::std::initializer_list<size_t> __il) noexcept -> size_t
{
  size_t __max = 0;
  for (auto i : __il)
  {
    if (i > __max)
    {
      __max = i;
    }
  }
  return __max;
}

_CCCL_API constexpr auto __find_pos(bool const* const __begin, bool const* const __end) noexcept -> size_t
{
  for (bool const* __where = __begin; __where != __end; ++__where)
  {
    if (*__where)
    {
      return static_cast<size_t>(__where - __begin);
    }
  }
  return __npos;
}

template <class _Ty, class... _Ts>
_CCCL_API constexpr auto __index_of() noexcept -> size_t
{
  constexpr bool __map[] = {__same_as<_Ty, _Ts>...};
  return execution::__find_pos(__map, __map + sizeof...(_Ts));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Ty, class _Uy = _Ty>
_CCCL_API constexpr auto __exchange(_Ty& __obj, _Uy&& __new_value) noexcept -> _Ty
{
  constexpr bool __is_nothrow = //
    noexcept(_Ty(static_cast<_Ty&&>(__obj))) && //
    noexcept(__obj = static_cast<_Uy&&>(__new_value)); //
  static_assert(__is_nothrow);

  _Ty old_value = static_cast<_Ty&&>(__obj);
  __obj         = static_cast<_Uy&&>(__new_value);
  return old_value;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Ty>
_CCCL_API constexpr void __swap(_Ty& __left, _Ty& __right) noexcept
{
  constexpr bool __is_nothrow = //
    noexcept(_Ty(static_cast<_Ty&&>(__left))) && //
    noexcept(__left = static_cast<_Ty&&>(__right)); //
  static_assert(__is_nothrow);

  _Ty __tmp = static_cast<_Ty&&>(__left);
  __left    = static_cast<_Ty&&>(__right);
  __right   = static_cast<_Ty&&>(__tmp);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Ty>
[[nodiscard]] _CCCL_API constexpr auto __decay_copy(_Ty&& __ty) noexcept(__nothrow_decay_copyable<_Ty>) -> decay_t<_Ty>
{
  return static_cast<_Ty&&>(__ty);
}

[[nodiscard]] _CCCL_HOST_API inline auto __get_pointer_attributes(const void* __pv) -> ::cudaPointerAttributes
{
  ::cudaPointerAttributes __attrs;
  _CCCL_TRY_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__attrs, __pv);
  return __attrs;
}

// This function can only be called from a catch handler.
[[nodiscard]] _CCCL_HOST_API inline auto __get_cuda_error_from_active_exception() -> ::cudaError_t
{
  try
  {
    throw; // rethrow the active exception
  }
  catch (::cuda::cuda_error& __err)
  {
    return __err.status();
  }
  catch (::std::bad_alloc&)
  {
    return ::cudaErrorMemoryAllocation;
  }
  catch (...)
  {
    return ::cudaErrorUnknown; // fallback if no cuda error is found
  }
  _CCCL_UNREACHABLE();
}

template <class _Ty>
struct __managed_box : private __immovable
{
  using value_type = _Ty;

  _CCCL_HIDE_FROM_ABI __managed_box() = default;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(::cuda::std::constructible_from<_Ty, _Args...>)
  _CCCL_HOST_API explicit __managed_box(_Args&&... __args) noexcept(__nothrow_constructible<_Ty, _Args...>)
      : __value{static_cast<_Args&&>(__args)...}
  {
    _CCCL_ASSERT(execution::__get_pointer_attributes(this).type == cudaMemoryTypeManaged,
                 "__managed_box must be allocated in managed memory");
  }

  template <class... _Args>
  _CCCL_HOST_API static auto __make_unique(_Args&&... __args) -> ::cuda::std::unique_ptr<__managed_box>
  {
    return ::cuda::std::make_unique<__managed_box>(static_cast<_Args&&>(__args)...);
  }

  _CCCL_HOST_API static auto operator new(size_t __size) -> void*
  {
    void* __ptr = nullptr;
    _CCCL_TRY_CUDA_API(::cudaMallocManaged, "cudaMallocManaged failed", &__ptr, __size);
    ::cuda::std::ignore = ::cudaDeviceSynchronize(); // Ensure the memory is allocated before returning it.
    return __ptr;
  }

  _CCCL_HOST_API static void operator delete(void* __ptr, size_t) noexcept
  {
    ::cuda::std::ignore = ::cudaDeviceSynchronize(); // Ensure all operations on the memory are complete.
    ::cuda::std::ignore = ::cudaFree(__ptr);
  }

  value_type __value;

private:
  // Prevent the construction of __managed_box without dynamic allocation.
  friend struct ::cuda::std::default_delete<__managed_box<_Ty>>;
  ~__managed_box() = default;
};

//! @brief A callable that wraps a set of functions and calls the first one that is
//! callable with a given set of arguments.
template <class... _Fns>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __first_callable
{
private:
  //! @brief Returns the first function that is callable with a given set of arguments.
  template <class... _Args, class _Self>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto __get_1st(_Self&& __self) noexcept -> decltype(auto)
  {
    // NOLINTNEXTLINE (modernize-avoid-c-arrays)
    constexpr bool __flags[] = {__callable<::cuda::std::__copy_cvref_t<_Self, _Fns>, _Args...>..., false};
    constexpr size_t __idx   = execution::__find_pos(__flags, __flags + sizeof...(_Fns));
    if constexpr (__idx != __npos)
    {
      return ::cuda::std::__get<__idx>(static_cast<_Self&&>(__self).__fns_);
    }
  }

  //! @brief Alias for the type of the first function that is callable with a given set of arguments.
  template <class _Self, class... _Args>
  using __1st_fn_t _CCCL_NODEBUG_ALIAS = decltype(__first_callable::__get_1st<_Args...>(declval<_Self>()));

public:
  //! @brief Calls the first function that is callable with a given set of arguments.
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_NODEBUG_API constexpr auto
  operator()(_Args&&... __args) && noexcept(__nothrow_callable<__1st_fn_t<__first_callable, _Args...>, _Args...>)
    -> __call_result_t<__1st_fn_t<__first_callable, _Args...>, _Args...>
  {
    return __first_callable::__get_1st<_Args...>(static_cast<__first_callable&&>(*this))(
      static_cast<_Args&&>(__args)...);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_NODEBUG_API constexpr auto operator()(_Args&&... __args) const& noexcept(
    __nothrow_callable<__1st_fn_t<__first_callable const&, _Args...>, _Args...>)
    -> __call_result_t<__1st_fn_t<__first_callable const&, _Args...>, _Args...>
  {
    return __first_callable::__get_1st<_Args...>(*this)(static_cast<_Args&&>(__args)...);
  }

  ::cuda::std::__tuple<_Fns...> __fns_;
};

template <class... _Fns>
_CCCL_HOST_DEVICE __first_callable(_Fns...) -> __first_callable<_Fns...>;

//////////////////////////////////////////////////////////////////////////////////////////
// __call_or
namespace __detail
{
// query an environment, or return a default value if the query is not supported
struct __call_or_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn, class _Default, class... _Args)
  _CCCL_REQUIRES(__callable<_Fn, _Args...>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Fn&& __fn, _Default&&, _Args&&... __args) const
    noexcept(__nothrow_callable<_Fn, _Args...>) -> __call_result_t<_Fn, _Args...>
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Default, class... _Args>
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::std::__ignore_t, _Default&& __default, _Args&&...) const
    noexcept(__nothrow_movable<_Default>) -> _Default
  {
    return static_cast<_Default&&>(__default);
  }
};
} // namespace __detail

_CCCL_GLOBAL_CONSTANT __detail::__call_or_t __call_or{};

template <class _Fn, class _Default, class... _Args>
using __call_result_or_t _CCCL_NODEBUG_ALIAS =
  decltype(__call_or(::cuda::std::declval<_Fn>(), ::cuda::std::declval<_Default>(), ::cuda::std::declval<_Args>()...));

//! @brief A callable that always return a value of type _Ty, regardless of the arguments
//! passed to it.
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __always
{
  template <class... _Args>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Args&&...) && noexcept -> _Ty&&
  {
    return static_cast<_Ty&&>(__value);
  }

  template <class... _Args>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Args&&...) const& noexcept -> _Ty const&
  {
    return __value;
  }

  _CCCL_NO_UNIQUE_ADDRESS _Ty __value{};
};

template <class _Ty>
_CCCL_HOST_DEVICE __always(_Ty) -> __always<_Ty>;

template <class _Ty, class... _Us>
using __unless_one_of_t = ::cuda::std::enable_if_t<__none_of<_Ty, _Us...>, _Ty>;

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")
_CCCL_DIAG_SUPPRESS_NVHPC(probable_guiding_friend)
_CCCL_BEGIN_NV_DIAG_SUPPRESS(probable_guiding_friend)

// __zip/__unzip is for keeping type names short. It has the unfortunate side
// effect of obfuscating the types.
namespace
{
template <size_t _Ny>
struct __slot
{
  friend constexpr auto __slot_allocated(__slot<_Ny>);
};

template <class _Type, size_t _Ny>
struct __allocate_slot
{
  static constexpr size_t __value = _Ny;

  friend constexpr auto __slot_allocated(__slot<_Ny>)
  {
    return static_cast<_Type (*)()>(nullptr);
  }
};

template <class _Type, size_t _Id = 0, size_t _Pow2 = 0>
constexpr auto __next(long) -> size_t;

// If __slot_allocated(__slot<_Id>) has NOT been defined, then SFINAE will keep
// this function out of the overload set...
template <class _Type, //
          size_t _Id   = 0,
          size_t _Pow2 = 0,
          bool         = !__slot_allocated(__slot<_Id + (1 << _Pow2) - 1>())>
constexpr auto __next(int) -> size_t
{
  return execution::__next<_Type, _Id, _Pow2 + 1>(0);
}

template <class _Type, size_t _Id, size_t _Pow2>
constexpr auto __next(long) -> size_t
{
  if constexpr (_Pow2 == 0)
  {
    return __allocate_slot<_Type, _Id>::__value;
  }
  else
  {
    return execution::__next<_Type, _Id + (1 << (_Pow2 - 1)), 0>(0);
  }
}

// Prior to Clang 12, we can't use the __slot trick to erase long type names
// because of a compiler bug. We'll just use the original type name in that case.
#if _CCCL_COMPILER(CLANG, <, 12)

template <class _Type>
using __zip _CCCL_NODEBUG_ALIAS = _Type;

template <class _Id>
using __unzip _CCCL_NODEBUG_ALIAS = _Id;

#else // ^^^ _CCCL_COMPILER(CLANG, <, 12) ^^^ / vvv !_CCCL_COMPILER(CLANG, <, 12) vvv

template <class _Type, size_t _Val = execution::__next<_Type>(0)>
using __zip _CCCL_NODEBUG_ALIAS = __slot<_Val>;

template <class _Id>
using __unzip _CCCL_NODEBUG_ALIAS = decltype(__slot_allocated(_Id())());

#endif // ^^^ !_CCCL_COMPILER(CLANG, <, 12) ^^^

// burn the first slot
using __ignore_this_typedef [[maybe_unused]] = __zip<void>;
} // namespace

_CCCL_END_NV_DIAG_SUPPRESS()
_CCCL_DIAG_POP

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_UTILITY
