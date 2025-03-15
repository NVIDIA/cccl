#ifndef _LIBCUDACXX___COMPLEX_STD_COMPLEX_ACCESS_H
#define _LIBCUDACXX___COMPLEX_STD_COMPLEX_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__type_traits/is_constant_evaluated.h>

#  include <complex>

#  if !defined(__CUDA_ARCH__) || defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#    define _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS       constexpr
#    define _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS() 1
#  else // ^^^ _CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^ / vvv !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED vvv
#    define _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS
#    define _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS() 0
#  endif // ^^^ !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED ^^^

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS _Tp
__get_std_complex_real(const ::std::complex<_Tp>& __v) noexcept
{
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return __v.real();
  }
  else
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __v.real();), (return reinterpret_cast<const _Tp(&)[2]>(__v)[0];))
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS _Tp
__get_std_complex_imag(const ::std::complex<_Tp>& __v) noexcept
{
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return __v.imag();
  }
  else
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __v.imag();), (return reinterpret_cast<const _Tp(&)[2]>(__v)[1];))
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _LIBCUDACXX___COMPLEX_STD_COMPLEX_ACCESS_H
