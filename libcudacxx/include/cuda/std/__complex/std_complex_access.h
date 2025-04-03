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

#  include <complex>

#  if !defined(__CUDA_ARCH__) || _CCCL_HAS_RELAXED_CONSTEXPR()
#    define _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS       constexpr
#    define _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS() 1
#  else // ^^^ can directly access std::complex ^^^ / vvv cannot directly access std::complex vvv
#    define _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS
#    define _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS() 0
#  endif // ^^^ cannot directly access std::complex ^^^

// silence warning about using non-standard floating point types in std::complex being unspecified behavior
// todo: specialize std::complex for extended floating point types (STL team recommends)
#  if _CCCL_COMPILER(MSVC)
_CCCL_NV_DIAG_SUPPRESS(1444)
#  endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS _Tp
__get_std_complex_real(const ::std::complex<_Tp>& __v) noexcept
{
#  if _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS()
  return __v.real();
#  else // ^^^ can directly access std::complex ^^^ / vvv cannot directly access std::complex vvv
  return reinterpret_cast<const _Tp(&)[2]>(__v)[0];
#  endif // ^^^ cannot directly access std::complex ^^^
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS _Tp
__get_std_complex_imag(const ::std::complex<_Tp>& __v) noexcept
{
#  if _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS()
  return __v.imag();
#  else // ^^^ can directly access std::complex ^^^ / vvv cannot directly access std::complex vvv
  return reinterpret_cast<const _Tp(&)[2]>(__v)[1];
#  endif // ^^^ cannot directly access std::complex ^^^
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_STD_COMPLEX_ACCESS ::std::complex<_Tp>
__make_std_complex(const _Tp& __r = _Tp(), const _Tp& __i = _Tp()) noexcept
{
#  if _LIBCUDACXX_HAS_CONSTEXPR_STD_COMPLEX_ACCESS()
  return ::std::complex<_Tp>{__r, __i};
#  else // ^^^ can directly access std::complex ^^^ / vvv cannot directly access std::complex vvv
  _CCCL_ALIGNAS_TYPE(::std::complex<_Tp>) _Tp __ret[]{__r, __i};
  return reinterpret_cast<::std::complex<_Tp>&>(__ret);
#  endif // ^^^ cannot directly access std::complex ^^^
}

_LIBCUDACXX_END_NAMESPACE_STD

#  if _CCCL_COMPILER(MSVC)
_CCCL_NV_DIAG_DEFAULT(1444)
#  endif // _CCCL_COMPILER(MSVC)

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _LIBCUDACXX___COMPLEX_STD_COMPLEX_ACCESS_H
