//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, c++98, c++03, c++11, c++14, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// static constexpr bool is_always_lock_free;

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

// NVRTC doesn't include host atomic making this feature test invalid
// TODO: Should we define __cccl_lib_atomic_is_always_lock_free for NVRTC?
#if !defined(TEST_COMPILER_NVRTC)
#  if !defined(__cccl_lib_atomic_is_always_lock_free)
#    error Feature test macro missing.
#  endif
#endif

template <typename T>
__host__ __device__ void checkAlwaysLockFree()
{
  if (cuda::std::atomic<T>::is_always_lock_free)
  {
    assert(cuda::std::atomic<T>().is_lock_free());
  }
}

// FIXME: This separate test is needed to work around llvm.org/PR31864
// which causes LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE to be defined as '1' in 32-bit builds
// even though __atomic_always_lock_free returns true for the same type.
constexpr bool NeedWorkaroundForPR31864 =
#if defined(__clang__)
  (sizeof(void*) == 4); // Needed on 32 bit builds
#else
  false;
#endif

template <bool Disable                      = NeedWorkaroundForPR31864,
          cuda::std::enable_if_t<!Disable>* = nullptr,
          class LLong                       = long long,
          class ULLong                      = unsigned long long>
__host__ __device__ void checkLongLongTypes()
{
  static_assert(cuda::std::atomic<LLong>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE), "");
  static_assert(cuda::std::atomic<ULLong>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE), "");
}

// Used to make the calls to __atomic_always_lock_free dependent on a template
// parameter.
template <class T>
__host__ __device__ constexpr size_t getSizeOf()
{
  return sizeof(T);
}

template <bool Enable                     = NeedWorkaroundForPR31864,
          cuda::std::enable_if_t<Enable>* = nullptr,
          class LLong                     = long long,
          class ULLong                    = unsigned long long>
__host__ __device__ void checkLongLongTypes()
{
  constexpr bool ExpectLockFree = __atomic_always_lock_free(getSizeOf<LLong>(), 0);
  static_assert(cuda::std::atomic<LLong>::is_always_lock_free == ExpectLockFree, "");
  static_assert(cuda::std::atomic<ULLong>::is_always_lock_free == ExpectLockFree, "");
  static_assert((0 != LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE) == ExpectLockFree, "");
}

__host__ __device__ void run()
{
// structs and unions can't be defined in the template invocation.
// Work around this with a typedef.
#define CHECK_ALWAYS_LOCK_FREE(T) \
  do                              \
  {                               \
    typedef T type;               \
    checkAlwaysLockFree<type>();  \
  } while (0)

  CHECK_ALWAYS_LOCK_FREE(bool);
  CHECK_ALWAYS_LOCK_FREE(char);
  CHECK_ALWAYS_LOCK_FREE(signed char);
  CHECK_ALWAYS_LOCK_FREE(unsigned char);
  CHECK_ALWAYS_LOCK_FREE(char16_t);
  CHECK_ALWAYS_LOCK_FREE(char32_t);
  CHECK_ALWAYS_LOCK_FREE(wchar_t);
  CHECK_ALWAYS_LOCK_FREE(short);
  CHECK_ALWAYS_LOCK_FREE(unsigned short);
  CHECK_ALWAYS_LOCK_FREE(int);
  CHECK_ALWAYS_LOCK_FREE(unsigned int);
  CHECK_ALWAYS_LOCK_FREE(long);
  CHECK_ALWAYS_LOCK_FREE(unsigned long);
  CHECK_ALWAYS_LOCK_FREE(long long);
  CHECK_ALWAYS_LOCK_FREE(unsigned long long);
  CHECK_ALWAYS_LOCK_FREE(cuda::std::nullptr_t);
  CHECK_ALWAYS_LOCK_FREE(void*);
  CHECK_ALWAYS_LOCK_FREE(float);
  CHECK_ALWAYS_LOCK_FREE(double);
  // CHECK_ALWAYS_LOCK_FREE(long double); // long double is not supported
#if _CCCL_HAS_ATTRIBUTE(vector_size) && defined(_LIBCUDACXX_VERSION) && !defined(__CUDACC__)
  // NOTE: NVCC doesn't support the vector_size attribute in device code.
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(1 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(2 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(4 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(16 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(32 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(1 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(2 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(4 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(16 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(32 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(1 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(2 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(4 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(16 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(32 * sizeof(double)))));
#endif // _CCCL_HAS_ATTRIBUTE(vector_size) && defined(_LIBCUDACXX_VERSION)
  CHECK_ALWAYS_LOCK_FREE(struct Empty{});
  CHECK_ALWAYS_LOCK_FREE(struct OneInt { int i; });
  CHECK_ALWAYS_LOCK_FREE(struct IntArr2 { int i[2]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr2 { long long int i[2]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr4 { long long int i[4]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr8 { long long int i[8]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr16 { long long int i[16]; });
  CHECK_ALWAYS_LOCK_FREE(struct Padding {
    char c; /* padding */
    long long int i;
  });
  CHECK_ALWAYS_LOCK_FREE(union IntFloat {
    int i;
    float f;
  });

  // C macro and static constexpr must be consistent.
  static_assert(cuda::std::atomic<bool>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE), "");
  static_assert(cuda::std::atomic<char>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(cuda::std::atomic<signed char>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(cuda::std::atomic<unsigned char>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(cuda::std::atomic<char16_t>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE), "");
  static_assert(cuda::std::atomic<char32_t>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE), "");
  static_assert(cuda::std::atomic<wchar_t>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE), "");
  static_assert(cuda::std::atomic<short>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE), "");
  static_assert(cuda::std::atomic<unsigned short>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE), "");
  static_assert(cuda::std::atomic<int>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_INT_LOCK_FREE), "");
  static_assert(cuda::std::atomic<unsigned int>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_INT_LOCK_FREE), "");
  static_assert(cuda::std::atomic<long>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_LONG_LOCK_FREE), "");
  static_assert(cuda::std::atomic<unsigned long>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_LONG_LOCK_FREE), "");
  checkLongLongTypes();
  static_assert(cuda::std::atomic<void*>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE), "");
  static_assert(
    cuda::std::atomic<cuda::std::nullptr_t>::is_always_lock_free == (2 == LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE), "");
}

int main(int, char**)
{
  run();
  return 0;
}
