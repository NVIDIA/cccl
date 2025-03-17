// This source file checks that:
// 1) Header <@header@> compiles without error.
// 2) Common macro collisions with platform/system headers are avoided.
// 3) half/bf16 aren't included when these are explicitly disabled.

// Define CUDAX_MACRO_CHECK(macro, header), which emits a diagnostic indicating
// a potential macro collision and halts.
//
// Use raw platform checks instead of the CCCL macros since we
// don't want to #include any headers other than the one being tested.
//
// This is only implemented for MSVC/GCC/Clang.
#if defined(_MSC_VER) // MSVC

// Fake up an error for MSVC
#  define CUDAX_MACRO_CHECK_IMPL(msg)             \
    /* Print message that looks like an error: */ \
    __pragma(message(__FILE__ ":" CUDAX_MACRO_CHECK_IMPL0(__LINE__) ": error: " #msg)) static_assert(false, #msg);
#  define CUDAX_MACRO_CHECK_IMPL0(x) CUDAX_MACRO_CHECK_IMPL1(x)
#  define CUDAX_MACRO_CHECK_IMPL1(x) #x

#elif defined(__clang__) || defined(__GNUC__)

// GCC/clang are easy:
#  define CUDAX_MACRO_CHECK_IMPL(msg)   CUDAX_MACRO_CHECK_IMPL0(GCC error #msg)
#  define CUDAX_MACRO_CHECK_IMPL0(expr) _Pragma(#expr)

#endif

// Hacky way to build a string, but it works on all tested platforms.
#define CUDAX_MACRO_CHECK(MACRO, HEADER) \
  CUDAX_MACRO_CHECK_IMPL(Identifier MACRO should not be used from CCCL headers due to conflicts with HEADER macros.)

// complex.h conflicts
#define I CUDAX_MACRO_CHECK('I', complex.h)

// windows.h conflicts
// @eniebler 2024-08-30: This test is disabled because it causes build
// failures in some configurations.
// #define small CUDAX_MACRO_CHECK('small', windows.h)
// We can't enable these checks without breaking some builds -- some standard
// library implementations unconditionally `#undef` these macros, which then
// causes random failures later.
// Leaving these commented out as a warning: Here be dragons.
// #define min(...) CUDAX_MACRO_CHECK('min', windows.h)
// #define max(...) CUDAX_MACRO_CHECK('max', windows.h)

// termios.h conflicts (NVIDIA/thrust#1547)
#define B0 CUDAX_MACRO_CHECK("B0", termios.h)

#include <@header@>

#if defined(CCCL_DISABLE_BF16_SUPPORT)
#  if defined(__CUDA_BF16_TYPES_EXIST__)
#    error We should not include cuda_bf16.h when BF16 support is disabled
#  endif // __CUDA_BF16_TYPES_EXIST__
#endif // CCCL_DISABLE_BF16_SUPPORT

#if defined(CCCL_DISABLE_FP16_SUPPORT)
#  if defined(__CUDA_FP16_TYPES_EXIST__)
#    error We should not include cuda_fp16.h when half support is disabled
#  endif // __CUDA_FP16_TYPES_EXIST__
#  if defined(__CUDA_BF16_TYPES_EXIST__)
#    error We should not include cuda_bf16.h when half support is disabled
#  endif // __CUDA_BF16_TYPES_EXIST__
#endif // CCCL_DISABLE_FP16_SUPPORT
