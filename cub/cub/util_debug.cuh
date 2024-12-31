/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Error and event logging routines.
 *
 * The following macros definitions are supported:
 * - \p CUB_LOG.  Simple event messages are printed to \p stdout.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target>

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:

/**
 * @def CUB_DEBUG_LOG
 *
 * Causes kernel launch configurations to be printed to the console
 */
#  define CUB_DEBUG_LOG

/**
 * @def CUB_DEBUG_SYNC
 *
 * Causes synchronization of the stream after every kernel launch to check
 * for errors. Also causes kernel launch configurations to be printed to the
 * console.
 */
#  define CUB_DEBUG_SYNC

/**
 * @def CUB_DEBUG_HOST_ASSERTIONS
 *
 * Extends `CUB_DEBUG_SYNC` effects by checking host-side precondition
 * assertions.
 */
#  define CUB_DEBUG_HOST_ASSERTIONS

/**
 * @def CUB_DEBUG_DEVICE_ASSERTIONS
 *
 * Extends `CUB_DEBUG_HOST_ASSERTIONS` effects by checking device-side
 * precondition assertions.
 */
#  define CUB_DEBUG_DEVICE_ASSERTIONS

/**
 * @def CUB_DEBUG_ALL
 *
 * Causes host and device-side precondition assertions to be checked. Apart
 * from that, causes synchronization of the stream after every kernel launch to
 * check for errors. Also causes kernel launch configurations to be printed to
 * the console.
 */
#  define CUB_DEBUG_ALL

#endif // _CCCL_DOXYGEN_INVOKED

// `CUB_DETAIL_DEBUG_LEVEL_*`: Implementation details, internal use only:

#define CUB_DETAIL_DEBUG_LEVEL_NONE                 0
#define CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY 1
#define CUB_DETAIL_DEBUG_LEVEL_LOG                  2
#define CUB_DETAIL_DEBUG_LEVEL_SYNC                 3
#define CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS      4
#define CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS    5
#define CUB_DETAIL_DEBUG_LEVEL_ALL                  1000

// `CUB_DEBUG_*`: User interfaces:

// Extra logging, no syncs
#ifdef CUB_DEBUG_LOG
#  define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_LOG
#endif

// Logging + syncs
#ifdef CUB_DEBUG_SYNC
#  define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_SYNC
#endif

// Logging + syncs + host assertions
#ifdef CUB_DEBUG_HOST_ASSERTIONS
#  define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS
#endif

// Logging + syncs + host assertions + device assertions
#ifdef CUB_DEBUG_DEVICE_ASSERTIONS
#  define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS
#endif

// All
#ifdef CUB_DEBUG_ALL
#  define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_ALL
#endif

// Default case, no extra debugging:
#ifndef CUB_DETAIL_DEBUG_LEVEL
#  ifdef NDEBUG
#    define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_NONE
#  else
#    define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY
#  endif
#endif

/*
 * `CUB_DETAIL_DEBUG_ENABLE_*`:
 * Internal implementation details, used for testing enabled debug features:
 */

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_LOG
#  define CUB_DETAIL_DEBUG_ENABLE_LOG
#endif

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_SYNC
#  define CUB_DETAIL_DEBUG_ENABLE_SYNC
#endif

#if (CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS) \
  || (CUB_DETAIL_DEBUG_LEVEL == CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY)
#  define CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS
#endif

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS
#  define CUB_DETAIL_DEBUG_ENABLE_DEVICE_ASSERTIONS
#endif

/// CUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG)) && !defined(CUB_STDERR)
#  define CUB_STDERR
#endif

#if defined(CUB_STDERR) || defined(CUB_DETAIL_DEBUG_ENABLE_LOG)
#  include <cstdio>
#endif

CUB_NAMESPACE_BEGIN

/**
 * \brief %If \p CUB_STDERR is defined and \p error is not \p cudaSuccess, the
 * corresponding error message is printed to \p stderr (or \p stdout in device
 * code) along with the supplied source context.
 *
 * \return The CUDA error.
 */
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t Debug(cudaError_t error, const char* filename, int line)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.

  // clang-format off
  #ifndef CUB_RDC_ENABLED
  #define CUB_TEMP_DEVICE_CODE
  #else
  #define CUB_TEMP_DEVICE_CODE last_error = cudaGetLastError()
  #endif

  cudaError_t last_error = cudaSuccess;

  NV_IF_TARGET(
    NV_IS_HOST,
    (last_error = cudaGetLastError();),
    (CUB_TEMP_DEVICE_CODE;)
  );

  #undef CUB_TEMP_DEVICE_CODE
  // clang-format on

  if (error == cudaSuccess && last_error != cudaSuccess)
  {
    error = last_error;
  }

#ifdef CUB_STDERR
  if (error)
  {
    NV_IF_TARGET(
      NV_IS_HOST,
      (fprintf(stderr, "CUDA error %d [%s, %d]: %s\n", error, filename, line, cudaGetErrorString(error));
       fflush(stderr);),
      (printf("CUDA error %d [block (%d,%d,%d) thread (%d,%d,%d), %s, %d]\n",
              error,
              blockIdx.z,
              blockIdx.y,
              blockIdx.x,
              threadIdx.z,
              threadIdx.y,
              threadIdx.x,
              filename,
              line);));
  }
#else
  (void) filename;
  (void) line;
#endif

  return error;
}

/**
 * \brief Debug macro
 */
#ifndef CubDebug
#  define CubDebug(e) CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)
#endif

/**
 * \brief Debug macro with exit
 */
#ifndef CubDebugExit
#  define CubDebugExit(e)                                               \
    if (CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)) \
    {                                                                   \
      exit(1);                                                          \
    }
#endif

/**
 * \brief Log macro for printf statements.
 */
#if !defined(_CubLog)
#  if defined(_NVHPC_CUDA) || !(defined(__clang__) && defined(__CUDA__))

// NVCC / NVC++
#    define _CubLog(format, ...)                                    \
      do                                                            \
      {                                                             \
        NV_IF_TARGET(                                               \
          NV_IS_HOST,                                               \
          (printf(format, __VA_ARGS__);),                           \
          (printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format, \
                  blockIdx.z,                                       \
                  blockIdx.y,                                       \
                  blockIdx.x,                                       \
                  threadIdx.z,                                      \
                  threadIdx.y,                                      \
                  threadIdx.x,                                      \
                  __VA_ARGS__);));                                  \
      } while (false)

#  else // Clang:

// XXX shameless hack for clang around variadic printf...
//     Compiles w/o supplying -std=c++11 but shows warning,
//     so we silence them :)
#    pragma clang diagnostic ignored "-Wc++11-extensions"
#    pragma clang diagnostic ignored "-Wunnamed-type-template-args"
#    ifdef CUB_STDERR
template <class... Args>
inline _CCCL_HOST_DEVICE void va_printf(char const* format, Args const&... args)
{
#      ifdef __CUDA_ARCH__
  printf(format, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, args...);
#      else
  printf(format, args...);
#      endif
}
#    else // !defined(CUB_STDERR)
template <class... Args>
inline _CCCL_HOST_DEVICE void va_printf(char const*, Args const&...)
{}
#    endif // !defined(CUB_STDERR)

#    ifndef __CUDA_ARCH__
#      define _CubLog(format, ...) CUB_NS_QUALIFIER::va_printf(format, __VA_ARGS__);
#    else
#      define _CubLog(format, ...)                               \
        CUB_NS_QUALIFIER::va_printf("[block (%d,%d,%d), thread " \
                                    "(%d,%d,%d)]: " format,      \
                                    __VA_ARGS__);
#    endif
#  endif
#endif

#define CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED             \
  CCCL_DEPRECATED_BECAUSE(                                         \
    "CUB no longer accepts `debug_synchronous` parameter. "        \
    "Define CUB_DEBUG_SYNC instead, or silence this message with " \
    "CCCL_IGNORE_DEPRECATED_API.")

#define CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG                     \
  if (debug_synchronous)                                            \
  {                                                                 \
    _CubLog("%s\n",                                                 \
            "CUB no longer accepts `debug_synchronous` parameter. " \
            "Define CUB_DEBUG_SYNC instead.");                      \
  }

CUB_NAMESPACE_END
