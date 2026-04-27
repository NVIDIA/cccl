/*===---- HostJIT CUDA runtime wrapper - replaces clang's wrapper ----------===
 *
 * This is a self-contained replacement for clang's __clang_cuda_runtime_wrapper.h.
 * Instead of #include_next-ing the real wrapper (which has fragile ordering
 * dependencies on system headers and CUDA toolkit version-specific branches),
 * we directly include only the clang-provided CUDA helper headers we need and
 * pull in the CUDA toolkit headers with explicit preprocessor guards.
 *
 * Key design decision: all clang-provided device function implementations and
 * CCCL-required intrinsics are defined BEFORE any CUDA toolkit headers that
 * might transitively include CCCL (via libcudacxx standard headers on our
 * include path). This eliminates the need for forward declarations.
 *
 * Assumptions:
 *   - CUDA >= 9.0  (no legacy code paths)
 *   - Clang CUDA compilation (__CUDA__ && __clang__)
 *   - Freestanding: all standard headers are stubs or from libcudacxx
 *   - cuda::std is bridged into std via using-directive
 *===-----------------------------------------------------------------------===*/
#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
#define __CLANG_CUDA_RUNTIME_WRAPPER_H__
#pragma clang system_header

#if defined(__CUDA__) && defined(__clang__)

// ============================================================================
// Phase 1: Forward-declare device math overloads before any <cmath> inclusion
// ============================================================================
// This prevents constexpr std library math functions from becoming implicitly
// host+device, which would block our __device__ overloads later.
#  include <__clang_cuda_math_forward_declares.h>

// ============================================================================
// Phase 2: Device-side definitions before any CUDA toolkit headers
// ============================================================================
// Everything here uses only compiler builtins and our stubs. No CUDA toolkit
// headers are included yet, so nothing can transitively pull in CCCL.

#  pragma push_macro("__THROW")
#  pragma push_macro("__CUDA_ARCH__")

#  ifndef __CUDA_ARCH__
#    define __CUDA_ARCH__ 9999
#  endif

// host_defines.h provides __device__, __host__, __forceinline__ macros.
// Its only transitive dep (ctype.h) hits our stub.
#  define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#  define __CUDACC__
#  define __CUDA_LIBDEVICE__
#  include "host_defines.h"

// ---- Builtin variables (threadIdx, blockIdx, etc.) ----
#  include "__clang_cuda_builtin_vars.h"

// ---- Stubs needed by clang device function headers below ----
#  include <climits>
#  include <cmath>
#  include <cstddef>

// ---- Clang device function wrappers (local copies, CUDA < 9.0 removed) ----
// clang-format off
// Order matters: libdevice_declares must precede device_functions (declares __nv_* builtins used there).
#  include "__clang_cuda_libdevice_declares.h"
#  include "__clang_cuda_device_functions.h"
#  include "__clang_cuda_math.h"
// clang-format on

// ---- Address-space intrinsics needed by CCCL headers ----
// (e.g. cuda/__memory/address_space.h, cuda/__ptx/ptx_helper_functions.h)
static __device__ __forceinline__ __attribute__((const)) unsigned int __isGlobal(const void* p)
{
  return __nvvm_isspacep_global(p);
}
static __device__ __forceinline__ __attribute__((const)) unsigned int __isShared(const void* p)
{
  return __nvvm_isspacep_shared(p);
}
static __device__ __forceinline__ __attribute__((const)) unsigned int __isConstant(const void* p)
{
  return __nvvm_isspacep_const(p);
}
static __device__ __forceinline__ __attribute__((const)) unsigned int __isLocal(const void* p)
{
  return __nvvm_isspacep_local(p);
}
#  define __FWD_DEVICE static __device__ __forceinline__
__FWD_DEVICE unsigned int __isClusterShared(const void*);
__FWD_DEVICE __SIZE_TYPE__ __cvta_generic_to_shared(const void*);
__FWD_DEVICE __SIZE_TYPE__ __cvta_generic_to_global(const void*);
__FWD_DEVICE void* __cvta_shared_to_generic(__SIZE_TYPE__);
__FWD_DEVICE void* __cvta_global_to_generic(__SIZE_TYPE__);
#  undef __FWD_DEVICE
#  ifndef _MSC_VER
__device__ bool __nv_fp128_isnan(__float128);
__device__ __float128 __nv_fp128_fmax(__float128, __float128);
__device__ __float128 __nv_fp128_fmin(__float128, __float128);
#  endif

// ---- Bridge cuda::std into std ----
namespace cuda
{
namespace std
{
}
} // namespace cuda
namespace std
{
using namespace cuda::std;
}

// ============================================================================
// Phase 3: CUDA toolkit headers
// ============================================================================
// By this point all device-side functions and intrinsics are defined, so
// any transitive CCCL includes from these headers will find them.
#  pragma push_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")

#  define __DEVICE_LAUNCH_PARAMETERS_H__

// Guard out CUDA's declaration-only headers; clang provides its own.
#  define __DEVICE_FUNCTIONS_H__
#  define __MATH_FUNCTIONS_H__
#  define __MATH_FUNCTIONS_HPP__
#  define __COMMON_FUNCTIONS_H__
#  define __DEVICE_FUNCTIONS_DECLS_H__

// ---- CUDA runtime types (cudaError_t, dim3, cudaStream_t, etc.) ----
// (host_defines.h already included in Phase 2)
#  undef __CUDACC__
#  include "cuda.h"
#  include "driver_types.h"
#  include "host_config.h"
#  if !defined(CUDA_VERSION) || CUDA_VERSION < 9000
#    error "Unsupported CUDA version (need >= 9.0)!"
#  endif

// Clang does not have __nvvm_memcpy/__nvvm_memset; emulate with builtins.
#  define __nvvm_memcpy(s, d, n, a) __builtin_memcpy(s, d, n)
#  define __nvvm_memset(d, c, n, a) __builtin_memset(d, c, n)

// __THROW may be in a weird state; keep it empty for CUDA includes.
#  undef __THROW
#  define __THROW

// ============================================================================
// Phase 4: Device-side function definitions from CUDA toolkit .hpp files
// ============================================================================
// Poison __host__ to ensure none of these definitions get host attributes.
#  pragma push_macro("__host__")
#  define __host__ UNEXPECTED_HOST_ATTRIBUTE

// Redefine __forceinline__ to include __device__.
#  pragma push_macro("__forceinline__")
#  define __forceinline__ __device__ __inline__ __attribute__((always_inline))

// Math functions: use fast or accurate variants based on compiler flag.
#  pragma push_macro("__USE_FAST_MATH__")
#  if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
#    define __USE_FAST_MATH__ 1
#  endif
#  include "crt/math_functions.hpp"
#  pragma pop_macro("__USE_FAST_MATH__")

#  pragma pop_macro("__forceinline__")

#  undef __MATH_FUNCTIONS_HPP__
#  undef __CUDABE__

// Re-include device functions with __host__ defined as empty to get
// the "other branch" of #if/#else in the .hpp files.
#  define __host__
#  undef __CUDABE__
#  define __CUDACC__

// Atomic function declarations (became builtins in CUDA 9).
#  include "device_atomic_functions.h"
#  undef __DEVICE_FUNCTIONS_HPP__
#  include "crt/device_double_functions.hpp"
#  include "crt/device_functions.hpp"
#  include "device_atomic_functions.hpp"
#  include "sm_20_atomic_functions.hpp"

// sm_20_intrinsics.hpp defines __isGlobal etc. without const attribute.
// Rename them so the definitions from Phase 4 (with const) prevail.
#  pragma push_macro("__isGlobal")
#  pragma push_macro("__isShared")
#  pragma push_macro("__isConstant")
#  pragma push_macro("__isLocal")
#  define __isGlobal   __ignored_cuda___isGlobal
#  define __isShared   __ignored_cuda___isShared
#  define __isConstant __ignored_cuda___isConstant
#  define __isLocal    __ignored_cuda___isLocal
#  include "sm_20_intrinsics.hpp"
#  pragma pop_macro("__isGlobal")
#  pragma pop_macro("__isShared")
#  pragma pop_macro("__isConstant")
#  pragma pop_macro("__isLocal")

#  include "sm_32_atomic_functions.hpp"

#  pragma push_macro("__CUDA_ARCH__")
#  undef __CUDA_ARCH__
#  include "sm_60_atomic_functions.hpp"
#  include "sm_61_intrinsics.hpp"
#  pragma pop_macro("__CUDA_ARCH__")

#  undef __MATH_FUNCTIONS_HPP__

// math_functions.hpp ::signbit conflicts with libstdc++ constexpr ::signbit.
#  pragma push_macro("signbit")
#  pragma push_macro("__GNUC__")
#  undef __GNUC__
#  define signbit __ignored_cuda_signbit
#  pragma push_macro("_GLIBCXX_MATH_H")
#  pragma push_macro("_LIBCPP_VERSION")
#  undef _GLIBCXX_MATH_H
#  ifdef _LIBCPP_VERSION
#    define _LIBCPP_VERSION 3700
#  endif
#  include "crt/math_functions.hpp"
#  pragma pop_macro("_GLIBCXX_MATH_H")
#  pragma pop_macro("_LIBCPP_VERSION")
#  pragma pop_macro("__GNUC__")
#  pragma pop_macro("signbit")

#  pragma pop_macro("__host__")

// ============================================================================
// Phase 5: cuda_runtime.h (first header that transitively pulls in CCCL)
// ============================================================================
// ============================================================================
// Phase 5: cuda_runtime.h (first header that transitively pulls in CCCL)
// ============================================================================
// Verify no libcudacxx header was pulled in yet. If this fires, a header
// above transitively included a system header that resolved to libcudacxx
// before all device-side definitions were ready.
#  ifdef CCCL_VERSION
#    error "libcudacxx was included before device-side definitions were set up"
#  endif

#  pragma push_macro("nv_weak")
#  define nv_weak weak
#  undef __CUDA_LIBDEVICE__
#  define __CUDACC__
#  include "cuda_runtime.h"
#  pragma pop_macro("nv_weak")
#  undef __CUDACC__
#  define __CUDABE__

#  include "crt/host_runtime.h"

// device_runtime.h defines __cxa_* macros that conflict with cxxabi.h.
#  undef __cxa_vec_ctor
#  undef __cxa_vec_cctor
#  undef __cxa_vec_dtor
#  undef __cxa_vec_new
#  undef __cxa_vec_new2
#  undef __cxa_vec_new3
#  undef __cxa_vec_delete2
#  undef __cxa_vec_delete
#  undef __cxa_vec_delete3
#  undef __cxa_pure_virtual

// Texture intrinsics (requires C++11).
#  if __cplusplus >= 201103L
#    include <__clang_cuda_texture_intrinsics.h>
#  else
template <typename T>
struct __nv_tex_needs_cxx11
{
  const static bool value = false;
};
template <class T>
__host__ __device__ void __nv_tex_surf_handler(const char* name, T* ptr, cudaTextureObject_t obj, float x)
{
  _Static_assert(__nv_tex_needs_cxx11<T>::value, "Texture support requires C++11");
}
#  endif
#  include "surface_indirect_functions.h"
#  if CUDA_VERSION < 13000
#    include "texture_fetch_functions.h"
#  endif
#  include "texture_indirect_functions.h"

// ============================================================================
// Phase 7: Restore saved state
// ============================================================================
#  pragma pop_macro("__CUDA_ARCH__")
#  pragma pop_macro("__THROW")
#  undef __CUDABE__
#  define __CUDACC__

// ============================================================================
// Phase 8: Device-side system calls & std wrappers
// ============================================================================
extern "C" {
__device__ int vprintf(const char*, const char*);
__device__ void free(void*) __attribute((nothrow));
__device__ void* malloc(size_t) __attribute((nothrow)) __attribute__((malloc));
__device__ void
__assertfail(const char* __message, const char* __file, unsigned __line, const char* __function, size_t __charSize);
__device__ static inline void
__assert_fail(const char* __message, const char* __file, unsigned __line, const char* __function)
{
  __assertfail(__message, __file, __line, __function, sizeof(char));
}
__device__ int printf(const char*, ...);
} // extern "C"

namespace std
{
__device__ static inline void free(void* __ptr)
{
  ::free(__ptr);
}
__device__ static inline void* malloc(size_t __size)
{
  return ::malloc(__size);
}
} // namespace std

// ============================================================================
// Phase 9: Builtin variable conversion operators
// ============================================================================
// These need dim3 and uint3 to be fully defined (from vector_types.h, pulled
// in by driver_types.h in Phase 5).
__device__ inline __cuda_builtin_threadIdx_t::operator dim3() const
{
  return dim3(x, y, z);
}
__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const
{
  return {x, y, z};
}
__device__ inline __cuda_builtin_blockIdx_t::operator dim3() const
{
  return dim3(x, y, z);
}
__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const
{
  return {x, y, z};
}
__device__ inline __cuda_builtin_blockDim_t::operator dim3() const
{
  return dim3(x, y, z);
}
__device__ inline __cuda_builtin_blockDim_t::operator uint3() const
{
  return {x, y, z};
}
__device__ inline __cuda_builtin_gridDim_t::operator dim3() const
{
  return dim3(x, y, z);
}
__device__ inline __cuda_builtin_gridDim_t::operator uint3() const
{
  return {x, y, z};
}

// ============================================================================
// Phase 10: Remaining clang CUDA headers
// ============================================================================
#  include <__clang_cuda_cmath.h>
#  include <__clang_cuda_complex_builtins.h>
#  include <__clang_cuda_intrinsics.h>

// curand_mtgp32_kernel redefines blockDim/threadIdx with dim3/uint3 types,
// which is incompatible with our builtins. Force-include it with types
// redefined to our builtin types.
// Skip when cuRAND headers are unavailable (e.g. pip-installed toolkit).
#  if __has_include("curand_mtgp32_kernel.h")
#    pragma push_macro("dim3")
#    pragma push_macro("uint3")
#    define dim3  __cuda_builtin_blockDim_t
#    define uint3 __cuda_builtin_threadIdx_t
#    include "curand_mtgp32_kernel.h"
#    pragma pop_macro("dim3")
#    pragma pop_macro("uint3")
#  endif
#  pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")

// Kernel launch configuration function.
#  if CUDA_VERSION >= 9020
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void* stream = 0);
#  endif

// The JIT shared library is linked without the C runtime (no libc on the link
// line) so atexit is unavailable.  The CUDA module constructor calls atexit()
// to register a cleanup function.  Provide a no-op stub — the JIT library is
// short-lived and unloaded explicitly.
#  if !defined(__HOSTJIT_DEVICE_COMPILATION__)
#    if defined(_MSC_VER)
extern "C" int atexit(void(__cdecl*)(void))
{
  return 0;
}
#    else
extern "C" int atexit(void (*)(void))
{
  return 0;
}
#    endif
#  endif

#endif // __CUDA__ && __clang__
#endif // __CLANG_CUDA_RUNTIME_WRAPPER_H__
