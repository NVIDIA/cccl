//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

// Minimal, self-contained, dependency-free implementation of the Itanium
// C++ ABI's thread-safe static-local-variable init guard functions
// (__cxa_guard_acquire/__cxa_guard_release/__cxa_guard_abort), force-included
// into every HostJIT host-side compile (see compileHostCode in compiler.cpp).
//
// Generated kernels can contain function-local statics with non-trivial
// initializers (e.g. CUB's device_count_cached_value() and
// logging_enabled() caches), which the compiler protects with calls to
// these functions. The generated .so is linked with --allow-shlib-undefined,
// and historically left these unresolved, relying on the host process
// happening to already have libstdc++ loaded with global symbol *scope*
// visibility -- an assumption that doesn't reliably hold (e.g. libstdc++
// commonly first enters the process as a transitive RTLD_LOCAL dependency
// of some other dlopen(), such as Python's own C++ extension modules,
// which keeps its symbols out of the scope our RTLD_LOCAL-loaded generated
// .so searches). Defining them here, force-included into every host
// compile, removes the dependency on the host environment entirely instead
// of relying on it being satisfiable -- and unlike an approach that
// disables the guard at the source level (-fno-threadsafe-statics), it
// doesn't require auditing every call site that might reach a guarded
// static to add external serialization.
//
// Guard object layout matches the real Itanium ABI / libc++abi (see
// llvm-project's libcxxabi/src/cxa_guard_impl.h): byte 0 (the "guard byte")
// is read directly by compiler-generated code as a fast path and must only
// be set, via a release store, once initialization is fully complete. The
// remaining bytes are implementation-defined bookkeeping; byte 1 is used
// here as a spin-lock bit, scoped per guard variable (not a single global
// lock), so initializing two different statics on the same thread -- one
// nested inside the other's initializer -- cannot deadlock. Only the 64-bit
// (non-ARM) guard layout is implemented -- HostJIT never targets ARM.
#ifndef _HOSTJIT_CXA_GUARD_H
#define _HOSTJIT_CXA_GUARD_H

extern "C" {

__attribute__((visibility("hidden"))) inline int __cxa_guard_acquire(unsigned long long* raw_guard)
{
  auto* bytes = reinterpret_cast<unsigned char*>(raw_guard);
  if (__atomic_load_n(&bytes[0], __ATOMIC_ACQUIRE) != 0)
  {
    return 0; // already initialized
  }
  for (;;)
  {
    unsigned char expected = 0;
    if (__atomic_compare_exchange_n(
          &bytes[1], &expected, static_cast<unsigned char>(1), false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE))
    {
      break; // we hold the lock bit
    }
    if (__atomic_load_n(&bytes[0], __ATOMIC_ACQUIRE) != 0)
    {
      return 0; // someone else finished while we were trying to acquire
    }
  }
  // Re-check after winning the lock: another thread may have completed
  // initialization and released the lock between our load above and the CAS.
  if (__atomic_load_n(&bytes[0], __ATOMIC_ACQUIRE) != 0)
  {
    __atomic_store_n(&bytes[1], static_cast<unsigned char>(0), __ATOMIC_RELEASE);
    return 0;
  }
  return 1; // caller should perform initialization, then call release/abort
}

__attribute__((visibility("hidden"))) inline void __cxa_guard_release(unsigned long long* raw_guard)
{
  auto* bytes = reinterpret_cast<unsigned char*>(raw_guard);
  __atomic_store_n(&bytes[0], static_cast<unsigned char>(1), __ATOMIC_RELEASE);
  __atomic_store_n(&bytes[1], static_cast<unsigned char>(0), __ATOMIC_RELEASE);
}

__attribute__((visibility("hidden"))) inline void __cxa_guard_abort(unsigned long long* raw_guard)
{
  auto* bytes = reinterpret_cast<unsigned char*>(raw_guard);
  __atomic_store_n(&bytes[1], static_cast<unsigned char>(0), __ATOMIC_RELEASE);
}

} // extern "C"

#endif // _HOSTJIT_CXA_GUARD_H
