//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// Unload test: a JIT module must be safely AND fully released on unload.
//
// What it checks:
//
//  1. Safety (dangling reference): after a fully-registered module is unmapped
//     mid-run, the CUDA runtime must not keep a reference into the freed module
//     image. We launch the JIT'd kernel and verify its result every iteration,
//     and keep issuing CUDA work across iterations (including a sync AFTER each
//     unload). A dangling registration would fault or produce a wrong value on a
//     later cycle.
//
//  2. No leak: the module must actually be unmapped, not retained for the life
//     of the process. After the load/launch/unload loop we count how many JIT
//     module images are still mapped -- expected 0.
//
//  3. Nothing left registered on the other side of the fence. Unmapping the
//     library says nothing about the runtime and the driver: CUDART only queues
//     a module for unloading and issues the driver call later, from the next
//     CUDA entry point, so a module can sit registered after the library is
//     gone. Free device memory is the observable we have from outside, and it is
//     sampled after each cycle (after a sync, which drains the queue); if a
//     registration survived a cycle, the samples drift down as the loop runs.
//     The module carries 16 MiB of device state to make that visible: measured
//     here, free memory is exactly 16 MiB lower while the module is resident and
//     back to the baseline after the unload. Without the ballast the difference
//     stays below what the driver reports and the check proves nothing.
//
//  4. And nothing left registered even for a while. Check #3 samples after a
//     sync, so it cannot see a module that is released only because the runtime
//     was called again. This one reads free memory through the DRIVER API, which
//     does not drain the queue, right after an unload and with no runtime call in
//     between: the device state has to be back by the time unload() returns.
//
//  5. Unloading with GPU work still in flight. The cycles above sync before the
//     unload, so the barrier inside unload() has nothing left to wait for and is
//     never really exercised. Here a kernel that runs for a while is launched and
//     the module is dropped immediately: the work must complete (its result is
//     read afterwards) and the runtime must stay healthy.
//
// (Skipping unload entirely would pass #1 but fail #2; unloading without the
// unregister and the drain would pass #2 but fail #1, so neither alone is enough.)
//
// The module's file name is not hard-coded: we ask the loader for the real path
// of a loaded module (getLoadedModulePath, via the OS) and match on that
// basename, so a rename in the compiler cannot silently make the leak probe a
// false pass. Module enumeration is platform-specific (Windows EnumProcessModules,
// Linux /proc/self/maps); where it is unavailable the leak check is skipped while
// the safety check still runs.

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
// clang-format off
// psapi.h builds on declarations from windows.h, so the order here is fixed.
#  include <windows.h>
#  include <psapi.h>
// clang-format on
#else
#  include <dlfcn.h>
#  if defined(__linux__)
#    include <set>
#  endif
#endif

static const char* k_source = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

// 16 MiB of device state, so that a module left registered costs device memory
// that cudaMemGetInfo can actually see. With a small module the difference is
// below the granularity the driver reports, and the check below would pass no
// matter what.
__device__ int ballast[4 * 1024 * 1024];

__global__ void device_kernel(int* ptr, int v)
{
  ballast[0] = v;
  *ptr       = ballast[0];
}

extern "C" _CCCL_VISIBILITY_EXPORT void host_entry(int* ptr, int v)
{
  device_kernel<<<1, 1>>>(ptr, v);
}
)";

// A kernel that is still running when the module is dropped. It spins on the
// device clock rather than on a trip count, so the duration does not depend on
// how fast the GPU is.
static const char* k_slow_source = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void slow_kernel(int* ptr, int v, long long cycles)
{
  const long long start = clock64();
  while (clock64() - start < cycles)
  {
  }
  *ptr = v;
}

extern "C" _CCCL_VISIBILITY_EXPORT void host_entry(int* ptr, int v, long long cycles)
{
  slow_kernel<<<1, 1>>>(ptr, v, cycles);
}
)";

// Directory-strip a path and drop the " (deleted)" suffix that /proc/self/maps
// appends for a mapping whose backing file was unlinked. Handles both separators.
static std::string basename_of(std::string p)
{
  const auto d = p.find(" (deleted)");
  if (d != std::string::npos)
  {
    p = p.substr(0, d);
  }
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

// Count distinct currently-mapped module images whose basename == target.
// Returns -1 if enumeration is unsupported / failed on this platform.
#if defined(_WIN32)
static int count_mapped(const std::string& target)
{
  HMODULE modules[8192];
  DWORD needed = 0;
  if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed))
  {
    return -1;
  }
  int n = static_cast<int>(needed / sizeof(HMODULE));
  if (n > 8192)
  {
    n = 8192;
  }
  int count = 0;
  for (int i = 0; i < n; ++i)
  {
    char path[MAX_PATH] = {};
    if (K32GetModuleFileNameExA(GetCurrentProcess(), modules[i], path, MAX_PATH))
    {
      if (_stricmp(basename_of(path).c_str(), target.c_str()) == 0)
      {
        ++count;
      }
    }
  }
  return count;
}
#elif defined(__linux__)
static int count_mapped(const std::string& target)
{
  FILE* f = std::fopen("/proc/self/maps", "r");
  if (!f)
  {
    return -1;
  }
  std::set<std::string> paths; // dedupe: one file spans several mapping lines
  char line[8192];
  while (std::fgets(line, sizeof(line), f))
  {
    // Line: "addr perms offset dev inode pathname". Skip the 5 fixed fields
    // (none contain spaces) and take the rest as the pathname.
    int pos = -1;
    std::sscanf(line, "%*s %*s %*s %*s %*s %n", &pos);
    if (pos < 0)
    {
      continue;
    }
    std::string path = line + pos;
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r'))
    {
      path.pop_back();
    }
    if (path.empty() || path[0] != '/')
    {
      continue; // anonymous / [heap] / [stack] etc.
    }
    const auto del      = path.find(" (deleted)");
    const std::string s = (del == std::string::npos) ? path : path.substr(0, del);
    if (basename_of(s) == target)
    {
      paths.insert(s);
    }
  }
  std::fclose(f);
  return static_cast<int>(paths.size());
}
#else
static int count_mapped(const std::string&)
{
  return -1;
}
#endif

// Free device memory as the driver sees it. Unlike cudaMemGetInfo, a driver call
// does not run CUDART's pending-unload queue, so it shows the state as it stands
// right after the unload rather than the state a runtime call would create.
using CuMemGetInfoFn = int (*)(size_t*, size_t*);

static CuMemGetInfoFn load_driver_mem_get_info()
{
#if defined(_WIN32)
  HMODULE libcuda = LoadLibraryA("nvcuda.dll");
  if (!libcuda)
  {
    return nullptr;
  }
  return reinterpret_cast<CuMemGetInfoFn>(GetProcAddress(libcuda, "cuMemGetInfo_v2"));
#else
  void* libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!libcuda)
  {
    return nullptr;
  }
  return reinterpret_cast<CuMemGetInfoFn>(dlsym(libcuda, "cuMemGetInfo_v2"));
#endif
}

// Check (4): load, launch, unload, and read the driver's view of free memory
// without touching the runtime in between.
static bool no_unload_window(hostjit::CompilerConfig config, int* d_ptr)
{
  CuMemGetInfoFn cu_mem_get_info = load_driver_mem_get_info();
  if (!cu_mem_get_info)
  {
    std::printf("unload: window check skipped (driver API unavailable)\n");
    return true;
  }

  size_t total = 0, before = 0, resident = 0, after_unload = 0, after_runtime_call = 0;
  // Before this load, so that the readings afterwards are compared against a level
  // the module was not part of. Comparing them only against each other cannot tell
  // a clean release from a module that stays resident for good, since both leave
  // them equal.
  if (cu_mem_get_info(&before, &total) != 0)
  {
    std::fprintf(stderr, "unload: window check: cuMemGetInfo failed before the load\n");
    return false;
  }

  {
    config.enable_pch = false;
    hostjit::JITCompiler compiler(config);
    if (!compiler.compile(k_source))
    {
      std::fprintf(stderr, "unload: window check compile failed:\n%s\n", compiler.getLastError().c_str());
      return false;
    }
    auto host_fn = compiler.getFunction<void (*)(int*, int)>("host_entry");
    if (!host_fn)
    {
      std::fprintf(stderr, "unload: window check: 'host_entry' not found\n");
      return false;
    }
    host_fn(d_ptr, 7);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
      std::fprintf(stderr, "unload: window check launch failed\n");
      return false;
    }
    if (cu_mem_get_info(&resident, &total) != 0)
    {
      std::fprintf(stderr, "unload: window check: cuMemGetInfo failed\n");
      return false;
    }
    // `compiler` goes out of scope here -> unload().
  }
  if (cu_mem_get_info(&after_unload, &total) != 0)
  {
    std::fprintf(stderr, "unload: window check: cuMemGetInfo failed after the unload\n");
    return false;
  }

  size_t runtime_free = 0, runtime_total = 0;
  // A runtime entry point: this is what drains the queue.
  if (cudaMemGetInfo(&runtime_free, &runtime_total) != cudaSuccess || cu_mem_get_info(&after_runtime_call, &total) != 0)
  {
    std::fprintf(stderr, "unload: window check: memory query failed after the runtime call\n");
    return false;
  }

  const long long held = static_cast<long long>(after_runtime_call) - static_cast<long long>(after_unload);
  const long long lost = static_cast<long long>(before) - static_cast<long long>(after_runtime_call);
  std::printf("unload: free memory before %zu, resident %zu, after unload %zu, after a runtime call %zu\n",
              before,
              resident,
              after_unload,
              after_runtime_call);

  // The module carries 16 MiB, so anything left registered is far above this.
  constexpr size_t kSlackBytes = 1u << 20;
  if (lost > static_cast<long long>(kSlackBytes))
  {
    std::fprintf(stderr, "unload: %lld byte(s) never came back -- that is a leak, not a window\n", lost);
    return false;
  }
  if (held > 0)
  {
    std::fprintf(stderr, "unload: %lld byte(s) held until the next runtime call -- the unload does not flush\n", held);
    return false;
  }
  std::printf("unload: no window -- the device state is back when unload() returns\n");
  return true;
}

// Check (5): drop the module while its kernel is still running on the device.
// unload() waits for it before unmapping, so the work completes and its result is
// there to read afterwards. Without that wait the module's image would go away
// under a running kernel and under a registration the runtime is still following.
static bool survives_unload_with_work_in_flight(hostjit::CompilerConfig config, int* d_ptr)
{
  // Around 100 ms on a GPU clocked at 1 GHz or above, and no shorter on a slower
  // one: long enough that the launch is still in flight when the scope ends.
  constexpr long long kSpinCycles = 100'000'000;
  constexpr int kValue            = 4242;

  if (cudaMemset(d_ptr, 0, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "unload: in-flight check: cudaMemset failed\n");
    return false;
  }

  {
    config.enable_pch = false;
    hostjit::JITCompiler compiler(config);
    if (!compiler.compile(k_slow_source))
    {
      std::fprintf(stderr, "unload: in-flight check compile failed:\n%s\n", compiler.getLastError().c_str());
      return false;
    }
    auto host_fn = compiler.getFunction<void (*)(int*, int, long long)>("host_entry");
    if (!host_fn)
    {
      std::fprintf(stderr, "unload: in-flight check: 'host_entry' not found\n");
      return false;
    }

    host_fn(d_ptr, kValue, kSpinCycles);
    // Deliberately no synchronization here. If the kernel did finish this fast the
    // check still passes, but it stops proving anything, so say so.
    if (cudaStreamQuery(nullptr) != cudaErrorNotReady)
    {
      std::printf("unload: in-flight check: the kernel already finished, the unload had nothing to wait for\n");
    }
    // `compiler` goes out of scope here -> unload(), with the kernel running.
  }

  int result    = -1;
  cudaError_t e = cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess || result != kValue)
  {
    std::fprintf(stderr,
                 "unload: unloading with work in flight lost the work: got %d, want %d, %s\n",
                 result,
                 kValue,
                 cudaGetErrorString(e));
    return false;
  }
  std::printf("unload: in-flight work survived the unload\n");
  return true;
}

int main()
{
  auto config = hostjit::detectDefaultConfig();

  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "unload: cudaMalloc failed\n");
    return 2;
  }

  constexpr int kIters = 16;
  int rc               = 0;
  std::string modname; // JIT module basename, learned at runtime on the first iter

  // Free memory before any module has been loaded. cudaMalloc above has created
  // the context, so this is a settled reading, and it is what the drift check
  // below is measured against: a module that stays resident for good, rather than
  // one that leaks a bit per cycle, keeps the samples flat and would otherwise
  // pass.
  size_t baseline = 0, baseline_total = 0;
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&baseline, &baseline_total) != cudaSuccess)
  {
    std::fprintf(stderr, "unload: could not read free device memory before the first load\n");
    cudaFree(d_ptr);
    return 2;
  }

  // Free device memory after each cycle. The first cycles are not comparable
  // (context and module caches settle), so the comparison starts at kSettle.
  constexpr int kSettle = 4;
  size_t free_after[kIters]{};

  // The numbers below name the checks listed at the top of the file. They do not
  // run in that order: (2) and (3) read what the finished loop left behind, and
  // the probes in between need d_ptr, which is freed after them.
  //
  // (1) Safety: launch + verify each cycle, and issue CUDA work after each unload.
  for (int i = 0; i < kIters; ++i)
  {
    const int expected = 1000 + i;
    {
      hostjit::JITCompiler compiler(config);
      if (!compiler.compile(k_source))
      {
        std::fprintf(stderr, "unload: compile failed on iter %d:\n%s\n", i, compiler.getLastError().c_str());
        rc = 2;
        break;
      }
      if (modname.empty())
      {
        modname = basename_of(compiler.getLoadedModulePath());
      }

      auto host_fn = compiler.getFunction<void (*)(int*, int)>("host_entry");
      if (!host_fn)
      {
        std::fprintf(stderr, "unload: 'host_entry' not found on iter %d\n", i);
        rc = 2;
        break;
      }

      host_fn(d_ptr, expected);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess)
      {
        std::fprintf(stderr, "unload: launch/sync error on iter %d: %s\n", i, cudaGetErrorString(e));
        rc = 1;
        break;
      }

      int result = -1;
      cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
      if (result != expected)
      {
        std::fprintf(stderr, "unload: WRONG result on iter %d: got %d, want %d\n", i, result, expected);
        rc = 1;
        break;
      }
      // `compiler` goes out of scope here -> unload() -> unregister fatbin + unmap.
    }

    // CUDA work AFTER the module was unmapped: a dangling reference would fault here.
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e2 != cudaSuccess)
    {
      std::fprintf(stderr, "unload: post-unload sync error on iter %d: %s\n", i, cudaGetErrorString(e2));
      rc = 1;
      break;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    {
      std::fprintf(stderr, "unload: cudaMemGetInfo failed on iter %d\n", i);
      rc = 2;
      break;
    }
    free_after[i] = free_bytes;
    std::printf("unload: iter %d ok (result=%d) after unload\n", i, expected);
  }

  // (4) No window: an unload must give the device state back on its own, without
  // waiting for the next runtime call.
  if (rc == 0 && !no_unload_window(config, d_ptr))
  {
    rc = 1;
  }

  // (5) An unload while the module's kernel is still running.
  if (rc == 0 && !survives_unload_with_work_in_flight(config, d_ptr))
  {
    rc = 1;
  }

  cudaFree(d_ptr);

  // (2) No leak: after the loop no JIT module image should remain mapped.
  if (rc == 0)
  {
    const int leaked = count_mapped(modname);
    if (leaked < 0)
    {
      std::printf("unload: leak probe skipped (module enumeration unsupported here)\n");
    }
    else
    {
      std::printf("unload: %d '%s' module(s) still mapped after %d cycles\n", leaked, modname.c_str(), kIters);
      if (leaked > 0)
      {
        rc = 1;
      }
    }
  }

  // (3) Nothing left registered in the runtime/driver: free device memory must
  // not drift down across cycles once it has settled, and it must come back to
  // the level it had before any module was loaded.
  if (rc == 0 && free_after[kSettle] != 0 && free_after[kIters - 1] != 0)
  {
    const size_t settled = free_after[kSettle];
    const size_t final   = free_after[kIters - 1];
    // The module carries 16 MiB, so one registration left behind is far above
    // this, while ordinary allocator noise is far below.
    constexpr size_t kSlackBytes = 1u << 20;
    const long long drift        = static_cast<long long>(settled) - static_cast<long long>(final);
    const long long held         = static_cast<long long>(baseline) - static_cast<long long>(final);
    std::printf("unload: free device memory after cycle %d vs %d: %lld byte(s) lower\n", kIters - 1, kSettle, drift);
    std::printf("unload: free device memory %zu before the first load, %zu after the last unload\n", baseline, final);
    if (drift > static_cast<long long>(kSlackBytes))
    {
      std::fprintf(stderr, "unload: device memory keeps dropping -- a module stays registered per cycle\n");
      rc = 1;
    }
    if (held > static_cast<long long>(kSlackBytes))
    {
      std::fprintf(stderr, "unload: %lld byte(s) never came back -- a module stays registered\n", held);
      rc = 1;
    }
  }

  std::printf("unload: %s (%d load/launch/unload cycles)\n", rc == 0 ? "PASS" : "FAIL", kIters);
  std::fflush(stdout);
  std::fflush(stderr);
  // Normal return (not std::_Exit): a CUDA context is live, so let the CRT run
  // cudart's orderly teardown instead of fast-failing at exit.
  return rc;
}
