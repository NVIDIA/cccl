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
// A library produced by libnvcc has to load and unload correctly for a caller
// that has nothing but the OS loader.
//
// The library it produces is freestanding: no C runtime, so no __cxa_atexit and
// no CRT finalizer to run the fatbin unregister that the CUDA registration glue
// schedules. Nothing else in the process is in a position to run it either. A
// caller can compile once, keep the library on disk, and open it much later from
// a process that never linked libnvcc, so whatever unregisters the fatbin has
// to be inside the image.
//
// The test therefore uses the library API and nothing else -- libnvccCreateProgram,
// libnvccCompileProgramToObject, libnvccLinkToSharedLibrary -- and then reaches
// the artifact the way an unrelated application does: dlopen, dlsym, dlclose.
//
// What it checks:
//
//  1. The kernel runs after a plain dlopen, so the fatbin was registered without
//     the caller running the module's constructors by hand.
//
//  2. The device state comes back after a plain dlclose, so the fatbin was
//     unregistered by the image itself. Free memory is compared against the
//     level before the first load, and the module carries 16 MiB of device data
//     to put that difference above what the driver reports. The absolute level
//     is what matters, not a drift across cycles: reloading the same image does
//     not cost another 16 MiB, so a fatbin left registered shows up once and
//     then holds steady.
//
//  3. CUDA still works afterwards. A module unmapped while still registered
//     leaves the runtime holding a pointer into freed memory.
//
//  4. Two holders of one artifact do not interfere: closing one leaves the other
//     one working, since the image is unregistered by the OS reference count
//     rather than by whoever closes first.
//
//  5. When the device state comes back. Checks 2 and 3 sample free memory with a
//     runtime call, which is itself what makes the runtime issue a queued module
//     unload, so they cannot tell "released at close" from "released at the next
//     CUDA call". This one reads the driver's view immediately after the close and
//     reports which of the two happened. A close cannot do the flush itself: it
//     runs under the loader lock, where a blocking call into CUDA is how deadlocks
//     are made. A caller that needs the state back at a known point calls CUDA
//     after closing, which is what the CCCL loader does.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <libnvcc/libnvcc.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  define LIBRARY_SUFFIX ".dll"
#else
#  include <dlfcn.h>
#  include <unistd.h>
#  define LIBRARY_SUFFIX ".so"
#endif

// Plain CUDA: the artifact under test must not need anything from the caller's
// side, and neither should the source it was built from.
static const char* k_source = R"(
#include <cuda_runtime.h>

#if defined(_WIN32)
#  define ENTRY_EXPORT __declspec(dllexport)
#else
#  define ENTRY_EXPORT __attribute__((visibility("default")))
#endif

// 16 MiB of device state, so that a module left registered costs device memory
// the driver actually reports.
__device__ int ballast[4 * 1024 * 1024];

__global__ void device_kernel(int* ptr, int v)
{
  ballast[0] = v;
  *ptr       = ballast[0];
}

extern "C" ENTRY_EXPORT void host_entry(int* ptr, int v)
{
  device_kernel<<<1, 1>>>(ptr, v);
}
)";

static void report(const char* step, libnvccProgram prog, libnvccResult result)
{
  std::fprintf(stderr, "artifact-unload: %s failed: %s\n", step, libnvccGetErrorString(result));
  size_t log_size = 0;
  if (prog && libnvccGetProgramLogSize(prog, &log_size) == LIBNVCC_SUCCESS && log_size > 1)
  {
    std::string log(log_size, '\0');
    if (libnvccGetProgramLog(prog, log.data()) == LIBNVCC_SUCCESS)
    {
      std::fprintf(stderr, "%s\n", log.c_str());
    }
  }
}

// Compile and link through the library API alone. The library carries its own
// defaults for the toolkit and header paths, so the architecture is the only
// thing the caller has to say.
static bool build_library(const std::filesystem::path& dir, std::string& library_path)
{
  int sm     = 75;
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess)
  {
    sm = prop.major * 10 + prop.minor;
  }

  const std::string arch    = "--gpu-architecture=sm_" + std::to_string(sm);
  const char* options[]     = {arch.c_str(), "-O2"};
  const int num_options     = static_cast<int>(sizeof(options) / sizeof(options[0]));
  const std::string object  = (dir / "artifact.o").string();
  const std::string library = (dir / ("artifact" LIBRARY_SUFFIX)).string();

  libnvccProgram prog = nullptr;
  libnvccResult r     = libnvccCreateProgram(&prog, k_source, "artifact.cu");
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCreateProgram", prog, r);
    return false;
  }

  r = libnvccCompileProgramToObject(prog, object.c_str(), nullptr, num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCompileProgramToObject", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }

  const char* objects[] = {object.c_str()};
  r                     = libnvccLinkToSharedLibrary(prog, 1, objects, library.c_str(), num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccLinkToSharedLibrary", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }

  libnvccDestroyProgram(&prog);
  library_path = library;
  return true;
}

// One cycle of what an application that never linked libnvcc does with it.
static bool run_cycle(const std::string& path, int* d_ptr, int expected)
{
#if defined(_WIN32)
  HMODULE handle = LoadLibraryA(path.c_str());
#else
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!handle)
  {
#if defined(_WIN32)
    std::fprintf(stderr, "artifact-unload: LoadLibrary failed (error %lu)\n", GetLastError());
#else
    std::fprintf(stderr, "artifact-unload: dlopen failed: %s\n", dlerror());
#endif
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
#if defined(_WIN32)
  auto host_fn = reinterpret_cast<host_entry_fn>(reinterpret_cast<void*>(GetProcAddress(handle, "host_entry")));
#else
  auto host_fn = reinterpret_cast<host_entry_fn>(dlsym(handle, "host_entry"));
#endif
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: 'host_entry' not found\n");
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
    return false;
  }

  host_fn(d_ptr, expected);
  cudaError_t e = cudaDeviceSynchronize();
  int result    = -1;
  cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess || result != expected)
  {
    std::fprintf(stderr, "artifact-unload: got %d, want %d, %s\n", result, expected, cudaGetErrorString(e));
    return false;
  }

#if defined(_WIN32)
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif
  return true;
}

// Two holders of one artifact. Opening the same file twice returns the same
// mapping with a bumped reference count and does not re-run the constructors, so
// the fatbin is registered once, and the OS runs the finalizer only when the last
// handle goes. Closing one holder must therefore leave the other one working.
static bool survives_a_second_holder(const std::string& path, int* d_ptr)
{
#if defined(_WIN32)
  HMODULE first  = LoadLibraryA(path.c_str());
  HMODULE second = LoadLibraryA(path.c_str());
#else
  void* first  = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  void* second = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!first || !second)
  {
    std::fprintf(stderr, "artifact-unload: could not open the artifact twice\n");
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
#if defined(_WIN32)
  auto host_fn = reinterpret_cast<host_entry_fn>(reinterpret_cast<void*>(GetProcAddress(second, "host_entry")));
#else
  auto host_fn = reinterpret_cast<host_entry_fn>(dlsym(second, "host_entry"));
#endif
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: 'host_entry' not found through the second handle\n");
    return false;
  }

#if defined(_WIN32)
  FreeLibrary(first);
#else
  dlclose(first);
#endif

  // Through the handle that is still open, after the other one is gone.
  int result = -1;
  host_fn(d_ptr, 7);
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);

#if defined(_WIN32)
  FreeLibrary(second);
#else
  dlclose(second);
#endif

  if (e != cudaSuccess || result != 7)
  {
    std::fprintf(stderr,
                 "artifact-unload: closing one holder broke the other: got %d, want 7, %s\n",
                 result,
                 cudaGetErrorString(e));
    return false;
  }
  return true;
}

// Free device memory as the driver sees it. A driver call does not run the
// runtime's pending-unload queue, so it shows the state as it stands right after
// the close rather than the state a runtime call would create.
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

// Check (5): when the device state comes back after a plain close. Fails only if
// it never does; whether it lands at the close or at the next CUDA call is
// reported, because that is a property of the runtime rather than of the image.
static bool report_close_window(const std::string& path, int* d_ptr)
{
  CuMemGetInfoFn cu_mem_get_info = load_driver_mem_get_info();
  if (!cu_mem_get_info)
  {
    std::printf("artifact-unload: close-window report skipped (driver API unavailable)\n");
    return true;
  }

  size_t total = 0, before = 0, resident = 0, after_close = 0, after_cuda_call = 0;
  // Before this load: the readings afterwards need a level the module was not part
  // of, since comparing them only against each other leaves a module that stays
  // resident for good indistinguishable from a clean release.
  if (cu_mem_get_info(&before, &total) != 0)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: cuMemGetInfo failed before the load\n");
    return false;
  }

#if defined(_WIN32)
  HMODULE handle = LoadLibraryA(path.c_str());
#else
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!handle)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: could not open the artifact\n");
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
#if defined(_WIN32)
  auto host_fn = reinterpret_cast<host_entry_fn>(reinterpret_cast<void*>(GetProcAddress(handle, "host_entry")));
#else
  auto host_fn = reinterpret_cast<host_entry_fn>(dlsym(handle, "host_entry"));
#endif
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: 'host_entry' not found\n");
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
    return false;
  }

  host_fn(d_ptr, 11);
  if (cudaDeviceSynchronize() != cudaSuccess || cu_mem_get_info(&resident, &total) != 0)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: launch or memory query failed\n");
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
    return false;
  }

#if defined(_WIN32)
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif

  // No CUDA call between the close and this reading.
  size_t runtime_free = 0, runtime_total = 0;
  if (cu_mem_get_info(&after_close, &total) != 0 || cudaMemGetInfo(&runtime_free, &runtime_total) != cudaSuccess
      || cu_mem_get_info(&after_cuda_call, &total) != 0)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: memory query failed\n");
    return false;
  }

  const long long held = static_cast<long long>(after_cuda_call) - static_cast<long long>(after_close);
  const long long lost = static_cast<long long>(before) - static_cast<long long>(after_cuda_call);
  std::printf("artifact-unload: free memory before %zu, resident %zu, after the close %zu, after a CUDA call %zu\n",
              before,
              resident,
              after_close,
              after_cuda_call);
  // The module carries 16 MiB, so anything left registered is far above this.
  constexpr size_t kSlackBytes = 1u << 20;
  if (lost > static_cast<long long>(kSlackBytes))
  {
    std::fprintf(stderr, "artifact-unload: %lld byte(s) never came back after the close\n", lost);
    return false;
  }
  std::printf("artifact-unload: %s\n",
              held > 0 ? "the device state comes back on the first CUDA call after the close, not at the close itself"
                       : "the device state is back at the close");
  return true;
}

static unsigned long process_id()
{
#if defined(_WIN32)
  return GetCurrentProcessId();
#else
  return static_cast<unsigned long>(getpid());
#endif
}

int main()
{
  std::error_code ec;
  const std::filesystem::path dir =
    std::filesystem::temp_directory_path(ec) / ("libnvcc_artifact_" + std::to_string(process_id()));
  std::filesystem::create_directories(dir, ec);

  std::string path;
  if (!build_library(dir, path))
  {
    std::filesystem::remove_all(dir, ec);
    return 2;
  }
  std::printf("artifact-unload: using %s\n", path.c_str());

  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "artifact-unload: cudaMalloc failed\n");
    std::filesystem::remove_all(dir, ec);
    return 2;
  }

  // Free memory before the image has ever been loaded. cudaMalloc has already
  // created the context, so this is a settled reading.
  size_t baseline = 0, total = 0, free_bytes = 0;
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&baseline, &total) != cudaSuccess)
  {
    std::fprintf(stderr, "artifact-unload: could not read free device memory before the first load\n");
    cudaFree(d_ptr);
    std::filesystem::remove_all(dir, ec);
    return 2;
  }

  constexpr int kIters = 8;
  int rc               = 0;
  for (int i = 0; i < kIters; ++i)
  {
    if (!run_cycle(path, d_ptr, 1000 + i))
    {
      std::fprintf(stderr, "artifact-unload: cycle %d failed\n", i);
      rc = 1;
      break;
    }

    // CUDA work after the image is gone: a registration left behind would have
    // the runtime follow a pointer into it.
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess)
    {
      std::fprintf(stderr, "artifact-unload: post-unload sync failed on cycle %d: %s\n", i, cudaGetErrorString(e));
      rc = 1;
      break;
    }
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess)
    {
      std::fprintf(stderr, "artifact-unload: cudaMemGetInfo failed on cycle %d\n", i);
      rc = 2;
      break;
    }
    std::printf("artifact-unload: cycle %d ok, %zu byte(s) free\n", i, free_bytes);
  }

  if (rc == 0)
  {
    rc = survives_a_second_holder(path, d_ptr) ? 0 : 1;
    if (rc == 0)
    {
      if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&free_bytes, &total) != cudaSuccess)
      {
        std::fprintf(stderr, "artifact-unload: memory query failed after the two-holder check\n");
        rc = 2;
      }
      else
      {
        std::printf("artifact-unload: two holders ok, %zu byte(s) free\n", free_bytes);
      }
    }
  }

  if (rc == 0 && !report_close_window(path, d_ptr))
  {
    rc = 1;
  }

  cudaFree(d_ptr);

  if (rc == 0)
  {
    const long long held = static_cast<long long>(baseline) - static_cast<long long>(free_bytes);
    std::printf("artifact-unload: free device memory %zu before the first load, %zu after the last unload\n",
                baseline,
                free_bytes);
    // The module holds 16 MiB while registered; allocator noise is orders of
    // magnitude below the slack.
    constexpr size_t kSlackBytes = 1u << 20;
    if (held > static_cast<long long>(kSlackBytes))
    {
      std::fprintf(
        stderr, "artifact-unload: %lld byte(s) never came back -- a plain close leaves the fatbin registered\n", held);
      rc = 1;
    }
  }

  std::filesystem::remove_all(dir, ec);
  std::printf("artifact-unload: %s (%d load/unload cycles)\n", rc == 0 ? "PASS" : "FAIL", kIters);
  std::fflush(stdout);
  std::fflush(stderr);
  return rc;
}
