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
// A library produced by libnvcc has to load and unload correctly for a caller that has
// nothing but the OS loader.
//
// That library is freestanding: no C runtime, so no __cxa_atexit and no CRT finalizer to run
// the fatbin unregister the CUDA registration glue schedules, and nothing else in the process
// is in a position to run it. A caller can compile once, keep the library on disk and open it
// much later from a process that never linked libnvcc, so whatever unregisters the fatbin has
// to be inside the image. The test therefore uses the library API and nothing else, then
// reaches the artifact the way an unrelated application does: dlopen, dlsym, dlclose.
//
// What it checks:
//
//  1. The kernel runs after a plain dlopen, so the fatbin was registered without the caller
//     running the module's constructors by hand.
//
//  2. The device state comes back after a plain dlclose, so the image unregistered the fatbin
//     itself. Free memory is compared against the level before the first load, with 16 MiB of
//     device data in the module to put the difference above what the driver reports. The
//     absolute level is what matters rather than a drift across cycles: reloading the same
//     image costs no further 16 MiB, so a fatbin left registered shows once and holds steady.
//
//  3. CUDA still works afterwards. A module unmapped while registered leaves the runtime
//     holding a pointer into freed memory.
//
//  4. Two holders of one artifact do not interfere: closing one leaves the other working, the
//     image being unregistered on the OS reference count rather than by whoever closes first.
//
//  5. When the device state comes back. Checks 2 and 3 sample free memory with a runtime
//     call, which is itself what makes the runtime issue a queued module unload, so they
//     cannot tell "released at the close" from "released at the next CUDA call". This one
//     reads the driver's view immediately after the close and reports which happened. The
//     close cannot flush by itself: it runs under the loader lock, where a blocking call into
//     CUDA is how deadlocks are made. A caller needing the state back at a known point calls
//     CUDA after closing, which is what the CCCL loader does.
//
// What this path cannot do, and the test therefore synchronizes before every close: wait for
// the module's GPU work. The image's hook runs under the loader lock, so it unregisters the
// fatbin and returns without a barrier, leaving a kernel still running at that point without
// its module. A caller closing an artifact by hand has to synchronize first; the loader in
// src/hostjit does that in unload().

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
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

// Tolerance for the free-memory checks. The module carries 16 MiB, so anything left
// registered is far above this, while ordinary allocator noise is far below.
constexpr size_t kSlackBytes = 1u << 20;

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

// The artifact, opened the way an application would. Closing on the way out is what
// makes it safe for a check to fail early: a failure that left the image mapped and
// its fatbin registered would show up again in the memory checks in main, on top of
// the real one.
class Artifact
{
public:
  explicit Artifact(const std::string& path)
  {
#if defined(_WIN32)
    handle_ = static_cast<void*>(LoadLibraryA(path.c_str()));
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  }

  ~Artifact()
  {
    close();
  }

  Artifact(const Artifact&)            = delete;
  Artifact& operator=(const Artifact&) = delete;

  explicit operator bool() const
  {
    return handle_ != nullptr;
  }

  void* symbol(const char* name) const
  {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    return dlsym(handle_, name);
#endif
  }

  void close()
  {
    if (handle_)
    {
#if defined(_WIN32)
      FreeLibrary(static_cast<HMODULE>(handle_));
#else
      dlclose(handle_);
#endif
      handle_ = nullptr;
    }
  }

  static std::string openError()
  {
#if defined(_WIN32)
    return "LoadLibrary error " + std::to_string(GetLastError());
#else
    const char* e = dlerror();
    return e ? std::string(e) : std::string("unknown dlopen error");
#endif
  }

private:
  void* handle_ = nullptr;
};

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

// The library carries its own defaults for the toolkit and header paths, so the
// architecture is the only thing the caller has to say.
static std::string arch_option()
{
  int sm     = 75;
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess)
  {
    sm = prop.major * 10 + prop.minor;
  }
  return "--gpu-architecture=sm_" + std::to_string(sm);
}

// Compile and link through the library API alone.
static bool build_library(const std::filesystem::path& dir, std::string& library_path)
{
  const std::string arch    = arch_option();
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

static bool read_whole_file(const std::filesystem::path& path, std::string& content)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    return false;
  }
  content.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

// The teardown runtime object is written into the directory the caller asked for the library
// in, and everything else in there is the caller's. The link picks the name by creating it,
// so no test can force a collision; what is checked is the property that follows -- a file
// the caller already has under a name of that shape keeps its content, and the link leaves no
// object of its own behind. The names planted first are the fixed one this object used to be
// written under and one inside the pattern it is drawn from now, so a return to a fixed name
// is caught whichever name it is.
//
// Only this one intermediate. The Windows link writes more files into the same directory
// under fixed names, a defect older than this change with a fix of its own, and this check
// widens to the whole directory along with it.
static bool leaves_the_callers_files_alone(const std::filesystem::path& dir)
{
  const std::string arch    = arch_option();
  const char* options[]     = {arch.c_str(), "-O2"};
  const int num_options     = static_cast<int>(sizeof(options) / sizeof(options[0]));
  const std::string object  = (dir / "occupied.o").string();
  const std::string library = (dir / ("occupied" LIBRARY_SUFFIX)).string();

#ifdef _WIN32
  const char* const object_suffix = ".obj";
#else
  const char* const object_suffix = ".o";
#endif
  const std::vector<std::filesystem::path> planted = {
    library + ".hostjit_runtime" + object_suffix,
    library + ".hostjit_runtime-abcdef" + object_suffix,
  };
  for (const auto& path : planted)
  {
    std::ofstream out(path, std::ios::binary);
    out << "not an object file: " << path.filename().string() << "\n";
    if (!out)
    {
      std::fprintf(stderr, "artifact-unload: could not write %s\n", path.string().c_str());
      return false;
    }
  }

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
    report("the compile before the link into an occupied directory", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }

  const char* objects[] = {object.c_str()};
  r                     = libnvccLinkToSharedLibrary(prog, 1, objects, library.c_str(), num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("the link into an occupied directory", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }
  libnvccDestroyProgram(&prog);

  for (const auto& path : planted)
  {
    std::string now;
    if (!read_whole_file(path, now))
    {
      std::fprintf(stderr, "artifact-unload: the link removed %s\n", path.string().c_str());
      return false;
    }
    if (now != "not an object file: " + path.filename().string() + "\n")
    {
      std::fprintf(stderr, "artifact-unload: the link wrote over %s\n", path.string().c_str());
      return false;
    }
  }

  // Whatever name the object was written under, it is inside this prefix, so anything left
  // under it that was not planted is the link's.
  const std::string prefix = std::filesystem::path(library).filename().string() + ".hostjit_runtime";
  for (const auto& entry : std::filesystem::directory_iterator(dir))
  {
    const std::string name = entry.path().filename().string();
    if (name.compare(0, prefix.size(), prefix) != 0)
    {
      continue;
    }
    const auto is_planted = [&entry](const std::filesystem::path& path) {
      return path == entry.path();
    };
    if (std::none_of(planted.begin(), planted.end(), is_planted))
    {
      std::fprintf(stderr, "artifact-unload: the link left %s behind\n", entry.path().string().c_str());
      return false;
    }
  }

  std::printf("artifact-unload: %zu file(s) of the caller's survived the link, and the link's own object was not left "
              "behind\n",
              planted.size());
  return true;
}

// Both writers here write atomically the same way: a run of '%' is appended to the output
// path, a file is created under the model that makes, and it is renamed into place. In a model
// every '%' stands for a character the call picks, wherever in the path it sits, so one of the
// caller's in a directory name sends that temporary file somewhere that does not exist. Only
// the link fails on it, clang's output backend opening the final path instead when the
// temporary cannot be created; a '%' in the file name reaches the temporary name only, and the
// rename produces what was asked for. So three cases and one refusal among them: a file name
// with a '%' compiles and links, a directory with one compiles, and only the link into it is
// refused -- up front, rather than inside the linker as a directory that is not there.
static bool handles_a_percent_by_where_it_sits(const std::filesystem::path& dir)
{
  const std::string arch = arch_option();
  const char* options[]  = {arch.c_str(), "-O2"};
  const int num_options  = static_cast<int>(sizeof(options) / sizeof(options[0]));

  libnvccProgram prog = nullptr;
  libnvccResult r     = libnvccCreateProgram(&prog, k_source, "artifact.cu");
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCreateProgram for the percent check", prog, r);
    return false;
  }

  const std::string named_object  = (dir / "100%_of_an_object.o").string();
  const std::string named_library = (dir / ("100%_of_a_library" LIBRARY_SUFFIX)).string();

  r = libnvccCompileProgramToObject(prog, named_object.c_str(), nullptr, num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("a compile to a file name with a percent", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }

  const char* objects[] = {named_object.c_str()};
  r                     = libnvccLinkToSharedLibrary(prog, 1, objects, named_library.c_str(), num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("a link to a file name with a percent", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }
  if (!std::filesystem::exists(named_library))
  {
    std::fprintf(stderr, "artifact-unload: the link reported success but %s is not there\n", named_library.c_str());
    libnvccDestroyProgram(&prog);
    return false;
  }

  const std::filesystem::path odd_dir = dir / "100%_of_a_directory";
  std::error_code ec;
  std::filesystem::create_directories(odd_dir, ec);
  if (ec)
  {
    std::fprintf(stderr, "artifact-unload: could not create %s: %s\n", odd_dir.string().c_str(), ec.message().c_str());
    libnvccDestroyProgram(&prog);
    return false;
  }

  const std::string odd_dir_object = (odd_dir / "artifact.o").string();
  r = libnvccCompileProgramToObject(prog, odd_dir_object.c_str(), nullptr, num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("a compile into a directory with a percent", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }
  if (!std::filesystem::exists(odd_dir_object))
  {
    std::fprintf(stderr, "artifact-unload: the compile reported success but %s is not there\n", odd_dir_object.c_str());
    libnvccDestroyProgram(&prog);
    return false;
  }

  r = libnvccLinkToSharedLibrary(
    prog, 1, objects, (odd_dir / ("artifact" LIBRARY_SUFFIX)).string().c_str(), num_options, options);
  if (r != LIBNVCC_ERROR_INVALID_INPUT)
  {
    std::fprintf(
      stderr, "artifact-unload: a link into a directory with a '%%' was not refused (%s)\n", libnvccGetErrorString(r));
    libnvccDestroyProgram(&prog);
    return false;
  }
  libnvccDestroyProgram(&prog);

  std::printf("artifact-unload: a '%%' in the output's file name is written as asked, and a directory named with one "
              "takes an object but not a library\n");
  return true;
}

// A source that defines atexit itself. The library's teardown depends on the
// registration constructor's atexit() call reaching the runtime libnvcc adds to the
// link, so a definition arriving from the caller's object would take that call and
// leave the library unregistering nothing at unload.
//
// The force-included header renames the name away, which is how a program calling
// atexit is refused while it compiles. That is undone here on purpose: what this
// checks is the object that gets the name from somewhere else, and what stops it is
// the link, not the compilation.
static const char* k_rival_atexit_source = R"(
#include <cuda_runtime.h>

#undef atexit

__global__ void device_kernel(int* ptr, int v)
{
  *ptr = v;
}

#if defined(_MSC_VER)
extern "C" int atexit(void(__cdecl* func)(void))
#else
extern "C" int atexit(void (*func)(void))
#endif
{
  (void) func;
  return 0;
}
)";

// A header a caller might precompile, carrying the two objects whose destructors have to run
// at exit. Neither names atexit, and for the MSVC ABI clang emits the call to it for both.
static const char* k_exit_dtor_header_source = R"(
#include <cuda_runtime.h>

struct closer
{
  int* p;
  closer() : p(nullptr) {}
  ~closer() { p = nullptr; }
};

static closer g_closer;

inline int* local_ptr(void)
{
  static closer s_closer;
  return s_closer.p;
}
)";

// The same header without the destructors, which has to precompile. The refusal below is a
// diagnostic turned into an error over the whole host compilation, so what it costs has to be
// measured next to what it buys.
static const char* k_plain_header_source = R"(
#include <cuda_runtime.h>

struct holder
{
  int* p;
};

static holder g_holder = {nullptr};

inline int* held_ptr(void)
{
  return g_holder.p;
}
)";

// Builds a host precompiled header from each, expecting the first to be refused and the second
// to be produced. A precompiled header is where an exit-time destructor would otherwise pass
// unseen: one call builds the header, another emits the code, and the second does not diagnose
// again what it read from the first. So the refusal has to belong to the call that builds it.
static bool refuses_a_precompiled_header_with_exit_time_destructors(const std::filesystem::path& dir)
{
  const std::string arch = arch_option();
  const char* options[]  = {arch.c_str(), "-O2"};
  const int num_options  = static_cast<int>(sizeof(options) / sizeof(options[0]));

  struct pch_case
  {
    const char* what;
    const char* source;
    const char* stem;
    bool expected;
  };
  const pch_case cases[] = {
    {"a header with exit-time destructors", k_exit_dtor_header_source, "exit_dtor", false},
    {"a header without them", k_plain_header_source, "plain_header", true},
  };

  for (const pch_case& c : cases)
  {
    libnvccProgram prog = nullptr;
    libnvccResult r     = libnvccCreateProgram(&prog, c.source, "header.cu");
    if (r != LIBNVCC_SUCCESS)
    {
      report("libnvccCreateProgram", prog, r);
      return false;
    }

    const std::string pch_source = (dir / (std::string(c.stem) + "_preamble.cu")).string();
    const std::string pch_output = (dir / (std::string(c.stem) + ".pch")).string();
    r = libnvccCreatePCH(prog, LIBNVCC_PCH_HOST, pch_source.c_str(), pch_output.c_str(), num_options, options);
    const bool produced = r == LIBNVCC_SUCCESS;
    if (produced != c.expected)
    {
      if (produced)
      {
        std::fprintf(stderr, "artifact-unload: %s was precompiled\n", c.what);
      }
      else
      {
        report("libnvccCreatePCH", prog, r);
      }
      libnvccDestroyProgram(&prog);
      return false;
    }

    // For the refused one, over the destructor and not over something else the header
    // happened to break on: a build that failed for another reason would pass a check that
    // only read the return value.
    if (!produced)
    {
      const bool right_result = r == LIBNVCC_ERROR_PCH_CREATE;
      std::string log;
      size_t log_size = 0;
      if (libnvccGetProgramLogSize(prog, &log_size) == LIBNVCC_SUCCESS && log_size > 1)
      {
        log.resize(log_size, '\0');
        if (libnvccGetProgramLog(prog, log.data()) != LIBNVCC_SUCCESS)
        {
          log.clear();
        }
      }
      if (!right_result || log.find("exit-time destructor") == std::string::npos)
      {
        std::fprintf(stderr,
                     "artifact-unload: %s was refused, but as %s: %s\n",
                     c.what,
                     libnvccGetErrorString(r),
                     log.empty() ? "(no log)" : log.c_str());
        libnvccDestroyProgram(&prog);
        return false;
      }
    }
    libnvccDestroyProgram(&prog);
  }

  std::printf("artifact-unload: a header with exit-time destructors is refused as it is precompiled, one without "
              "them is not\n");
  return true;
}

// Two links that would leave the produced library unable to tear itself down, and
// have to be refused instead of produced: one with a second object, which brings a
// second fatbin registration for one body of code, and one whose object defines
// atexit. The first is refused at the API, the second by the linker, on a duplicate
// symbol -- which is what the runtime being one strong definition per library buys.
static bool refuses_links_that_break_the_teardown(const std::filesystem::path& dir)
{
  const std::string arch = arch_option();
  const char* options[]  = {arch.c_str(), "-O2"};
  const int num_options  = static_cast<int>(sizeof(options) / sizeof(options[0]));

  libnvccProgram prog = nullptr;
  libnvccResult r     = libnvccCreateProgram(&prog, k_source, "artifact.cu");
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCreateProgram", prog, r);
    return false;
  }

  const std::string object = (dir / "refused.o").string();
  r                        = libnvccCompileProgramToObject(prog, object.c_str(), nullptr, num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCompileProgramToObject", prog, r);
    libnvccDestroyProgram(&prog);
    return false;
  }

  const std::string two_object_library = (dir / ("two_objects" LIBRARY_SUFFIX)).string();
  const char* two_objects[]            = {object.c_str(), object.c_str()};
  r = libnvccLinkToSharedLibrary(prog, 2, two_objects, two_object_library.c_str(), num_options, options);
  libnvccDestroyProgram(&prog);
  if (r == LIBNVCC_SUCCESS)
  {
    std::fprintf(stderr, "artifact-unload: a link of two objects was accepted\n");
    return false;
  }

  libnvccProgram rival = nullptr;
  r                    = libnvccCreateProgram(&rival, k_rival_atexit_source, "rival.cu");
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCreateProgram", rival, r);
    return false;
  }

  const std::string rival_object = (dir / "rival.o").string();
  r = libnvccCompileProgramToObject(rival, rival_object.c_str(), nullptr, num_options, options);
  if (r != LIBNVCC_SUCCESS)
  {
    report("libnvccCompileProgramToObject", rival, r);
    libnvccDestroyProgram(&rival);
    return false;
  }

  const std::string rival_library = (dir / ("rival" LIBRARY_SUFFIX)).string();
  const char* rival_objects[]     = {rival_object.c_str()};
  r = libnvccLinkToSharedLibrary(rival, 1, rival_objects, rival_library.c_str(), num_options, options);
  libnvccDestroyProgram(&rival);
  if (r == LIBNVCC_SUCCESS)
  {
    std::fprintf(stderr, "artifact-unload: an object defining atexit was linked into a library\n");
    return false;
  }

  std::printf("artifact-unload: a second object and an object defining atexit are both refused\n");
  return true;
}

// One cycle of what an application that never linked libnvcc does with it.
static bool run_cycle(const std::string& path, int* d_ptr, int expected)
{
  Artifact artifact(path);
  if (!artifact)
  {
    std::fprintf(stderr, "artifact-unload: could not open the artifact: %s\n", Artifact::openError().c_str());
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
  auto host_fn        = reinterpret_cast<host_entry_fn>(artifact.symbol("host_entry"));
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: 'host_entry' not found\n");
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

  return true;
}

// Two holders of one artifact. Opening the same file twice returns the same
// mapping with a bumped reference count and does not re-run the constructors, so
// the fatbin is registered once, and the OS runs the finalizer only when the last
// handle goes. Closing one holder must therefore leave the other one working.
static bool survives_a_second_holder(const std::string& path, int* d_ptr)
{
  Artifact first(path);
  Artifact second(path);
  if (!first || !second)
  {
    std::fprintf(stderr, "artifact-unload: could not open the artifact twice: %s\n", Artifact::openError().c_str());
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
  auto host_fn        = reinterpret_cast<host_entry_fn>(second.symbol("host_entry"));
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: 'host_entry' not found through the second handle\n");
    return false;
  }

  first.close();

  // Through the handle that is still open, after the other one is gone.
  int result = -1;
  host_fn(d_ptr, 7);
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);

  second.close();

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

  Artifact artifact(path);
  if (!artifact)
  {
    std::fprintf(
      stderr, "artifact-unload: close-window report: could not open the artifact: %s\n", Artifact::openError().c_str());
    return false;
  }

  using host_entry_fn = void (*)(int*, int);
  auto host_fn        = reinterpret_cast<host_entry_fn>(artifact.symbol("host_entry"));
  if (!host_fn)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: 'host_entry' not found\n");
    return false;
  }

  host_fn(d_ptr, 11);
  if (cudaDeviceSynchronize() != cudaSuccess || cu_mem_get_info(&resident, &total) != 0)
  {
    std::fprintf(stderr, "artifact-unload: close-window report: launch or memory query failed\n");
    return false;
  }

  artifact.close();

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

  if (rc == 0 && !refuses_links_that_break_the_teardown(dir))
  {
    rc = 1;
  }

  if (rc == 0 && !refuses_a_precompiled_header_with_exit_time_destructors(dir))
  {
    rc = 1;
  }

  if (rc == 0 && !leaves_the_callers_files_alone(dir))
  {
    rc = 1;
  }

  if (rc == 0 && !handles_a_percent_by_where_it_sits(dir))
  {
    rc = 1;
  }

  cudaFree(d_ptr);

  if (rc == 0)
  {
    // A fresh reading: the checks above loaded and closed the artifact again after
    // the last one taken in the loop.
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&free_bytes, &total) != cudaSuccess)
    {
      std::fprintf(stderr, "artifact-unload: could not read free device memory after the last unload\n");
      rc = 2;
    }
    else
    {
      const long long held = static_cast<long long>(baseline) - static_cast<long long>(free_bytes);
      std::printf("artifact-unload: free device memory %zu before the first load, %zu after the last unload\n",
                  baseline,
                  free_bytes);
      if (held > static_cast<long long>(kSlackBytes))
      {
        std::fprintf(stderr,
                     "artifact-unload: %lld byte(s) never came back -- a plain close leaves the fatbin registered\n",
                     held);
        rc = 1;
      }
    }
  }

  std::filesystem::remove_all(dir, ec);
  std::printf("artifact-unload: %s (%d load/unload cycles)\n", rc == 0 ? "PASS" : "FAIL", kIters);
  std::fflush(stdout);
  std::fflush(stderr);
  return rc;
}
