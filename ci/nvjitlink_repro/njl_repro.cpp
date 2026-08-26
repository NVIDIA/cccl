// Standalone reproducer for the Windows CUDA driver fail-fast seen in CCCL's
// Python CI (exception 0xC0000409 raised inside nvcuda64.dll).
//
// It replicates only the driver-facing shape of cuda.compute's build path --
// nvrtc compile to LTO-IR, nvJitLink link under -lto, then cuLibraryLoadData /
// cuLibraryGetKernel / cuLibraryUnload -- with no CCCL, Python, or CuPy
// involved. Each iteration compiles a uniquely *named* kernel so every cubin is
// distinct and cannot be served from a driver cache.
//
// Build (Windows):
//   cl /nologo /std:c++17 /EHsc /O2 /I"%CUDA_PATH%\include" njl_repro.cpp ^
//      /link "%CUDA_PATH%\lib\x64\nvrtc.lib" "%CUDA_PATH%\lib\x64\nvJitLink.lib" ^
//            "%CUDA_PATH%\lib\x64\cuda.lib"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvJitLink.h>
#include <nvrtc.h>

namespace
{
void die(const char* what, const char* detail, int iteration)
{
  std::fprintf(stderr, "FAIL at iteration %d: %s: %s\n", iteration, what, detail);
  std::fflush(stderr);
  std::exit(2);
}

void check_cu(CUresult rc, const char* what, int iteration)
{
  if (rc == CUDA_SUCCESS)
  {
    return;
  }
  const char* name = nullptr;
  cuGetErrorName(rc, &name);
  die(what, name ? name : "unknown CUDA error", iteration);
}

void check_nvrtc(nvrtcResult rc, const char* what, int iteration)
{
  if (rc == NVRTC_SUCCESS)
  {
    return;
  }
  die(what, nvrtcGetErrorString(rc), iteration);
}

void check_njl(nvJitLinkResult rc, const char* what, int iteration)
{
  if (rc == NVJITLINK_SUCCESS)
  {
    return;
  }
  die(what, ("nvJitLinkResult=" + std::to_string(static_cast<int>(rc))).c_str(), iteration);
}

// One full build+load+unload cycle, mirroring c/parallel/src/*.cu.
void one_cycle(int iteration, const std::string& arch_compute, const std::string& arch_sm)
{
  const std::string name = "repro_kernel_" + std::to_string(iteration);
  const std::string src =
    "extern \"C\" __global__ void " + name
    + "(int* out) {\n"
      "  if (out) { out[threadIdx.x] = "
    + std::to_string(iteration)
    + "; }\n"
      "}\n";

  nvrtcProgram prog{};
  check_nvrtc(nvrtcCreateProgram(&prog, src.c_str(), (name + ".cu").c_str(), 0, nullptr, nullptr),
              "nvrtcCreateProgram",
              iteration);

  // -dlto is what makes nvrtc emit LTO-IR rather than PTX.
  const char* opts[] = {"-dlto", arch_compute.c_str()};
  if (nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS)
  {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::vector<char> log(log_size ? log_size : 1);
    nvrtcGetProgramLog(prog, log.data());
    die("nvrtcCompileProgram", log.data(), iteration);
  }

  std::size_t ltoir_size = 0;
  check_nvrtc(nvrtcGetLTOIRSize(prog, &ltoir_size), "nvrtcGetLTOIRSize", iteration);
  std::vector<char> ltoir(ltoir_size);
  check_nvrtc(nvrtcGetLTOIR(prog, ltoir.data()), "nvrtcGetLTOIR", iteration);
  check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram", iteration);

  nvJitLinkHandle linker{};
  const char* link_opts[] = {"-lto", arch_sm.c_str()};
  check_njl(nvJitLinkCreate(&linker, 2, link_opts), "nvJitLinkCreate", iteration);
  check_njl(nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, ltoir.data(), ltoir.size(), name.c_str()),
            "nvJitLinkAddData",
            iteration);
  check_njl(nvJitLinkComplete(linker), "nvJitLinkComplete", iteration);

  std::size_t cubin_size = 0;
  check_njl(nvJitLinkGetLinkedCubinSize(linker, &cubin_size), "nvJitLinkGetLinkedCubinSize", iteration);
  std::vector<char> cubin(cubin_size);
  check_njl(nvJitLinkGetLinkedCubin(linker, cubin.data()), "nvJitLinkGetLinkedCubin", iteration);
  check_njl(nvJitLinkDestroy(&linker), "nvJitLinkDestroy", iteration);

  // The driver-facing half: this is where the fail-fast is raised in CI.
  CUlibrary library{};
  check_cu(cuLibraryLoadData(&library, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
           "cuLibraryLoadData",
           iteration);
  CUkernel kernel{};
  check_cu(cuLibraryGetKernel(&kernel, library, name.c_str()), "cuLibraryGetKernel", iteration);
  check_cu(cuLibraryUnload(library), "cuLibraryUnload", iteration);
}
} // namespace

int main(int argc, char** argv)
{
  const int max_iterations = argc > 1 ? std::atoi(argv[1]) : 100000;
  const int max_seconds    = argc > 2 ? std::atoi(argv[2]) : 0;

  check_cu(cuInit(0), "cuInit", -1);
  CUdevice device{};
  check_cu(cuDeviceGet(&device, 0), "cuDeviceGet", -1);
  // Primary context rather than cuCtxCreate: the latter became cuCtxCreate_v4 in
  // CUDA 13 and takes a params struct, whereas cuDevicePrimaryCtxRetain is stable
  // across 12 and 13 -- and it is the context cuda.compute actually runs on.
  CUcontext context{};
  check_cu(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain", -1);
  check_cu(cuCtxSetCurrent(context), "cuCtxSetCurrent", -1);

  int major = 0, minor = 0;
  check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
           "cuDeviceGetAttribute(major)",
           -1);
  check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
           "cuDeviceGetAttribute(minor)",
           -1);
  const std::string cc           = std::to_string(major) + std::to_string(minor);
  const std::string arch_compute = "-arch=compute_" + cc;
  const std::string arch_sm      = "-arch=sm_" + cc;

  char name[256] = {};
  cuDeviceGetName(name, sizeof(name), device);
  std::printf("device=%s sm_%s iterations=%d time_budget=%ds\n", name, cc.c_str(), max_iterations, max_seconds);
  std::fflush(stdout);

  const auto start = std::chrono::steady_clock::now();
  int completed    = 0;
  for (int i = 0; i < max_iterations; ++i)
  {
    one_cycle(i, arch_compute, arch_sm);
    completed = i + 1;

    if ((completed % 250) == 0)
    {
      const auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
      std::printf("  %d cycles, %llds elapsed\n", completed, static_cast<long long>(elapsed));
      std::fflush(stdout);
      if (max_seconds > 0 && elapsed >= max_seconds)
      {
        std::printf("time budget reached\n");
        break;
      }
    }
  }

  const auto elapsed =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
  std::printf("CLEAN: %d cycles in %llds with no fail-fast\n", completed, static_cast<long long>(elapsed));
  std::fflush(stdout);
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
