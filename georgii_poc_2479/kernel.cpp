#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include <nvJitLink.h>
#include <nvrtc.h>

void check(nvrtcResult result)
{
  if (result != NVRTC_SUCCESS)
  {
    throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(result));
  }
}

void check(CUresult result)
{
  if (result != CUDA_SUCCESS)
  {
    const char* str = nullptr;
    cuGetErrorString(result, &str);
    throw std::runtime_error(std::string("CUDA error: ") + str);
  }
}

void check(nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS)
  {
    throw std::runtime_error(std::string("nvJitLink error: ") + std::to_string(result));
  }
}

extern "C" void host_code(
  int iterator_size,
  int iterator_alignment,
  void* pointer_to_cpu_bytes_storing_value,
  const char* prefix,
  int num_items,
  void* pointer_to_to_gpu_memory,
  const void** input_ltoirs,
  const int* input_ltoir_sizes,
  int num_input_ltoirs)
{
  std::string deref = std::string("#define DEREF ") + prefix + "_dereference\n"
                    + "extern \"C\" __device__ int DEREF(const char *state); \n";
  std::string adv = std::string("#define ADV ") + prefix + "_advance\n"
                  + "extern \"C\" __device__ void ADV(char *state, int distance); \n";
  std::string state = std::string("struct __align__(") + std::to_string(iterator_alignment) + R"XXX() iterator_t {
    // using iterator_category = cuda::std::random_access_iterator_tag; // TODO add include to libcu++
    using value_type = int;
    using difference_type = int;
    using pointer = int;
    using reference = int;
    __device__ value_type operator*() const { return DEREF(data); }
    __device__ iterator_t& operator+=(difference_type diff) {
        ADV(data, diff);
        return *this;
    }
    __device__ value_type operator[](difference_type diff) const {
        return *(*this + diff);
    }
    __device__ iterator_t operator+(difference_type diff) const {
        iterator_t result = *this;
        result += diff;
        return result;
    }
    char data[)XXX" + std::to_string(iterator_size)
                    + "]; };\n";

  // CUB kernel accepts an iterator and does some of the following operations on it
  std::string kernel_source = deref + adv + state + R"XXX(
    extern "C" __global__ void device_code(int num_items, iterator_t iterator, int *pointer) {
      iterator_t it = iterator + blockIdx.x;
      for (int i = 0; i < num_items; i++) {
        pointer[i] = it[i];
      }
    }
  )XXX";

  nvrtcProgram prog;
  const char* name = "test_kernel";
  nvrtcCreateProgram(&prog, kernel_source.c_str(), name, 0, nullptr, nullptr);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major     = deviceProp.major;
  const int cc_minor     = deviceProp.minor;
  const std::string arch = std::string("-arch=sm_") + std::to_string(cc_major) + std::to_string(cc_minor);

  const char* args[] = {arch.c_str(), "-rdc=true", "-dlto"};
  const int num_args = sizeof(args) / sizeof(args[0]);

  std::size_t log_size{};
  nvrtcResult compile_result = nvrtcCompileProgram(prog, num_args, args);

  check(nvrtcGetProgramLogSize(prog, &log_size));

  std::unique_ptr<char[]> log{new char[log_size]};
  check(nvrtcGetProgramLog(prog, log.get()));

  if (log_size > 1)
  {
    std::cerr << log.get() << std::endl;
  }

  check(compile_result);

  std::size_t ltoir_size{};
  check(nvrtcGetLTOIRSize(prog, &ltoir_size));
  std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
  check(nvrtcGetLTOIR(prog, ltoir.get()));
  check(nvrtcDestroyProgram(&prog));

  nvJitLinkHandle handle;
  const char* lopts[] = {"-lto", arch.c_str()};
  check(nvJitLinkCreate(&handle, 2, lopts));

  check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, name));

  for (int ltoir_id = 0; ltoir_id < num_input_ltoirs; ltoir_id++)
  {
    check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, input_ltoirs[ltoir_id], input_ltoir_sizes[ltoir_id], name));
  }

  check(nvJitLinkComplete(handle));

  std::size_t cubin_size{};
  check(nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  check(nvJitLinkGetLinkedCubin(handle, cubin.get()));
  check(nvJitLinkDestroy(&handle));

  CUlibrary library;
  CUkernel kernel;
  cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&kernel, library, "device_code"));

  void* pointer_to_cpu_bytes_storing_pointer_to_gpu_memory = &pointer_to_to_gpu_memory;
  void* kernel_args[]                                      = {
    &num_items, pointer_to_cpu_bytes_storing_value, pointer_to_cpu_bytes_storing_pointer_to_gpu_memory};

  check(cuLaunchKernel((CUfunction) kernel, 1, 1, 1, 1, 1, 1, 0, 0, kernel_args, nullptr));
  check(cuStreamSynchronize(0));
  check(cuLibraryUnload(library));
}
