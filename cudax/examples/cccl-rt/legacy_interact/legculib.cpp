#include "legculib.h"

#include <cuda.h>
#include <nvrtc.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

//--------------------------------------------------------------------------------------------------------------------//
// Library private data
//--------------------------------------------------------------------------------------------------------------------//

//! @brief The CUDA device handle.
static CUdevice device{};

//! @brief The device compute capability.
static int device_cc{};

//! @brief The device primary context.
static CUcontext context{};

//! @brief The device memory pool.
static CUmemoryPool device_memory_pool{};

//! @brief The cache of modules containing the kernels for sum_of_products.
static CUmodule sum_of_products_modules[LEGCULIB_DATATYPE_COUNT]{};

//! @brief The cache of kernel handles for sum_of_products.
static CUfunction sum_of_products_kernels[LEGCULIB_DATATYPE_COUNT]{};

//! @brief Array of datatype names. Should be indexed by \c legculib_datatype.
static constexpr const char* datatype_name[]{
  "int",
  "long long",
  "unsigned",
  "unsigned long long",
  "float",
  "double",
};

//--------------------------------------------------------------------------------------------------------------------//
// Error handling
//--------------------------------------------------------------------------------------------------------------------//

//! @brief Gets the error string for a given \c error.
//! @note Public API.
extern "C" const char* legculib_get_error_string(legculib_error error)
{
  switch (error)
  {
    case LEGCULIB_SUCCESS:
      return "operation was successful";
    case LEGCULIB_ERROR_NOT_INITIALIZED:
      return "library was not initialized before calling the API";
    case LEGCULIB_ERROR_INVALID_ARGUMENT:
      return "an invalid argument was passed to the API call";
    case LEGCULIB_ERROR_KERNEL_COMPILATION:
      return "a kernel compilation failed";
    case LEGCULIB_ERROR_MALLOC:
      return "failed to allocate memory from the heap";
    case LEGCULIB_ERROR_CUDA_DRIVER:
      return "a CUDA Driver error occurred";
    default:
      return "unknown error";
  }
}

//! @brief Makes the success result.
static inline legculib_result success()
{
  return {LEGCULIB_SUCCESS, {}};
}

//! @brief Makes the error result.
template <legculib_error e>
static inline legculib_result error()
{
  static_assert(e != LEGCULIB_ERROR_CUDA_DRIVER);
  return {e, {}};
}

//! @brief Makes the error result for CUDA call.
static inline legculib_result cuda_error(CUresult error)
{
  return {LEGCULIB_ERROR_CUDA_DRIVER, static_cast<cudaError_t>(error)};
}

#define CALL_CUDA(...)                  \
  do                                    \
  {                                     \
    const auto _result = (__VA_ARGS__); \
    if (_result != CUDA_SUCCESS)        \
    {                                   \
      return cuda_error(_result);       \
    }                                   \
  } while (false)
#define CALL_NVRTC(...)                                  \
  do                                                     \
  {                                                      \
    const auto _result = (__VA_ARGS__);                  \
    if (_result != NVRTC_SUCCESS)                        \
    {                                                    \
      return error<LEGCULIB_ERROR_KERNEL_COMPILATION>(); \
    }                                                    \
  } while (false)

//--------------------------------------------------------------------------------------------------------------------//
// Initialization
//--------------------------------------------------------------------------------------------------------------------//

//! @brief Checks whether the library is initialized.
static inline bool is_initialized()
{
  return context != nullptr;
}

//! @brief Initializes the library for the ordinal device.
//! @note Public API.
extern "C" legculib_result legculib_init(int ordinal)
{
  if (is_initialized())
  {
    return success();
  }

  // Initialize CUDA Driver.
  CALL_CUDA(cuInit(0));

  // Get the CUdevice from the ordinal.
  CALL_CUDA(cuDeviceGet(&device, ordinal));

  // Get the compute capability.
  int cc_major{};
  CALL_CUDA(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  int cc_minor{};
  CALL_CUDA(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  device_cc = 10 * cc_major + cc_minor;

  // Retain the primary context.
  CALL_CUDA(cuDevicePrimaryCtxRetain(&context, device));

  // Create the memory pool to be used with the library.
  CUmemPoolProps memory_pool_props{};
  memory_pool_props.allocType   = CU_MEM_ALLOCATION_TYPE_PINNED;
  memory_pool_props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  memory_pool_props.location    = {CU_MEM_LOCATION_TYPE_DEVICE, ordinal};
  memory_pool_props.usage       = CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE;
  CALL_CUDA(cuMemPoolCreate(&device_memory_pool, &memory_pool_props));

  return success();
}

//! @brief Deinitializes the library.
//! @note Public API.
extern "C" legculib_result legculib_finalize()
{
  if (!is_initialized())
  {
    return success();
  }

  // Destroy the memory pool.
  CALL_CUDA(cuMemPoolDestroy(device_memory_pool));
  device_memory_pool = nullptr;

  // Unload all modules.
  for (auto& mod : sum_of_products_modules)
  {
    if (mod != nullptr)
    {
      CALL_CUDA(cuModuleUnload(mod));
      mod = nullptr;
    }
  }

  // Release the primary context.
  CALL_CUDA(cuDevicePrimaryCtxRelease(device));
  context = nullptr;

  return success();
}

//--------------------------------------------------------------------------------------------------------------------//
// Memory pool
//--------------------------------------------------------------------------------------------------------------------//

//! @brief Gets the device memory pool used by the legculib library.
//! @note Public API.
extern "C" legculib_result legculib_get_device_mempool(cudaMemPool_t* mempool_ptr)
{
  // Check initialization.
  if (!is_initialized())
  {
    return error<LEGCULIB_ERROR_NOT_INITIALIZED>();
  }

  *mempool_ptr = (cudaMemPool_t) device_memory_pool;
  return success();
}

//--------------------------------------------------------------------------------------------------------------------//
// Sum of products
//--------------------------------------------------------------------------------------------------------------------//

//! @brief The CUDA source file for sum of products kernel.
static constexpr char sum_of_product_src[] = R"(
#ifndef TYPE
#  error "TYPE macro must be defined."
#endif

using T = TYPE;
constexpr unsigned block_size = 128;

__device__ static __forceinline__ void warp_reduce(volatile T* sdata, unsigned tid)
{
  if constexpr (block_size >= 64) sdata[tid] += sdata[tid + 32];
  if constexpr (block_size >= 32) sdata[tid] += sdata[tid + 16];
  if constexpr (block_size >= 16) sdata[tid] += sdata[tid + 8];
  if constexpr (block_size >= 8) sdata[tid] += sdata[tid + 4];
  if constexpr (block_size >= 4) sdata[tid] += sdata[tid + 2];
  if constexpr (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

extern "C" __global__ void kernel(T* result, const T* a, const T* b, unsigned n)
{
  __shared__ T sdata[block_size];
  const unsigned tid      = threadIdx.x;
  const unsigned gridSize = block_size * 2 * gridDim.x;

  sdata[tid] = 0;
  for (unsigned i = blockIdx.x * (block_size * 2) + tid; i < n; i += gridSize)
  {
    sdata[tid] += a[i] * b[i] + a[i + block_size] + b[i + block_size];
  }
  __syncthreads();

  if constexpr (block_size >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if constexpr (block_size >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if constexpr (block_size >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
  if (tid < 32) warp_reduce(sdata, tid);

  if (threadIdx.x == 0)
  {
    atomicAdd(result, sdata[0]);
  }
}
)";

//! @brief Compiles and caches the sum of products kernel.
static legculib_result make_sum_of_products_kernel(legculib_datatype type, CUfunction& kernel)
{
  // Check if the kernel has been cached previously.
  if (sum_of_products_kernels[type] != nullptr)
  {
    kernel = sum_of_products_kernels[type];
    return success();
  }

  // Make the architecture option value.
  char arch_opt[128]{};
  snprintf(arch_opt, 128, "sm_%d", device_cc);

  // Make the TYPE=type definition option.
  char type_def[128]{};
  snprintf(type_def, 128, "TYPE=%s", datatype_name[type]);

  // Array of pointers to the compile options.
  const char* opts[]{
    "-std=c++17", // Use C++17 dialect.
    "-dopt=on", // Enable device optimizations.
    "-D",
    type_def, // Pass the compiled type.
    "-arch",
    arch_opt, // Define the architecture.
  };

  // Compile the kernel using NVRTC.
  nvrtcProgram prog{};
  CALL_NVRTC(nvrtcCreateProgram(&prog, sum_of_product_src, "sum_of_products.cu", 0, nullptr, nullptr));
  CALL_NVRTC(nvrtcCompileProgram(prog, sizeof(opts) / sizeof(const char*), opts));

  // Get the CUBIN size.
  size_t cubin_size{};
  CALL_NVRTC(nvrtcGetCUBINSize(prog, &cubin_size));

  // Allocate the memory for the CUBIN.
  char* cubin = reinterpret_cast<char*>(malloc(cubin_size));
  if (cubin == nullptr)
  {
    CALL_NVRTC(nvrtcDestroyProgram(&prog));
    return error<LEGCULIB_ERROR_MALLOC>();
  }

  // Get the CUBIN data.
  CALL_NVRTC(nvrtcGetCUBIN(prog, cubin));

  // Load the module with the kernel. Will be loaded to current context.
  CUmodule mod{};
  CALL_CUDA(cuCtxPushCurrent(context));
  CALL_CUDA(cuModuleLoadData(&mod, cubin));
  CALL_CUDA(cuCtxPopCurrent(nullptr));

  // Get the kernel from the module.
  CUfunction func{};
  CALL_CUDA(cuModuleGetFunction(&func, mod, "kernel"));

  // Store the module and kernel in the cache.
  sum_of_products_modules[type] = mod;
  sum_of_products_kernels[type] = func;

  // Free allocated memory and destroy the program.
  free(cubin);
  CALL_NVRTC(nvrtcDestroyProgram(&prog));

  // Write output parameters and return.
  kernel = func;
  return success();
}

//! @brief Computes the sum of products.
//! @note Public API.
extern "C" legculib_result legculib_sum_of_products(
  cudaStream_t stream, void* result, const void* a, const void* b, unsigned n, legculib_datatype type)
{
  // Check initialization.
  if (!is_initialized())
  {
    return error<LEGCULIB_ERROR_NOT_INITIALIZED>();
  }

  // Get the kernel.
  CUfunction kernel{};
  if (const auto build_result = make_sum_of_products_kernel(type, kernel); build_result.error != LEGCULIB_SUCCESS)
  {
    return build_result;
  }

  // Early exit if there is no work to be done.
  if (n == 0)
  {
    return success();
  }

  // Validate arguments.
  if (result == nullptr)
  {
    return error<LEGCULIB_ERROR_INVALID_ARGUMENT>();
  }
  if (a == nullptr)
  {
    return error<LEGCULIB_ERROR_INVALID_ARGUMENT>();
  }
  if (b == nullptr)
  {
    return error<LEGCULIB_ERROR_INVALID_ARGUMENT>();
  }
  if (type < 0 || type >= LEGCULIB_DATATYPE_COUNT)
  {
    return error<LEGCULIB_ERROR_INVALID_ARGUMENT>();
  }

  // Set-up the kernel configuration and arguments.
  constexpr auto threads_per_block = 128u;
  const auto blocks_per_grid       = (n + threads_per_block - 1) / threads_per_block;
  void* args[]{&result, &a, &b, &n};

  // Launch the kernel.
  CALL_CUDA(cuCtxPushCurrent(context));
  CALL_CUDA(cuLaunchKernel(kernel, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, stream, args, nullptr));
  CALL_CUDA(cuCtxPopCurrent(nullptr));

  return success();
}
