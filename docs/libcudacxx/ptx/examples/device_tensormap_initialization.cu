#include <cuda.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

#include <cuda/ptx>

#include <cassert>
#include <cstdio>

// clang-format off

namespace ptx = cuda::ptx;

////////////////////////////////////////////////////////////////////////////////
// Error-checking macros
///////////////////////////////////////////////////////////////////////////////
#define CUDA_CHECK(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }

#define CU_CHECK(ans) \
  { cuAssert((ans), __FILE__, __LINE__); }

void cudaAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

void cuAssert(CUresult code, const char* file, int line) {
  if (code != CUDA_SUCCESS) {
    const char* err_name;
    const char* err_str;
    cuGetErrorName(code, &err_name);
    cuGetErrorString(code, &err_str);
    fprintf(stderr, "CU error: (%s) %s at %s:%d\n", err_name, err_str, file, line);
    exit(code);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host-side tensormap creation
///////////////////////////////////////////////////////////////////////////////

// Get pointer to driver API
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled();
// Make a tensormap template using the driver API that can be filled in by a
// device kernel.
CUtensorMap make_tensormap_template();

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuGetProcAddress
  cudaDriverEntryPointQueryResult driver_status;
  void* cuGetProcAddress_ptr = nullptr;
  cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault, &driver_status);
  assert(driver_status == cudaDriverEntryPointSuccess);
  PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

  // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
  CUdriverProcAddressQueryResult symbol_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUresult res                     = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
  assert(res == CUDA_SUCCESS && symbol_status == CU_GET_PROC_ADDRESS_SUCCESS);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

// example-begin make-template
CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}
// example-end make-template


////////////////////////////////////////////////////////////////////////////////
// Device-side tensormap initialization
///////////////////////////////////////////////////////////////////////////////

// Parameter struct to define the tensormap parameters:
struct tensormap_params;
// Kernel to initialize the global buffer;
__global__ void fill_global_buf(char* global_buf);
// Kernels that initialize and consume the tensormap:
__global__ void initialize_tensor_map(const __grid_constant__ CUtensorMap tmap_template, tensormap_params p, CUtensorMap* out);
__global__ void consume_tensor_map(CUtensorMap* tensor_map);

// example-begin tensormap_params
struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};
// example-end tensormap_params


__global__ void fill_global_buf(char* global_buf) {
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 256; ++i) {
      global_buf[256 * j + i] = j + i;
    }
  }
}

// example-begin modification
// launch with 1 warp.
__launch_bounds__(32)
__global__ void initialize_tensor_map(const __grid_constant__ CUtensorMap tmap_template, tensormap_params p, CUtensorMap* out) {
  __shared__ alignas(128) CUtensorMap smem_tmap;
  if (threadIdx.x == 0) {
    // Copy template to shared memory:
    smem_tmap = tmap_template;

    const auto space_shared = ptx::space_shared;
    ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
    // For field .rank, the operand new_val must be ones less than the desired
    // tensor rank as this field uses zero-based numbering.
    ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

    // Set box dimensions:
    if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
    if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
    if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
    if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
    if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
    // Set global dimensions:
    if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
    if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
    if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
    if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
    if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
    // Set global stride:
    if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
    if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
    if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
    if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
    // Set element stride:
    if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
    if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
    if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
    if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
    if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

    // These constants are documented in this table:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
    auto u8_elem_type = ptx::n32_t<0>{};
    ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
    auto no_interleave = ptx::n32_t<0>{};
    ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
    auto no_swizzle = ptx::n32_t<0>{};
    ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
    auto zero_fill = ptx::n32_t<0>{};
    ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
  }
  // Sync updates with other threads in warp
  __syncwarp();
  // Copy the tensormap to global memory collectively with threads in the warp.
  // In addition: make the updated tensormap visible to other threads on device that
  // for use with cp.async.bulk.
  ptx::n32_t<128> bytes_128;
  ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}
// example-end modification

// example-begin use
// Consumer of tensormap in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensormap:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence..

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Make barrier init visible in async proxy, i.e., to TMA engine
    ptx::fence_proxy_async(ptx::space_shared);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);

    // Arrive on barrier. Expect 4 * 128 bytes.
    // ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, 0);
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}
// example-end use

////////////////////////////////////////////////////////////////////////////////
// Tying it all together
///////////////////////////////////////////////////////////////////////////////

int main() {
  // example-begin overview
  // Initialize device context:
  CUDA_CHECK(cudaDeviceSynchronize());

  // Create a tensormap template
  CUtensorMap template_tensor_map = make_tensormap_template();

  // Allocate tensor map and tensor in global memory
  CUtensorMap* global_tensor_map;
  CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
  char* global_buf;
  CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

  // Fill global buffer with data.
  fill_global_buf<<<1, 1>>>(global_buf);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Define the parameters of the tensormap that will be created on device.
  tensormap_params p{};
  p.global_address    = global_buf;
  p.rank              = 2;
  p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
  p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
  p.global_dim[0]     = 256; //
  p.global_dim[1]     = 8;   //
  p.global_stride[0]  = 256; //
  p.element_stride[0] = 1;   //
  p.element_stride[1] = 1;   //

  // Initialize global_tensor_map on device:
  initialize_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Use it from another kernel:
  consume_tensor_map<<<1, 1>>>(global_tensor_map);

  // Check for errors:
  CUDA_CHECK(cudaDeviceSynchronize());
  // example-end overview
}
