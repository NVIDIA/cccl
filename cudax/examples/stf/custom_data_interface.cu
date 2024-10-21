//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief This example illustrates how to create a custom data interface and use them in tasks
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

/**
 * @brief A simple class describing a contiguous matrix of size (m, n)
 */
template <typename T>
class matrix
{
public:
  matrix(size_t m, size_t n, T* base)
      : m(m)
      , n(n)
      , base(base)
  {}

  __host__ __device__ T& operator()(size_t i, size_t j)
  {
    return base[i + j * m];
  }

  __host__ __device__ const T& operator()(size_t i, size_t j) const
  {
    return base[i + j * m];
  }

  size_t m, n;
  T* base;
};

/**
 * @brief defines the shape of a matrix
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <typename T>
class cuda::experimental::stf::shape_of<matrix<T>>
{
public:
  /**
   * @brief The default constructor.
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of() = default;

  explicit shape_of(size_t m, size_t n)
      : m(m)
      , n(n)
  {}

  /**
   * @name Copies a shape.
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of(const shape_of&) = default;

  /**
   * @brief Extracts the shape from a matrix
   *
   * @param M matrix to get the shape from
   *
   * All `shape_of` specializations must define this constructor.
   */
  shape_of(const matrix<T>& M)
      : shape_of<matrix<T>>(M.m, M.n)
  {}

  /// Mandatory method : defined the total number of elements in the shape
  size_t size() const
  {
    return m * n;
  }

  using coords_t = ::std::tuple<size_t, size_t>;

  // This transforms a tuple of (shape, 1D index) into a coordinate
  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    return coords_t(index % m, index / m);
  }

  size_t m;
  size_t n;
};

/**
 * @brief Data interface to manipulate a matrix in the CUDA stream backend
 */
template <typename T>
class matrix_stream_interface : public stream_data_interface_simple<matrix<T>>
{
public:
  using base = stream_data_interface_simple<matrix<T>>;
  using typename base::shape_t;

  /// Initialize from an existing matrix
  matrix_stream_interface(matrix<T> m)
      : base(std::move(m))
  {}

  /// Initialize from a shape of matrix
  matrix_stream_interface(typename base::shape_t s)
      : base(s)
  {}

  /// Copy the content of an instance to another instance
  ///
  /// This implementation assumes that we have registered memory if one of the data place is the host
  void stream_data_copy(
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    cudaStream_t stream) override
  {
    assert(src_memory_node != dst_memory_node);

    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
    if (src_memory_node == data_place::host)
    {
      kind = cudaMemcpyHostToDevice;
    }

    if (dst_memory_node == data_place::host)
    {
      kind = cudaMemcpyDeviceToHost;
    }

    const matrix<T>& src_instance = this->instance(src_instance_id);
    const matrix<T>& dst_instance = this->instance(dst_instance_id);

    size_t sz = src_instance.m * src_instance.n * sizeof(T);

    cuda_safe_call(cudaMemcpyAsync((void*) dst_instance.base, (void*) src_instance.base, sz, kind, stream));
  }

  /// allocate an instance on a specific data place
  ///
  /// setting *s to a negative value informs CUDASTF that the allocation
  /// failed, and that a memory reclaiming mechanism need to be performed.
  void stream_data_allocate(
    backend_ctx_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** /*unused*/,
    cudaStream_t stream) override
  {
    matrix<T>& instance = this->instance(instance_id);
    size_t sz           = instance.m * instance.n * sizeof(T);

    T* base_ptr;

    if (memory_node == data_place::host)
    {
      // Fallback to a synchronous method as there is no asynchronous host allocation API
      cuda_safe_call(cudaStreamSynchronize(stream));
      cuda_safe_call(cudaHostAlloc(&base_ptr, sz, cudaHostAllocMapped));
    }
    else
    {
      cuda_safe_call(cudaMallocAsync(&base_ptr, sz, stream));
    }

    // By filling a positive number, we notify that the allocation was succesful
    s = sz;

    instance.base = base_ptr;
  }

  /// deallocate an instance
  void stream_data_deallocate(
    backend_ctx_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* /*unused*/,
    cudaStream_t stream) override
  {
    matrix<T>& instance = this->instance(instance_id);
    if (memory_node == data_place::host)
    {
      // Fallback to a synchronous method as there is no asynchronous host deallocation API
      cuda_safe_call(cudaStreamSynchronize(stream));
      cuda_safe_call(cudaFreeHost(instance.base));
    }
    else
    {
      cuda_safe_call(cudaFreeAsync(instance.base, stream));
    }
  }

  /// Register the host memory associated to an instance of matrix
  ///
  /// Note that this pin_host_memory method is not mandatory, but then it is
  /// the responsability of the user to only passed memory that is already
  /// registered, and the allocation method on the host must allocate
  /// registered memory too. Otherwise, copy methods need to be synchronous.
  bool pin_host_memory(instance_id_t instance_id) override
  {
    matrix<T>& instance = this->instance(instance_id);
    if (!instance.base)
    {
      return false;
    }

    cuda_safe_call(pin_memory(instance.base, instance.m * instance.n * sizeof(T)));

    return true;
  }

  /// Unregister memory pinned by pin_host_memory
  void unpin_host_memory(instance_id_t instance_id) override
  {
    matrix<T>& instance = this->instance(instance_id);
    unpin_memory(instance.base);
  }
};

/**
 * @brief Define how the CUDA stream backend must manipulate a matrix
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends streamed_interface_of
 */
template <typename T>
struct cuda::experimental::stf::streamed_interface_of<matrix<T>>
{
  using type = matrix_stream_interface<T>;
};

/**
 * @brief A hash of the matrix
 */
template <typename T>
struct cuda::experimental::stf::hash<matrix<T>>
{
  std::size_t operator()(matrix<T> const& m) const noexcept
  {
    // Combine hashes from the base address and sizes
    return cuda::experimental::stf::hash_all(m.m, m.n, m.base);
  }
};

template <typename T>
__global__ void kernel(matrix<T> M)
{
  int tid_x      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads_x = gridDim.x * blockDim.x;

  int tid_y      = blockIdx.y * blockDim.y + threadIdx.y;
  int nthreads_y = gridDim.y * blockDim.y;

  for (int x = tid_x; x < M.m; x += nthreads_x)
  {
    for (int y = tid_y; y < M.n; y += nthreads_y)
    {
      M(x, y) += -x + 7 * y;
    }
  }
}

int main()
{
  stream_ctx ctx;

  const size_t m = 8;
  const size_t n = 10;
  std::vector<int> v(m * n);

  matrix<int> M(m, n, &v[0]);

  // M(i,j) = 17 * i + 23 * j
  for (size_t j = 0; j < n; j++)
  {
    for (size_t i = 0; i < m; i++)
    {
      M(i, j) = 17 * i + 23 * j;
    }
  }

  auto lM = ctx.logical_data(M);

  // M(i,j) +=  -i + 7*i
  ctx.task(lM.rw())->*[](cudaStream_t s, auto dM) {
    kernel<<<dim3(8, 8), dim3(8, 8), 0, s>>>(dM);
  };

  // M(i,j) +=  2*i + 6*j
  ctx.parallel_for(lM.shape(), lM.rw())->*[] _CCCL_DEVICE(size_t i, size_t j, auto dM) {
    dM(i, j) += 2 * i + 6 * j;
  };

  ctx.finalize();

  for (size_t j = 0; j < n; j++)
  {
    for (size_t i = 0; i < m; i++)
    {
      assert(M(i, j) == (17 * i + 23 * j) + (-i + 7 * j) + (2 * i + 6 * j));
    }
  }
}
