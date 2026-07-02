.. _libcudacxx-extended-api-mdspan-shared-memory-accessor:

``shared_memory`` ``mdspan`` and ``accessor``
=============================================

``shared_memory`` ``mdspan`` and ``accessor`` allow to express multi-dimensional views of the CUDA shared memory space and provide additional safety checks and performance optimizations.

Types and Traits
----------------

.. code:: cpp

    namespace cuda {

    template <typename AccessorPolicy>
    using shared_memory_accessor;

    template <typename ElementType,
              typename Extents,
              typename LayoutPolicy   = cuda::std::layout_right,
              typename AccessorPolicy = cuda::shared_memory_accessor<ElementType>>
    class shared_memory_mdspan;

    } // namespace cuda

``mdspan`` type and accessor tailored for the *shared* memory space.

----

.. code:: cpp

    namespace cuda {

    template <typename T>
    inline constexpr bool is_shared_memory_accessor_v = /* true if T is a shared_memory_accessor, false otherwise */;

    template <typename T>
    inline constexpr bool is_shared_memory_mdspan_v = /* true if T is a shared_memory_mdspan, false otherwise */;

    } // namespace cuda

Features
--------

**Constraints**

- Accessor ``data_handle_type`` must be a pointer type.

**Preconditions**

- Accessing elements through a ``shared_memory_accessor`` is only allowed in device code.
- The underlying pointer must be in the *shared* memory space.
- Access offset must be within the maximum possible shared memory allocation size.

**Performance considerations**

- The functionality guarantees that the accesses use shared memory instructions (``STS/LDS``) rather than generic memory instructions.

Example
-------

.. code:: cuda

    #include <cuda/mdspan>
    #include <cstdio>

    __global__ void kernel() {
        extern __shared__ int shmem[];

        // Create a shared_memory_mdspan over the dynamic shared memory
        cuda::shared_memory_mdspan md(shmem, cuda:std::dims<2>{32, 32});

        if (threadIdx.x < 32) {
             md[threadIdx.x][threadIdx.x] = threadIdx.x; // write on the diagonal
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            printf("md[5][5] = %d\n", md[5][5]); // read from the diagonal
        }
    }

    int main() {
        kernel<<<1, 32, 32 * 32 * sizeof(int)>>>();
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/sojGnKoY9>`_
