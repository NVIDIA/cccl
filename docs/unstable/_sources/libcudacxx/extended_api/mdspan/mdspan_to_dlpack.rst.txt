.. _libcudacxx-extended-api-mdspan-mdspan-to-dlpack:

``mdspan`` to DLPack
====================

This functionality provides a conversion from ``cuda::host_mdspan``, ``cuda::device_mdspan``, and ``cuda::managed_mdspan`` to `DLPack <https://dmlc.github.io/dlpack/latest/>`__ ``DLTensor`` view.

Defined in the ``<cuda/mdspan>`` header.

Conversion functions
--------------------

.. code:: cuda

   namespace cuda {

   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] /*dlpack_tensor*/<Extents::rank()>
   to_dlpack_tensor(const host_mdspan<T, Extents, Layout, Accessor>& mdspan);

   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] /*dlpack_tensor*/<Extents::rank()>
   to_dlpack_tensor(const device_mdspan<T, Extents, Layout, Accessor>& mdspan);

   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] /*dlpack_tensor*/<Extents::rank()>
   to_dlpack_tensor(const managed_mdspan<T, Extents, Layout, Accessor>& mdspan);

   } // namespace cuda

Types
-----

``/*dlpack_tensor*/`` is a internal helper class that stores a ``DLTensor`` and owns the backing storage for its ``shape`` and ``strides`` pointers. The class does not use any heap allocation.

.. code:: cuda

  namespace cuda {

  template <size_t Rank>
  struct /*dlpack_tensor*/ {
    // cuda::std::array<int64_t, Rank> shape;
    // cuda::std::array<int64_t, Rank> strides;

    DLTensor get() & const noexcept [[lifetimebound]];

    DLTensor get() && = delete;
  };

  } // namespace cuda

``/*dlpack_tensor*/`` stores a ``DLTensor`` and owns the backing storage for its ``shape`` and ``strides`` pointers. The class does not use any heap allocation.

.. note:: **Lifetime**

  The ``DLTensor`` associated with ``/*dlpack_tensor*/`` must not outlive the wrapper. If the wrapper is destroyed, the returned ``DLTensor::shape`` and ``DLTensor::strides`` pointers will dangle.

.. note:: **Const-correctness**

  ``DLTensor::data`` points at ``mdspan.data_handle()`` (or is ``nullptr`` if ``mdspan.size() == 0``). If ``T`` is ``const``, the pointer is ``const_cast``'d because ``DLTensor::data`` is unqualified.

Semantics
---------

The conversion produces a non-owning DLPack view of the ``mdspan`` data and metadata:

- ``DLTensor::ndim`` is ``mdspan.rank()``.
- For rank > 0, ``DLTensor::shape[i]`` is ``mdspan.extent(i)``.
- For rank > 0, ``DLTensor::strides[i]`` is ``mdspan.stride(i)``.
- ``DLTensor::byte_offset`` is always ``0``.
- ``DLTensor::device`` is:

  - ``{kDLCPU, 0}`` for ``cuda::host_mdspan``
  - ``{kDLCUDA, /*device_id*/}`` for ``cuda::device_mdspan``
  - ``{kDLCUDAManaged, 0}`` for ``cuda::managed_mdspan``

Element types are mapped to ``DLDataType`` according to the DLPack conventions, including:

- ``bool``.
- Signed and unsigned integers.
- IEEE-754 Floating-point and extended precision floating-point, including ``__half``, ``__nv_bfloat16``, ``__float128``, FP8, FP6, FP4 when available.
- Complex: ``cuda::std::complex<__half>``, ``cuda::std::complex<float>``, and ``cuda::std::complex<double>``.
- `CUDA built-in vector types <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-types>`__, such as ``int2``, ``float4``, etc.
- Vector types for extended floating-point, such as ``__half2``, ``__nv_fp8x4_e4m3``, etc.

Constraints
-----------

- The accessor ``data_handle_type`` must be a pointer type.

Runtime errors
--------------

- If any ``extent(i)`` or ``stride(i)`` cannot be represented in ``int64_t``, the conversion raises an ``std::invalid_argument`` exception.

Availability notes
------------------

- This API is available only when DLPack header is present, namely ``<dlpack/dlpack.h>`` is found in the include path.
- This API can be disabled by defining ``CCCL_DISABLE_DLPACK`` before including any library headers. In this case, ``<dlpack/dlpack.h>`` will not be included.

References
----------

- `DLPack C API <https://dmlc.github.io/dlpack/latest/c_api.html>`__ documentation.

Example
-------

.. code:: cuda

  #include <dlpack/dlpack.h>
  #include <cuda/mdspan>
  #include <cuda/std/cassert>
  #include <cuda/std/cstdint>

  int main() {
    using extents_t = cuda::std::extents<size_t, 2, 3>;

    int data[6] = {0, 1, 2, 3, 4, 5};
    cuda::host_mdspan<int, extents_t> md{data, extents_t{}};

    auto dl       = cuda::to_dlpack_tensor(md);
    auto dltensor = dl.get();

    // `dl` owns the shape/stride storage; `dltensor.data` is a non-owning pointer to `data`.
    assert(dltensor.device.device_type == kDLCPU);
    assert(dltensor.ndim == 2);
    assert(dltensor.shape[0] == 2 && dltensor.shape[1] == 3);
    assert(dltensor.strides[0] == 3 && dltensor.strides[1] == 1);
    assert(dltensor.data == data);
  }

Examples of invalid usage:

.. code:: cuda

  #include <dlpack/dlpack.h>
  #include <cuda/mdspan>
  #include <cuda/std/cstdint>

  void show_invalid_usage1() {
    using extents_t = cuda::std::extents<size_t, 2, 3>;

    int data[6] = {0, 1, 2, 3, 4, 5};
    cuda::host_mdspan<int, extents_t> md{data, extents_t{}};

    // WRONG: calling get() on a temporary is deleted to prevent dangling references.
    // const DLTensor& dltensor = cuda::to_dlpack_tensor(md).get(); // compile error
  }

.. code:: cuda

  #include <dlpack/dlpack.h>
  #include <cuda/mdspan>
  #include <cuda/std/cstdint>

  int64_t* show_invalid_usage2() {
    using extents_t = cuda::std::extents<size_t, 2, 3>;

    int data[6] = {0, 1, 2, 3, 4, 5};
    cuda::host_mdspan<int, extents_t> md{data, extents_t{}};

    auto dl       = cuda::to_dlpack_tensor(md);
    auto dltensor = dl.get();
    return dltensor.shape; // WRONG: returns a dangling pointer
  }
