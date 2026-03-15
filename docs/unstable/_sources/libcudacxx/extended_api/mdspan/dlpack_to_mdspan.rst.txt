.. _libcudacxx-extended-api-mdspan-dlpack-to-mdspan:

DLPack to ``mdspan``
====================

This functionality provides a conversion from `DLPack <https://dmlc.github.io/dlpack/latest/>`__ ``DLTensor`` to ``cuda::host_mdspan``, ``cuda::device_mdspan``, and ``cuda::managed_mdspan``.

Defined in the ``<cuda/mdspan>`` header.

Conversion functions
--------------------

.. code:: cuda

   namespace cuda {

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::std::layout_stride>
   [[nodiscard]] cuda::host_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_host_mdspan(const DLTensor& tensor);

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::std::layout_stride>
   [[nodiscard]] cuda::device_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_device_mdspan(const DLTensor& tensor);

   template <typename ElementType, size_t Rank, typename LayoutPolicy = cuda::std::layout_stride>
   [[nodiscard]] cuda::managed_mdspan<ElementType, cuda::std::dims<Rank, int64_t>, LayoutPolicy>
   to_managed_mdspan(const DLTensor& tensor);

   } // namespace cuda

Template parameters
-------------------

- ``ElementType``: The element type of the resulting ``mdspan``. Must match the ``DLTensor::dtype``.
- ``Rank``: The number of dimensions. Must match ``DLTensor::ndim``.
- ``LayoutPolicy``: The layout policy for the resulting ``mdspan``. Defaults to ``cuda::std::layout_stride``. Supported layouts are:

  - ``cuda::std::layout_right`` (C-contiguous, row-major)
  - ``cuda::std::layout_left`` (Fortran-contiguous, column-major)
  - ``cuda::std::layout_stride`` (general strided layout)

Semantics
---------

The conversion produces a non-owning ``mdspan`` view of the ``DLTensor`` data:

- The ``mdspan`` data pointer is computed as ``static_cast<char*>(tensor.data) + tensor.byte_offset``.
- For ``rank > 0``, ``mdspan.extent(i)`` is ``tensor.shape[i]``.
- For ``layout_stride``, ``mdspan.stride(i)`` is ``tensor.strides[i]`` (or computed as row-major if ``strides`` is ``nullptr`` for DLPack < v1.2).
- The device type is validated:

  - ``kDLCPU`` for ``to_host_mdspan``
  - ``kDLCUDA`` for ``to_device_mdspan``
  - ``kDLCUDAManaged`` for ``to_managed_mdspan``

Supported element types:

- ``bool``.
- Signed and unsigned integers.
- IEEE-754 Floating-point and extended precision floating-point, including ``__half``, ``__nv_bfloat16``, ``__float128``, FP8, FP6, FP4 when available.
- Complex: ``cuda::std::complex<__half>``, ``cuda::std::complex<float>``, and ``cuda::std::complex<double>``.
- `CUDA built-in vector types <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-types>`__, such as ``int2``, ``float4``, etc.
- Vector types for extended floating-point, such as ``__half2``, ``__nv_fp8x4_e4m3``, etc.

Constraints
-----------

- ``LayoutPolicy`` must be one of ``cuda::std::layout_right``, ``cuda::std::layout_left``, or ``cuda::std::layout_stride``.
- For ``layout_right`` and ``layout_left``, the ``DLTensor`` strides must be compatible with the layout.

Runtime errors
--------------

The conversion throws ``std::invalid_argument`` in the following cases:

- ``DLTensor::ndim`` does not match the specified ``Rank``.
- ``DLTensor::dtype`` does not match ``ElementType``.
- ``DLTensor::data`` is ``nullptr``.
- ``DLTensor::shape`` is ``nullptr`` (for rank > 0).
- Any ``DLTensor::shape[i]`` is negative.
- ``DLTensor::strides`` is ``nullptr`` for DLPack v1.2 or later.
- ``DLTensor::strides`` is ``nullptr`` for ``layout_left`` with rank > 1 (DLPack < v1.2).
- ``DLTensor::strides[i]`` is not positive for ``layout_stride``.
- ``DLTensor::strides`` are not compatible with the requested ``layout_right`` or ``layout_left``.
- ``DLTensor::device.device_type`` does not match the target mdspan type.
- Data pointer is not properly aligned for the element type.

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
    int data[6] = {0, 1, 2, 3, 4, 5};

    // Create a DLTensor manually for demonstration
    int64_t shape[2]   = {2, 3};
    int64_t strides[2] = {3, 1};  // row-major strides

    DLTensor tensor{};
    tensor.data        = data;
    tensor.device      = {kDLCPU, 0};
    tensor.ndim        = 2;
    tensor.dtype       = DLDataType{kDLInt, 32, 1};
    tensor.shape       = shape;
    tensor.strides     = strides;
    tensor.byte_offset = 0;

    // Convert to host_mdspan
    auto md = cuda::to_host_mdspan<int, 2>(tensor);

    assert(md.rank() == 2);
    assert(md.extent(0) == 2 && md.extent(1) == 3);
    assert(md.stride(0) == 3 && md.stride(1) == 1);
    assert(md.data_handle() == data);
    assert(md(0, 0) == 0 && md(1, 2) == 5);
  }

See also
--------

- :ref:`libcudacxx-extended-api-mdspan-mdspan-to-dlpack` for the reverse conversion.
