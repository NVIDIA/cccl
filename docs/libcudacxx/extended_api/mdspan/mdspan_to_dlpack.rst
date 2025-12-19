.. _libcudacxx-extended-api-mdspan-mdspan-to-dlpack:

``mdspan`` to DLPack
====================

This functionality provides a conversion from ``cuda::host_mdspan``, ``cuda::device_mdspan``, and ``cuda::managed_mdspan`` to a `DLPack <https://dmlc.github.io/dlpack/latest/>`__ ``DLTensor`` view.

Defined in the ``<cuda/mdspan>`` header.

Conversion functions
--------------------

.. code:: cuda

   namespace cuda {

   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] DLPackWrapper<Extents::rank()>
   mdspan_to_dlpack(const cuda::host_mdspan<T, Extents, Layout, Accessor>& mdspan);
 
   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] DLPackWrapper<Extents::rank()>
   mdspan_to_dlpack(const cuda::device_mdspan<T, Extents, Layout, Accessor>& mdspan,
                    cuda::device_ref device = cuda::device_ref{0});
 
   template <typename T, typename Extents, typename Layout, typename Accessor>
   [[nodiscard]] DLPackWrapper<Extents::rank()>
   mdspan_to_dlpack(const cuda::managed_mdspan<T, Extents, Layout, Accessor>& mdspan);
 
   } // namespace cuda

Types
-----

.. code:: cuda

  namespace cuda {

  template <size_t Rank>
  class DLPackWrapper {
  public:
      DLPackWrapper();
      DLPackWrapper(const DLPackWrapper&) noexcept;
      DLPackWrapper(DLPackWrapper&&) noexcept;
      DLPackWrapper& operator=(const DLPackWrapper&) noexcept;
      DLPackWrapper& operator=(DLPackWrapper&&) noexcept;
      ~DLPackWrapper() noexcept = default;

      DLTensor*       operator->() noexcept;
      const DLTensor* operator->() const noexcept;
      DLTensor&       get() noexcept;
      const DLTensor& get() const noexcept;
    };

  } // namespace cuda

``cuda::DLPackWrapper`` stores a ``DLTensor`` and owns the backing storage for its ``shape`` and ``strides`` pointers. The class does not use any heap allocation.

.. note:: Lifetime

  The ``DLTensor`` associated with ``cuda::DLPackWrapper`` must not outlive the wrapper. If the wrapper is destroyed, the returned ``DLTensor::shape`` and ``DLTensor::strides`` pointers will dangle.

.. note:: Const-correctness

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
  - ``{kDLCUDA, device.get()}`` for ``cuda::device_mdspan``
  - ``{kDLCUDAManaged, 0}`` for ``cuda::managed_mdspan``

Element types are mapped to ``DLDataType`` according to the DLPack conventions, including:

- Signed and unsigned integers.
- IEEE-754 Floating-point and extended precision floating-point, including ``__half``, ``__nv_bfloat16``, FP8, FP6, FP4 when available.
- Complex: ``cuda::std::complex<__half>``, ``cuda::std::complex<float>``, and ``cuda::std::complex<double>``.
- `CUDA built-in vector types <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#built-in-types>`__, such as ``int2``, ``float4``, etc..

Constraints and errors
----------------------

**Constraints**

- The accessor ``data_handle_type`` must be a pointer type.

**Runtime errors**

- If any ``extent(i)`` or ``stride(i)`` cannot be represented in ``int64_t``, the conversion raises an exception.

Availability notes
------------------

- This API is available only when DLPack headers are present (``<dlpack/dlpack.h>`` is found in the include path).
* ``dlpack/dlpack.h`` (`DLPack v1 <https://github.com/dmlc/dlpack>`__) must be discoverable at compile time, namely available in the include path.

References
----------

- `DLPack C API <https://dmlc.github.io/dlpack/latest/c_api.html>`__ documentation.

Example
-------

.. code:: cuda

  #include <cuda/mdspan>

  #include <dlpack/dlpack.h>
  #include <cassert>
  #include <cstdint>

  int main() {
    using extents_t = cuda::std::extents<std::size_t, 2, 3>;

    int data[6] = {0, 1, 2, 3, 4, 5};
    cuda::host_mdspan<int, extents_t> md{data, extents_t{}};

    auto dl = cuda::mdspan_to_dlpack(md);

    // `dl` owns the shape/stride storage; `dl->data` is a non-owning pointer to `data`.
    assert(dl->device.device_type == kDLCPU);
    assert(dl->ndim == 2);
    assert(dl->shape[0] == 2 && dl->shape[1] == 3);
    assert(dl->strides[0] == 3 && dl->strides[1] == 1);
    assert(dl->data == data);
  }
