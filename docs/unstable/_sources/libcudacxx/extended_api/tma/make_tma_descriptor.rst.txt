.. _libcudacxx-extended-api-tma-make_tma_descriptor:

``cuda::make_tma_descriptor``
=============================

Defined in the ``<cuda/tma>`` header.

**Function signatures**

.. code:: cuda

    namespace cuda {

    [[nodiscard]] inline
    CUtensorMap make_tma_descriptor(
      const DLTensor&            tensor,
      cuda::std::span<const int> box_sizes,
      cuda::std::span<const int> elem_strides,
      tma_interleave_layout      interleave_layout = tma_interleave_layout::none,
      tma_swizzle                swizzle           = tma_swizzle::none,
      tma_l2_fetch_size          l2_fetch_size     = tma_l2_fetch_size::none,
      tma_oob_fill               oobfill           = tma_oob_fill::none);

    [[nodiscard]] inline
    CUtensorMap make_tma_descriptor(
        const DLTensor&            tensor,
        cuda::std::span<const int> box_sizes,
        tma_interleave_layout      interleave_layout = tma_interleave_layout::none,
        tma_swizzle                swizzle           = tma_swizzle::none,
        tma_l2_fetch_size          l2_fetch_size     = tma_l2_fetch_size::none,
        tma_oob_fill               oobfill           = tma_oob_fill::none);

    } // namespace cuda

**Enumerators**

.. code:: cuda

    namespace cuda {

    enum class tma_oob_fill { none, nan };

    enum class tma_l2_fetch_size { none, bytes64, bytes128, bytes256 };

    enum class tma_interleave_layout { none, bytes16, bytes32 };

    enum class tma_swizzle {
        none,
        bytes32,
        bytes64,
        bytes128,
        bytes128_atom_32B,        // only CUDA Toolkit 12.8 and later, compute capability >= 10
        bytes128_atom_32B_flip_8B,// only CUDA Toolkit 12.8 and later, compute capability >= 10
        bytes128_atom_64B         // only CUDA Toolkit 12.8 and later, compute capability >= 10
    };

    } // namespace cuda

The functions construct a `CUDA Tensor Memory Accelerator (TMA) descriptor <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays>`__ from a ``DLTensor``. The resulting ``CUtensorMap`` can be bound to TMA-based copy instructions to efficiently stage multi-dimensional tiles in shared memory on Compute Capability 9.0 and newer GPUs.


.. note::

  - **DLPack** assumes *row-major* convention for sizes and strides, where the fastest changing dimension is the last one (``rank - 1``).
  - **cuTensorMap** assumes *column-major* convention for sizes and strides, where the fastest changing dimension is the first one (``0``).
  - ``box_sizes`` and ``elem_strides`` are expected to be in the same order as the input tensor's dimensions provided by **DLPack**, namely *row-major*.

Parameters
----------

- ``tensor``: The DLPack tensor describing the logical layout in device memory.
- ``box_sizes``: Extent of the shared memory tile, one entry per tensor dimension.
- ``elem_strides``: Stride, in elements, between consecutive accesses inside the shared memory tile. The second overload assumes a stride of ``1`` for every dimension with the special meaning of contiguous memory.

*Optional parameters*:

- ``interleave_layout``: Interleaving applied to the underlying memory.
- ``swizzle``: Swizzle pattern matching the chosen interleave layout.
- ``l2_fetch_size``: L2 cache promotion for TMA transfers.
- ``oobfill``: Out-of-bounds fill policy for floating-point tensors.

Return value
------------

- ``CUtensorMap`` encoding all metadata required to launch TMA transfers.

Preconditions
-------------

See :ref:`libcudacxx-extended-api-exceptions` for more details on exception handling.

**General preconditions**:

* Compute Capability 9.0 or newer is required.
* ``dlpack/dlpack.h`` (`DLPack v1 <https://github.com/dmlc/dlpack>`__) must be discoverable at compile time, namely available in the include path.

**DLPack preconditions**:

``tensor.device.device_type``:

* Must be ``kDLCUDA`` or ``kDLCUDAManaged``.

``tensor.device.device_id``:

* Must be a valid GPU device ordinal
* The selected device must have Compute Capability 9.0 or newer.

``tensor.ndim`` (rank):

* Must be greater than 0 and less than or equal to 5.
* Must be greater than or equal to ``3`` when an interleaved layout is requested.

``tensor.dtype``:

* ``kDLUInt``:

  - ``bits == 4``, ``lanes == 16``, namely ``U4 x 16``. Additionally, the innermost dimension must be a multiple of ``2`` when only 16-byte alignment is available. Requires CUDA Toolkit 12.8 and later, and compute capability >= 10.
  - ``bits == 8``, ``lanes == 1``, namely ``uint8_t``.
  - ``bits == 16``, ``lanes == 1``, namely ``uint16_t``.
  - ``bits == 32``, ``lanes == 1``, namely ``uint32_t``.
  - ``bits == 64``, ``lanes == 1``, namely ``uint64_t``.

* ``kDLInt``

  - ``bits == 32``, ``lanes == 1``, namely ``int32_t``.
  - ``bits == 64``, ``lanes == 1``, namely ``int64_t``.

* ``kDLFloat``

  - ``bits == 16``, ``lanes == 1``, namely ``__half``.
  - ``bits == 32``, ``lanes == 1``, namely ``float``.
  - ``bits == 64``, ``lanes == 1``, namely ``double``.

* ``kDLBfloat``

  - ``bits == 16``, ``lanes == 1``, namely ``__nv_bfloat16``.

* ``kDLFloat4_e2m1fn``

  - ``bits == 4``, ``lanes == 16``, mapped to ``U4 x 16``.  See ``kDLUInt`` for additional requirements.

* ``kDLBool``, ``kDLFloat8_e3m4``, ``kDLFloat8_e4m3``, ``kDLFloat8_e4m3b11fnuz``, ``kDLFloat8_e4m3fn``, ``kDLFloat8_e4m3fnuz``, ``kDLFloat8_e5m2``, ``kDLFloat8_e5m2fnuz``, ``kDLFloat8_e8m0fnu``: mapped to ``uint8_t``.

``tensor.data`` (pointer):

* Must be a valid GPU global address.
* Must be aligned to at least 16 bytes. Must be aligned to 32 bytes when ``interleave_layout`` is ``bytes32``.

``tensor.shape``:

* Must be greater than 0 and not exceed ``2^32`` elements per dimension.
* The innermost dimension must be a multiple of ``2`` when ``kDLFloat4_e2m1fn`` or ``U4 x 16`` are used.

``tensor.strides``:

* Each stride in bytes, namely ``tensor.strides[i] * element_size``, must be greater than 0 and not exceed ``2^40`` bytes per dimension.
* The tensor mapping must be unique, namely ``tensor.strides[i]`` must be greater than or equal to ``tensor.shape[i - 1] * strides[i - 1]`` or equal to ``0``.
* Each stride in bytes must be a multiple of the alignment 16 bytes when ``interleave_layout`` is ``none`` or ``bytes16``. It must be a multiple of 32 bytes when ``interleave_layout`` is ``bytes32``.
* ``tensor.strides`` can be ``nullptr`` to indicate that the tensor is contiguous in memory.

**User parameter preconditions**:

``box_sizes``, ``elem_strides``, and ``tensor.ndim`` must have the same rank.

``box_sizes``:

* Must be positive and not exceed ``256`` elements per dimension.
* ``box_sizes[i]`` must be less than or equal to ``tensor.shape[i]``.
* The full size of ``box_sizes`` must fit in shared memory.
* If the ``interleave_layout`` is ``tma_interleave_layout::none``, the inner dimension in bytes, computed as ``box_sizes[rank - 1] * element_size`` has the following additional requirements:

  - It must be a multiple of 16 bytes.
  - It must not exceed the byte-width of the selected ``swizzle`` pattern (``32``, ``64``, or ``128`` bytes).

``elem_strides``:

* Must be positive and not exceed ``8`` elements per dimension.
* ``elem_strides[i]`` must be less than or equal to ``tensor.shape[i]``.
*  If the ``interleave_layout`` is ``tma_interleave_layout::none``, the innner dimension (``elem_strides[0]``) is ignored.

``oobfill``:

* Must be ``tma_oob_fill::none`` for all integer data types.

``interleave_layout``:

* If ``interleave_layout`` is ``tma_interleave_layout::bytes32``, ``swizzle`` must be ``tma_swizzle::bytes32``.

References
----------

- `DLPack C API <https://dmlc.github.io/dlpack/latest/c_api.html>`__ documentation.
- `CUDA Tensor Memory Accelerator (TMA) <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays>`__ documentation.
- ``cuTensorMapEncodeTiled()`` `CUDA driver API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7>`__ documentation.

Example
-------

.. code:: cuda

    #include <cuda/tma>
    #include <cuda/std/cstdint>
    #include <dlpack/dlpack.h>

    CUtensorMap create_2d_tile_descriptor(float* device_ptr) {
        // Define DLPack tensor descriptor, commonly provided externally by the user, library, or framework.
        constexpr int64_t shape_storage[2]   = {64, 64};
        constexpr int64_t strides_storage[2] = {64, 1};

        DLTensor tensor{};
        tensor.data        = device_ptr;
        tensor.device      = {kDLCUDA, 0};
        tensor.ndim        = 2;
        tensor.dtype.code  = static_cast<uint8_t>(kDLFloat);
        tensor.dtype.bits  = 32;
        tensor.dtype.lanes = 1;
        tensor.shape       = const_cast<int64_t*>(shape_storage);
        tensor.strides     = const_cast<int64_t*>(strides_storage);
        tensor.byte_offset = 0;

        // Define shared memory box sizes and element strides.
        constexpr int BoxSizeX      = 8; // rows
        constexpr int BoxSizeY      = 8; // columns
        int box_sizes_storage[2]    = {BoxSizeX, BoxSizeY};
        int elem_strides_storage[2] = {BoxSizeY, 1}; // {1, ..., 1} is also valid to specify contiguous memory

        return cuda::make_tma_descriptor(tensor, box_sizes_storage, elem_strides_storage);
    }
