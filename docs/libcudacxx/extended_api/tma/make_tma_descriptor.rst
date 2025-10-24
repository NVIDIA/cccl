.. _libcudacxx-extended-api-tma-make_tma_descriptor:

``cuda::make_tma_descriptor``
=============================

Defined in the ``<cuda/tma>`` header.

**Function signatures**

.. code:: cuda

    namespace cuda {

    template <size_t BoxDimSize, size_t ElemStrideSize>
    [[nodiscard]] inline CUtensorMap
    make_tma_descriptor(
      const DLTensor&                            tensor,
      cuda::std::span<const int, BoxDimSize>     box_sizes,
      cuda::std::span<const int, ElemStrideSize> elem_strides,
      tma_interleave_layout                      interleave_layout = tma_interleave_layout::none,
      tma_swizzle                                swizzle           = tma_swizzle::none,
      tma_l2_fetch_size                          l2_fetch_size     = tma_l2_fetch_size::none,
      tma_oob_fill                               oobfill           = tma_oob_fill::none) noexcept;

    template <size_t BoxDimSize>
    [[nodiscard]] inline CUtensorMap
    make_tma_descriptor(
        const DLTensor&                        tensor,
        cuda::std::span<const int, BoxDimSize> box_sizes,
        tma_interleave_layout                  interleave_layout = tma_interleave_layout::none,
        tma_swizzle                            swizzle           = tma_swizzle::none,
        tma_l2_fetch_size                      l2_fetch_size     = tma_l2_fetch_size::none,
        tma_oob_fill                           oobfill           = tma_oob_fill::none) noexcept;

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
        bytes128_atom_32B,        // only CUDA Toolkit 12.8 and later
        bytes128_atom_32B_flip_8B,// only CUDA Toolkit 12.8 and later
        bytes128_atom_64B         // only CUDA Toolkit 12.8 and later
    };

    } // namespace cuda

The functions construct a `CUDA Tensor Memory Accelerator (TMA) descriptor <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays>`__ from a ``DLTensor``. The resulting ``CUtensorMap`` can be bound to TMA-based copy instructions to efficiently stage multi-dimensional tiles in shared memory on Compute Capability 9.0 and newer GPUs.

The API is available when ``dlpack/dlpack.h`` (`DLPack v1 <https://github.com/dmlc/dlpack>`__) is discovered at compile time.

.. note::

  - **DLPack** assumes *row-major* convention for sizes and strides, where the fastest changing dimension is the last one (``rank - 1``).
  - **cuTensorMap** assumes *column-major* convention for sizes and strides, where the fastest changing dimension is the first one (``0``).
  - ``box_sizes`` and ``elem_strides`` are expected to be in the same order as the input tensor's dimensions provided by **DLPack**, namely *row-major*.

Parameters
----------

- ``tensor``: The DLPack tensor describing the logical layout in device memory.
- ``box_sizes``: Extent of the shared memory tile, one entry per tensor dimension.
- ``elem_strides``: Stride, in elements, between consecutive accesses inside the shared memory tile. The second overload assumes a stride of ``1`` for every dimension.

*Optional parameters*:

- ``interleave_layout``: Interleaving applied to the underlying memory.
- ``swizzle``: Swizzle pattern matching the chosen interleave layout.
- ``l2_fetch_size``: L2 cache promotion for TMA transfers.
- ``oobfill``: Out-of-bounds fill policy for floating-point tensors.

Return value
------------

- A ``CUtensorMap`` encoding all metadata required to launch TMA transfers.

Preconditions
-------------

**General preconditions**:

* Compute Capability 9.0 or newer is required.
* ``dlpack/dlpack.h`` (DLPack v1) must be discoverable at compile time.

**DLPack preconditions**:

* ``tensor.device.device_type`` must be ``kDLCUDA`` or ``kDLCUDAManaged``.
* ``tensor.device.device_id`` must be a valid GPU device ordinal and the device must have Compute Capability 9.0 or newer.
* ``tensor.ndim`` (rank) must be greater than 0 and less than or equal to 5.

  - ``tensor.ndim`` must be greater than or equal to ``3`` when an interleaved layout is requested.

* The following data types are accepted for ``tensor.dtype``:

  - ``kDLUInt``:

    - ``bits == 4``, ``lanes == 16``, namely ``U4 x 16``. Additionally, the innermost dimension must be a multiple of ``2`` when only 16-byte alignment is available. Requires CUDA Toolkit 12.8 and later.
    - ``bits == 8``, ``lanes == 1``, namely ``uint8_t``.
    - ``bits == 16``, ``lanes == 1``, namely ``uint16_t``.
    - ``bits == 32``, ``lanes == 1``, namely ``uint32_t``.
    - ``bits == 64``, ``lanes == 1``, namely ``uint64_t``.

  - ``kDLInt``

    -  ``bits == 32``, ``lanes == 1``, namely ``int32_t``.
    - ``bits == 64``, ``lanes == 1``, namely ``int64_t``.

  - ``kDLFloat``

    - ``bits == 16``, ``lanes == 1``, namely ``__half``.
    - ``bits == 32``, ``lanes == 1``, namely ``float``.
    - ``bits == 64``, ``lanes == 1``, namely ``double``.

  - ``kDLBfloat``

    - ``bits == 16``, ``lanes == 1``, namely ``__nv_bfloat16``.

* ``tensor.data`` must be a valid GPU global address and aligned to at least 16 bytes; 32 bytes for ``tma_interleave_layout::bytes32``.

* ``tensor.shape`` must be greater than 0 and not exceed ``2^32`` elements per dimension.

* ``tensor.strides`` must be greater than 0 and not exceed ``2^40`` bytes per dimension.

  - The tensor mapping must be unique, namely ``tensor.strides[i]`` must be greater than or equal to ``tensor.shape[i - 1]`` or equal to ``0``.
  - ``tensor.strides[i]`` in bytes must be a multiple of the alignment (16 or 32 bytes) for the selected ``interleave_layout``.

**User parameter preconditions**:

* ``box_sizes``, ``elem_strides``, and ``tensor.ndim`` must have the same rank.

* ``box_sizes`` must be positive and not exceed ``256`` elements per dimension. The full size of ``box_sizes`` must fit in shared memory.

  - The inner dimension in bytes, computed as ``box_sizes[rank - 1] * sizeof(data_type)``, must be a multiple of 16 bytes if ``interleave_layout`` is ``tma_interleave_layout::none``.
  - Otherwise, the inner dimension in bytes must not exceed the byte-width of the selected ``swizzle`` pattern (``32``, ``64``, or ``128`` bytes).

* ``elem_strides`` must be positive and not exceed ``8`` elements per dimension.

* ``oobfill`` must be ``tma_oob_fill::none`` for integer data types.

* If ``interleave_layout`` is ``tma_interleave_layout::bytes32``, ``swizzle`` must be ``tma_swizzle::bytes32``.

References
----------

- `DLPack C API <https://dmlc.github.io/dlpack/latest/c_api.html>`__ documentation.
- `CUDA Tensor Memory Accelerator (TMA) <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays>`__ documentation.
- ``cuTensorMapEncodeTiled`` `CUDA driver API <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7>`__ documentation.

Example
-------

.. code:: cuda

    #include <cuda/tma>
    #include <cuda/std/span>
    #include <dlpack/dlpack.h>

    CUtensorMap create_2d_tile_descriptor(float* device_ptr) {
        // Define DLPack tensor descriptor, commonly provided externally by the user, library, or framework.
        constexpr int64_t shape_storage[2]   = {64, 64};
        constexpr int64_t strides_storage[2] = {1, 64};

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
        int box_sizes_storage[2]    = {16, 16};
        int elem_strides_storage[2] = {1, 1};

        return cuda::make_tma_descriptor(
            tensor,
            cuda::std::span{box_sizes_storage},
            cuda::std::span{elem_strides_storage});
    }
