.. _cccl-runtime:

Runtime
=======

.. toctree::
   :hidden:
   :maxdepth: 1

   runtime/stream
   runtime/event
   runtime/algorithm
   runtime/device

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`devices <cccl-runtime-device-devices>`
     - A range of all available CUDA devices
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`device_ref <cccl-runtime-device-device-ref>`
     - A non-owning representation of a CUDA device
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`arch_traits <cccl-runtime-device-arch-traits>`
     - Per-architecture trait accessors
     - CCCL 3.1.0
     - CUDA 13.1


   * - :ref:`stream_ref <cccl-runtime-stream-stream-ref>`
     - A non-owning wrapper around a ``cudaStream_t``
     - CCCL 2.2.0
     - CUDA 12.3

   * - :ref:`stream <cccl-runtime-stream-stream>`
     - An owning wrapper around a ``cudaStream_t``
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`event_ref <cccl-runtime-event-event-ref>`
     - A non-owning wrapper around a ``cudaEvent_t``
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`event <cccl-runtime-event-event>`
     - An owning wrapper around a ``cudaEvent_t`` (timing disabled)
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`timed_event <cccl-runtime-event-timed-event>`
     - An owning wrapper around a ``cudaEvent_t`` with timing enabled and elapsed-time queries
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`copy_bytes <cccl-runtime-algorithm-copy_bytes>`
     - Byte-wise copy into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` sources and destinations
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`fill_bytes <cccl-runtime-algorithm-fill_bytes>`
     - Byte-wise fill into a ``cuda::stream_ref`` for ``cuda::std::span``/``cuda::std::mdspan`` destinations
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`Memory Resources <libcudacxx-extended-api-memory-resources>`
     - ``cuda::mr`` interfaces (resources, wrappers, properties) usable with streams
     - CCCL 2.2.0 (experimental), CCCL 3.1.0 (stable)
     - CUDA 12.3 (experimental), CUDA 13.1 (stable)
