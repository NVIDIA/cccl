.. _cccl-runtime-device:

Devices
=======

``cuda::device_ref``
---------------------
.. _cccl-runtime-device-device-ref:

``cuda::device_ref`` is a lightweight, non-owning handle to a CUDA device ordinal. It allows to query information about a device and serves as an argument to other runtime APIs which are tied to a specific device.
It offers:

- ``get()``: native device ordinal
- ``name()``: device name
- ``init()``: initialize the device context
- ``peers()``: list peers for which peer access can be enabled
- ``has_peer_access_to(device_ref)``: query if peer access can be enabled to the given device
- ``attribute(attr)`` / ``attribute<::cudaDeviceAttr>()``: attribute queries

Availability: CCCL 3.1.0 / CUDA 13.1

``cuda::devices``
------------------
.. _cccl-runtime-device-devices:

``cuda::devices`` is a random-access view of all available CUDA devices in the form of ``cuda::device_ref`` objects. It
provides indexing, size, and iteration for use
in range-based loops.

Availability: CCCL 3.1.0 / CUDA 13.1

Example:

.. code:: cpp

   #include <cuda/devices>
   #include <iostream>

   void print_devices() {
     for (auto& dev : cuda::devices) {
       std::cout << "Device " << dev.get() << ": " << dev.name() << std::endl;
     }
   }

Device attributes
-----------------
.. _cccl-runtime-device-attributes:

``cuda::device_attributes`` provides strongly-typed attribute query objects usable with
``device_ref::attribute``. Selected examples:

- ``compute_capability``
- ``multiprocessor_count``
- ``concurrent_managed_access``
- ``clock_rate``
- ``numa_id``

Availability: CCCL 3.1.0 / CUDA 13.1

Example:

.. code:: cpp

   #include <cuda/devices>

   int get_max_blocks_on_device(cuda::device_ref dev) {
     return cuda::device_attributes::multiprocessor_count(dev) * cuda::device_attributes::blocks_per_multiprocessor(dev);
   }

``cuda::arch_traits``
---------------------
.. _cccl-runtime-device-arch-traits:

Per-architecture trait accessors providing limits and capabilities common to all devices of an architecture.
Compared to ``device_attributes``, ``cuda::arch_traits`` provide a compile-time accessible structure that describes common characteristics of all devices of an architecture, while attributes are run-time queries of a single characteristic of a specific device.

- ``cuda::arch_traits<cuda::arch_id::sm_80>()`` (compile-time) or
  ``cuda::arch_traits_for(cuda::arch_id)`` / ``cuda::arch_traits_for(cuda::compute_capability)`` (run-time).
- Returns a ``cuda::arch_traits_t`` with fields like
  ``max_threads_per_block``, ``max_shared_memory_per_block``, ``cluster_supported`` and other capability flags.
- Traits for the current architecture can be accessed with ``cuda::device::current_arch_traits()``

Availability: CCCL 3.1.0 / CUDA 13.1

Example:

.. code:: cpp

   #include <cuda/devices>

   template <cuda::arch_id Arch>
   __device__ void fn() {
     auto traits = cuda::arch_traits<Arch>();
     if constexpr (traits.cluster_supported) {
       // cluster specific code
     } else {
       // non-cluster code
     }
   }

   __global__ void kernel() {
     fn<cuda::arch_id::sm_90>();
   }
