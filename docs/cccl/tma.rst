.. _cccl-tma:

Tensor Memory Accelerator (TMA)
===============================

The Tensor Memory Accelerator (TMA) is a hardware feature available on Hopper (SM90) and newer GPUs
that enables efficient asynchronous memory copies of tensor data between global and (cluster) shared memory.
The use of TMA is required to reach SOL memory throughput for some workloads,
notable those where the necessary load/store vectorization, unrolling, or pipelining are limited by the register file or other factors.
CCCL offer several tools to help users leverage TMA in their applications.

In general, we recommend users to reach for high-level algorithms if they fit their problem.
Several algorithms, like ``cub::DeviceTransform``, ``cub::DeviceMerge``, ``cub::DeviceScan`` already use TMA internally today,
with many Thrust algorithms building on those.
And more algorithms will be added over time.
Relying on high level algorithms leaves the complexity of implementing and tuning TMA to CCCL team,
while providing users with safer interfaces, high productivity and SOL performance from the start.

If direct use of TMA is required to author new kernels, CCCL offers the following tools to help users get started,
from high-level to low-level:

 - ``cub::BlockLoadToShared`` coming soon :)
 - :ref:`cuda::memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`
 - :ref:`cuda::device::memcpy_async_tx <libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx>`
 - :ref:`cuda::ptx::cp_async_bulk* variants <libcudacxx-ptx-instructions>`

``cub::BlockLoadToShared`` and ``cuda::memcpy_async`` have fallback implementations for pre-Hopper GPUs,
using ``cp.async``/``LDGSTS`` on Ampere (SM80+) and ordinary loads/stores on older architectures.
Furthermore, they gracefully handle unaligned data and copying regions of arbitrary size.

The various ``cuda::ptx::cp_async_bulk*`` versions and ``cuda::device::memcpy_async_tx``
are thin wrappers of the corresponding PTX instructions
and provide no fallback path on older GPUs and also require the copied data to be aligned and sized appropriately.

Some further TMA-related utilities are provided by the :ref:`libcu++ extended API <libcudacxx-extended-api-tma>`.
