.. _cccl-python-coop-gpu-dataclass:

gpu_dataclass and KernelTraits-Style Workflows
==============================================

``cuda.coop`` provides :func:`coop.gpu_dataclass` to bundle multiple primitives
and expose their temp-storage requirements as a single object. This mirrors the
common CUB pattern of defining a ``KernelTraits`` struct that collects block-
level primitives and shared-memory metadata.

KernelTraits example (Mamba-style)
----------------------------------

The simplified Mamba selective scan example uses a dataclass to collect several
block primitives and then wraps it with :func:`coop.gpu_dataclass`:

.. literalinclude:: ../../python/cuda_cccl/tests/coop/mamba_selective_scan_fwd.py
   :language: python
   :pyobject: KernelTraits

.. literalinclude:: ../../python/cuda_cccl/tests/coop/mamba_selective_scan_fwd.py
   :language: python
   :pyobject: make_kernel_traits

``gpu_dataclass`` adds:

* ``temp_storage_bytes_sum`` — sum of all primitive temp storage sizes.
* ``temp_storage_bytes_max`` — max size across primitives.
* ``temp_storage_alignment`` — max alignment across primitives.

You can then allocate a shared-memory buffer once and pass it into multiple
primitives inside the kernel.

This page is intentionally about host-side organization rather than raw kernel
mechanics. ``gpu_dataclass`` is most useful when you want one place to collect
primitive instances, temp-storage metadata, and kernel-traits-style constants
that several kernels or helper functions share.

Compared to ad-hoc host code, this gives you a named, typed container for:

* pre-created primitive instances,
* aggregate temp-storage requirements,
* specialization constants such as block size or items per thread,
* KernelTraits-like reuse patterns familiar from advanced CUB code.

Notes
-----

* ``gpu_dataclass(..., compute_temp_storage=True)`` (default) will bundle LTOIR
  for the contained primitives so size/alignment can be computed.
* For advanced pipelines, combine ``gpu_dataclass`` with explicit
  :class:`coop.TempStorage` and manual synchronization.
