.. _cub-developer-guide-nvtx:

NVTX
=====

The `NVIDIA Tools Extension SDK (NVTX) <https://nvidia.github.io/NVTX/>`_ is a cross-platform API
for annotating source code to provide contextual information to developer tools.
All device-scope algorithms in CUB are annotated with NVTX ranges,
allowing their start and stop to be visualized in profilers
like `NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`_.
Only the public APIs available in the ``<cub/device/device_xxx.cuh>`` headers are annotated,
excluding direct calls to the dispatch layer.
NVTX annotations can be disabled by defining ``NVTX_DISABLE`` during compilation.
When CUB device algorithms are called on a stream subject to
`graph capture <https://developer.nvidia.com/blog/cuda-graphs/>`_,
the NVTX range is reported for the duration of capture (where no execution happens),
and not when a captured graph is executed later (the actual execution).
