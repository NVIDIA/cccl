.. _cccl-python-cooperative:

``cooperative``: Cooperative Algorithms
=======================================

The ``cuda.cccl.cooperative`` library provides cooperative algorithms that operate
at the level of blocks and warps. It is designed to be used within
`Numba CUDA kernels <https://numba.readthedocs.io/en/stable/cuda/kernels.html>`_.

Here's an example showing how to use the ``cuda.cccl.cooperative`` library to
perform block-level reduction within a Numba CUDA kernel.

.. literalinclude:: ../../python/cuda_cccl/tests/cooperative/examples/block/reduce.py
   :language: python
   :pyobject: custom_reduce_example
   :caption: Block-level reduction example. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/cooperative/examples/block/reduce.py>`__

Example Collections
-------------------

For complete runnable examples and more advanced usage patterns, see our
full collection of `examples <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/cooperative/examples>`_.

External API References
------------------------

- :ref:`cuda_cooperative-module`
