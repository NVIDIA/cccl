.. _cuda_compute-module:

``cuda.compute`` API Reference
==============================

.. warning::
  ``cuda.compute`` is in public beta.
  The API is subject to change without notice.

Algorithms
----------

.. automodule:: cuda.compute.algorithms
  :members:
  :undoc-members:
  :imported-members:

Iterators
---------

.. automodule:: cuda.compute.iterators
  :members:
  :undoc-members:
  :imported-members:

Operators
---------

.. py:currentmodule:: cuda.compute.op

.. Unfortunately, we need to manually document the OpKind enum here because
.. the `._bindings` module, where OpKind is defined, is mocked out when building
.. docs. The mock out is needed to avoid the need for CUDA to be installed
.. at docs build time.

.. py:class:: OpKind

   Enumeration of operator kinds for CUDA parallel algorithms.

   This enum defines the types of operations that can be performed
   in parallel algorithms, including arithmetic, logical, and bitwise operations.

   .. py:attribute:: PLUS
   .. py:attribute:: MINUS
   .. py:attribute:: MULTIPLIES
   .. py:attribute:: DIVIDES
   .. py:attribute:: MODULUS
   .. py:attribute:: EQUAL_TO
   .. py:attribute:: NOT_EQUAL_TO
   .. py:attribute:: GREATER
   .. py:attribute:: LESS
   .. py:attribute:: GREATER_EQUAL
   .. py:attribute:: LESS_EQUAL
   .. py:attribute:: LOGICAL_AND
   .. py:attribute:: LOGICAL_OR
   .. py:attribute:: LOGICAL_NOT
   .. py:attribute:: BIT_AND
   .. py:attribute:: BIT_OR
   .. py:attribute:: BIT_XOR
   .. py:attribute:: BIT_NOT
   .. py:attribute:: IDENTITY
   .. py:attribute:: NEGATE
   .. py:attribute:: MINIMUM
   .. py:attribute:: MAXIMUM

.. autoclass:: cuda.compute.op.RawOp
   :members:
   :undoc-members:

Ahead-of-Time Compilation
-------------------------

The :func:`serialize <cuda.compute.algorithms.serialize>` and
:func:`deserialize <cuda.compute.algorithms.deserialize>` functions (listed under
`Algorithms`_ above) persist and restore built algorithms. To build ahead of time
for architectures other than the current device's—or with no GPU present—pass the
following dtype-only placeholders to a ``make_*`` factory in place of real arrays
and scalars:

.. py:class:: cuda.compute.ProxyArray(dtype)

   A dtype-only placeholder for a device array. Use in place of a real device
   array when calling a ``make_*`` factory to compile an algorithm without
   allocating GPU memory. Satisfies the ``DeviceArrayLike`` protocol; accessing
   its data pointer raises ``RuntimeError``. See
   :ref:`cuda.compute.ahead_of_time_compilation`.

.. py:class:: cuda.compute.ProxyValue(dtype)

   A dtype-only placeholder for a scalar or initial-value argument (such as
   ``h_init``). Use in place of a real numpy scalar or array to compile an
   algorithm without real data; accessing its data raises ``RuntimeError``. See
   :ref:`cuda.compute.ahead_of_time_compilation`.

Utilities
---------

.. automodule:: cuda.compute.struct
   :members:

Typing
------

.. automodule:: cuda.compute.typing
   :members:
