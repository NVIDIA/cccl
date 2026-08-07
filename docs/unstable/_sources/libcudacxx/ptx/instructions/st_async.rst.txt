.. _libcudacxx-ptx-instructions-st-async:

st.async
========

-  PTX ISA:
   `st.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async>`__
-  Used in: :ref:`How to use st.async <libcudacxx-ptx-examples-st-async>`

**NOTE.** Alignment of ``addr`` must be a multiple of vector size. For
instance, the ``addr`` supplied to the ``v2.b32`` variant must be
aligned to ``2 x 4 = 8`` bytes.

.. include:: generated/st_async.rst
