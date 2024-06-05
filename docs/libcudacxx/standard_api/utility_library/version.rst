.. _libcudacxx-standard-api-utility-version:

<cuda/std/version>
======================

See the documentation of the standard header `\<version\> <https://en.cppreference.com/w/cpp/header/version>`_

Extensions
----------

The following version macros, which are explained in the :ref:`versioning section <libcudacxx-releases-versioning>`,
are defined in this header:

-  ``_LIBCUDACXX_CUDA_API_VERSION``
-  ``_LIBCUDACXX_CUDA_API_VERSION_MAJOR``
-  ``_LIBCUDACXX_CUDA_API_VERSION_MINOR``
-  ``_LIBCUDACXX_CUDA_API_VERSION_PATCH``
-  ``_LIBCUDACXX_CUDA_ABI_VERSION``
-  ``_LIBCUDACXX_CUDA_ABI_VERSION_LATEST``

Restrictions
------------

When using NVCC, the definition of C++ feature test macros is provided
by the host Standard Library, not libcu++.
