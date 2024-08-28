.. _cudax-memory-resource:

Memory Resources
=================

.. toctree::
   :glob:
   :maxdepth: 3

   ${repo_docs_api_path}/*any__resource*

The ``<cuda/experimental/memory_resource.cuh>`` header provides:
   -  :ref:`any_resource <cudax-memory-resource-any-resource>` and
      :ref:`async_any_resource <cudax-memory-resource-async-any-resource>` type erased memory resources similar to
      ``std::any``. In contrast to :ref:`resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` they
      own the contained resource.
