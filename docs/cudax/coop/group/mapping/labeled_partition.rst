.. _cudax-coop-group-mapping-labeled-partition:

``cuda::experimental::coop::labeled_partition``
===============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename Fn>
  class labeled_partition : public /*group-mapping-interface-for*/<labeled_partition>
  {
    Fn /*fn_*/; // exposition-only

  public:
    __device__ explicit labeled_partition(Fn fn)
      : /*fn_*/(cuda::std::move(fn))
    {}
  };

  template <typename Fn>
  labeled_partition(Fn) -> labeled_partition<Fn>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::labeled_partition`` is a mapping that splits every group from the previous mapping result into subgroups of threads with equal labels. The label is obtained by invoking the ``/*fn_*/`` callable with the previous mapping result and must have an integral or enumeration type. The callable must be move constructible. Currently, this mapping can only be used to group threads within the warp level.

The returned mapping result is exhaustive if the previous mapping result was exhaustive. Threads excluded by a previous mapping remain excluded, and the callable is not invoked for them.

The returned mapping result is not always contiguous.

Examples
--------

.. code:: cuda

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Creates a mapping that groups threads by their previous rank modulo 3.
    cudax::coop::labeled_partition m{[](const auto& prev_mapping_result) {
      return prev_mapping_result.unit_rank() % 3;
    }};
  }
