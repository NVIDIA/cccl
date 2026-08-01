libcudacxx_add_pretty_printer_test(
  NAME mdspan
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_rank1_dynamic 1 mdspan.rank1_dynamic rank1_dynamic
    --case inspect_rank2_static 1 mdspan.rank2_static rank2_static
    --case inspect_rank2_mixed 1 mdspan.rank2_mixed rank2_mixed
    --case inspect_rank2_dynamic 1 mdspan.rank2_dynamic rank2_dynamic
    --case inspect_rank0 1 mdspan.rank0 rank0
    --case inspect_empty 1 mdspan.empty empty_span
    --case inspect_layout_left 1 mdspan.layout_left layout_left_span
    --case inspect_layout_stride 1 mdspan.layout_stride layout_stride_span
    --case inspect_const_element 1 mdspan.const_element const_element
    --case inspect_pointer_element 1 mdspan.pointer_element pointer_element
    --case inspect_nested 1 mdspan.nested nested
    --case inspect_custom_accessor 1 mdspan.custom_accessor custom_accessor
    --case inspect_vector 1 mdspan.vector mdspan_vector
    --case inspect_before_update 1 mdspan.update.before mutable_values
    --case inspect_after_update 1 mdspan.update.after mutable_values
    # gersemi: on
)
