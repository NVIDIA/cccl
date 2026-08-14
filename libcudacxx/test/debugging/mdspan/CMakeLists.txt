libcudacxx_add_pretty_printer_test(
  NAME mdspan
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_rank1_dynamic mdspan.rank1_dynamic values
    --case inspect_rank2_static mdspan.rank2_static values
    --case inspect_rank2_mixed mdspan.rank2_mixed values
    --case inspect_rank2_dynamic mdspan.rank2_dynamic values
    --case inspect_rank0 mdspan.rank0 values
    --case inspect_empty mdspan.empty values
    --case inspect_layout_right mdspan.layout_right values
    --case inspect_layout_left mdspan.layout_left values
    --case inspect_layout_stride mdspan.layout_stride values
    --case inspect_rank0_layout_stride mdspan.rank0_layout_stride values
    --case inspect_const_element mdspan.const_element values
    --case inspect_pointer_element mdspan.pointer_element values
    --case inspect_nested mdspan.nested values
    --case inspect_custom_accessor mdspan.custom_accessor values
    --case inspect_host_mdspan mdspan.host_mdspan values
    --case inspect_device_mdspan mdspan.device_mdspan values
    --case inspect_device_mdspan_real_memory mdspan.device_mdspan_real_memory values
    --case inspect_mdspan_device_backed mdspan.device_backed values
    --case inspect_mdspan_copy_failure mdspan.copy_failure values
    --case inspect_managed_mdspan mdspan.managed_mdspan values
    --case inspect_restrict_mdspan mdspan.restrict_mdspan values
    --case inspect_array mdspan.array values
    --case inspect_before_update mdspan.update.before values
    --case inspect_after_update mdspan.update.after values
    # gersemi: on
)
