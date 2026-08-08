libcudacxx_add_pretty_printer_test(
  NAME span
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_static 1 span.static static_extent
    --case inspect_dynamic 1 span.dynamic dynamic
    --case inspect_static_empty 1 span.static_empty static_empty
    --case inspect_default 1 span.default default_constructed
    --case inspect_const 1 span.const const_elements
    --case inspect_nested 1 span.nested nested
    --case inspect_alias 1 span.alias alias
    --case inspect_before_update 1 span.update.before updated
    --case inspect_after_update 1 span.update.after updated
    # gersemi: on
)
