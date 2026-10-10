libcudacxx_add_pretty_printer_test(
  NAME event
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_ref event.ref value
    --case inspect_owning event.owning value
    --case inspect_timed event.timed value
    --case inspect_alias event.alias value
    --case inspect_null_ref event.null_ref value
    --case inspect_no_init event.no_init value
    --case inspect_timed_no_init event.timed_no_init value
    --case inspect_nested event.nested values
    --case inspect_before_update event.update.before value
    --case inspect_after_update event.update.after value
    # gersemi: on
)
