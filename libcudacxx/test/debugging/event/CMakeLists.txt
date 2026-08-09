libcudacxx_add_pretty_printer_test(
  NAME event
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_ref 1 event.ref event_reference
    --case inspect_owning 1 event.owning owning_event
    --case inspect_timed 1 event.timed timed_event
    --case inspect_alias 1 event.alias aliased_event
    --case inspect_null_ref 1 event.null_ref null_reference
    --case inspect_no_init 1 event.no_init no_init_event
    --case inspect_timed_no_init 1 event.timed_no_init no_init_timed_event
    --case inspect_nested 1 event.nested nested_events
    --case inspect_before_update 1 event.update.before updated_event
    --case inspect_after_update 1 event.update.after updated_event
    # gersemi: on
)
