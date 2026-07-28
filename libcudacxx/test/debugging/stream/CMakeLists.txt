libcudacxx_add_pretty_printer_test(
  NAME stream
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_owning 1 stream.owning owning_stream
    --case inspect_ref 1 stream.ref stream_reference
    --case inspect_alias 1 stream.alias aliased_stream
    --case inspect_default 1 stream.default default_stream
    --case inspect_legacy 1 stream.legacy legacy_stream
    --case inspect_per_thread 1 stream.per_thread per_thread_stream
    --case inspect_invalid 1 stream.invalid invalid_stream
    --case inspect_moved_from 1 stream.moved_from moved_from_stream
    # gersemi: on
)
