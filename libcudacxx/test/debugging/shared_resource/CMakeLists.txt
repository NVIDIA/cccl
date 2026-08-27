libcudacxx_add_pretty_printer_test(
  NAME shared_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_unique shared_resource.unique resource
    --case inspect_alias shared_resource.alias resource
    --case inspect_before_copy shared_resource.copy.before resource
    --case inspect_copy_source shared_resource.copy.after.source resource
    --case inspect_copy_target shared_resource.copy.after.target resource
    --case inspect_before_move shared_resource.move.before resource
    --case inspect_move_source shared_resource.move.after.source resource
    --case inspect_move_target shared_resource.move.after.target resource
    # gersemi: on
)
