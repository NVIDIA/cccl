libcudacxx_add_pretty_printer_test(
  NAME shared_resource
  SOURCES source.cu
  CASES
    # gersemi: off
    --case inspect_unique 1 shared_resource.unique unique_resource
    --case inspect_alias 1 shared_resource.alias aliased_resource
    --case inspect_before_copy 1 shared_resource.copy.before copy_source
    --case inspect_after_copy 1 shared_resource.copy.after.source copy_source
    --case inspect_after_copy 1 shared_resource.copy.after.target copy_target
    --case inspect_before_move 1 shared_resource.move.before move_source
    --case inspect_after_move 1 shared_resource.move.after.source move_source
    --case inspect_after_move 1 shared_resource.move.after.target move_target
    # gersemi: on
)
