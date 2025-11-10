from gersemi.builtin_commands import builtin_commands

command_definitions = {
    # This just passes ARGN to execute_process, easy:
    "cccl_execute_non_fatal_process": builtin_commands["execute_process"],
}
