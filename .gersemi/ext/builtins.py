from gersemi.builtin_commands import builtin_commands

# TODO override max_inline_items for these commands when/if
# https://github.com/BlankSpruce/gersemi/issues/78 is implemented.
# I want these to expand anytime there's more than one source / lib / def / option / etc.
# - add_library
# - add_executable
# - target_link_libraries
# - target_compile_definitions
# - target_compile_options
# - target_include_directories
#
# I want these to always inline unless they exceed line length:
# - cmake_parse_arguments
# - function
# - macro

# Patch up https://github.com/BlankSpruce/gersemi/pull/80 if needed:
mod_find_package = builtin_commands["find_package"]
if "REQUIRED" in mod_find_package["multi_value_keywords"]:
    mod_find_package["multi_value_keywords"].remove("REQUIRED")
    mod_find_package["options"] += ["REQUIRED"]

command_definitions = {
    "if ": builtin_commands["if"],
    "elseif ": builtin_commands["elseif"],
    "foreach ": builtin_commands["foreach"],
    "find_package": mod_find_package,
}
