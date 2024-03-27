# Print "ARG=${ARG}" for all args.
function print_var_values() {
    # Iterate through the arguments
    for var_name in "$@"; do
        if [ -z "$var_name" ]; then
            echo "Usage: print_var_values <variable_name1> <variable_name2> ..."
            return 1
        fi

        # Dereference the variable and print the result
        echo "$var_name=${!var_name:-(undefined)}"
    done
}

# begin_group: Start a named section of log output, possibly with color.
# Usage: begin_group "Group Name" [Color]
#   Group Name: A string specifying the name of the group.
#   Color (optional): ANSI color code to set text color. Default is blue (1;34).
function begin_group() {
    # See options for colors here: https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
    local blue="34"
    local name="${1:-}"
    local color="${2:-$blue}"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo -e "::group::\e[${color}m${name}\e[0m"
    else
        echo -e "\e[${color}m================== ${name} ======================\e[0m"
    fi
}

# end_group: End a named section of log output and print status based on exit status.
# Usage: end_group "Group Name" [Exit Status]
#   Group Name: A string specifying the name of the group.
#   Exit Status (optional): The exit status of the command run within the group. Default is 0.
function end_group() {
    local name="${1:-}"
    local build_status="${2:-0}"
    local duration="${3:-}"
    local red="31"
    local blue="34"

    if [ -n "${GITHUB_ACTIONS:-}" ]; then
        echo "::endgroup::"

        if [ "$build_status" -ne 0 ]; then
            echo -e "::error::\e[${red}m ${name} - Failed (⬆️ click above for full log ⬆️)\e[0m"
        fi
    else
        if [ "$build_status" -ne 0 ]; then
            echo -e "\e[${red}m================== End ${name} - Failed${duration:+ - Duration: ${duration}s} ==================\e[0m"
        else
            echo -e "\e[${blue}m================== End ${name} - Success${duration:+ - Duration: ${duration}s} ==================\n\e[0m"
        fi
    fi
}

declare -A command_durations

# Runs a command within a named group, handles the exit status, and prints appropriate messages based on the result.
# Usage: run_command "Group Name" command [arguments...]
function run_command() {
    local group_name="${1:-}"
    shift
    local command=("$@")
    local status

    begin_group "$group_name"
    set +e
    local start_time=$(date +%s)
    "${command[@]}"
    status=$?
    local end_time=$(date +%s)
    set -e
    local duration=$((end_time - start_time))
    end_group "$group_name" $status $duration
    command_durations["$group_name"]=$duration
    return $status
}

function string_width() {
    local str="$1"
    echo "$str" | awk '{print length}'
}

function print_time_summary() {
    local max_length=0
    local group

    # Find the longest group name for formatting
    for group in "${!command_durations[@]}"; do
        local group_length=$(echo "$group" | awk '{print length}')
        if [ "$group_length" -gt "$max_length" ]; then
            max_length=$group_length
        fi
    done

    echo "Time Summary:"
    for group in "${!command_durations[@]}"; do
        printf "%-${max_length}s : %s seconds\n" "$group" "${command_durations[$group]}"
    done

    # Clear the array of timing info
    declare -gA command_durations=()
}
