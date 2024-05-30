#!/usr/bin/env bash

set -euo pipefail

# Ensure the script is being executed in the cccl/ root
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/..";

print_help() {
    echo "Usage: $0 [-c|--cuda <CUDA version>] [-H|--host <Host compiler>] [-d|--docker]"
    echo "Launch a development container. If no CUDA version or Host compiler are specified,"
    echo "the top-level devcontainer in .devcontainer/devcontainer.json will be used."
    echo ""
    echo "Options:"
    echo "  -c, --cuda       Specify the CUDA version. E.g., 12.2"
    echo "  -H, --host       Specify the host compiler. E.g., gcc12"
    echo "  -d, --docker     Launch the development environment in Docker directly without using VSCode."
    echo "  -h, --help       Display this help message and exit."
}

_upvar() {
    if unset -v "$1"; then
        if (( $# == 2 )); then
            eval $1=\"\$2\";
        else
            eval $1=\(\"\${@:2}\"\);
        fi;
    fi
}

parse_options() {
    local -;
    set -euo pipefail;

    local UNPARSED="${!#}";
    set -- "${@:1:$#-1}";

    local OPTIONS=c:H:dh
    local LONG_OPTIONS=cuda:,host:,docker,help
    # shellcheck disable=SC2155
    local PARSED_OPTIONS="$(getopt -n "$0" -o "${OPTIONS}" --long "${LONG_OPTIONS}" -- "$@")"

    # shellcheck disable=SC2181
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    eval set -- "${PARSED_OPTIONS}"

    while true; do
        case "$1" in
            -c|--cuda)
                cuda_version="$2"
                shift 2
                ;;
            -H|--host)
                host_compiler="$2"
                shift 2
                ;;
            -d|--docker)
                docker_mode=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            --)
                shift
                _upvar "${UNPARSED}" "${@}"
                break
                ;;
            *)
                echo "Invalid option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

# shellcheck disable=SC2155
launch_docker() {
    local -;
    set -euo pipefail;

    tabs_to_spaces() {
        sed $'s/\t/ /g'
    }

    trim_leading_whitespace() {
        sed 's/^[[:space:]]*//'
    }

    trim_trailing_whitespace() {
        sed 's/[[:space:]]*$//'
    }

    remove_leading_char() {
        cut -d"$1" -f1 --complement
    }

    remove_trailing_char() {
        rev | cut -d"$1" -f1 --complement | rev
    }

    strip_enclosing_chars() {
        tabs_to_spaces \
      | trim_leading_whitespace \
      | trim_trailing_whitespace \
      | tr -s '\n' '\t' \
      | remove_leading_char "$1" \
      | remove_trailing_char "$2" \
      | tr -s '\t' '\n'
    }

    json_map_key_to_shell_syntax() {
        sed -r 's/"(.*)":[ ]*?(.*)/\1=\2/' \
      | remove_trailing_char ','
    }

    translate_local_envvars_to_shell_syntax() {
        sed -r 's/\$\{localEnv:([^\:]*):?(.*)\}/${\1:-\2}/'
    }

    inline_local_workspace_folder() {
        sed "s@\${localWorkspaceFolder}@$(pwd)@g"
    }

    inline_container_workspace_folder() {
        sed "s@\${containerWorkspaceFolder}@${WORKSPACE_FOLDER:-}@g"
    }

    inline_local_workspace_folder_basename() {
        sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g"
    }

    json_string() {
        grep "\"$1\":" \
      | sed -r 's/.*:[ ]*?"(.*)",/\1/' \
      | inline_local_workspace_folder \
      | inline_container_workspace_folder \
      | inline_local_workspace_folder_basename \
      | translate_local_envvars_to_shell_syntax
    }

    json_map() {
        grep -Pzo "(?s)\"$1\": {(.*?)\s+}" \
      | strip_enclosing_chars '{' '}' \
      | json_map_key_to_shell_syntax \
      | inline_local_workspace_folder \
      | inline_container_workspace_folder \
      | inline_local_workspace_folder_basename \
      | translate_local_envvars_to_shell_syntax
    }

    json_array() {
        grep -Pzo "(?s)\"$1\": \[(.*?)\s*\]" \
      | strip_enclosing_chars '[' ']' \
      | sed -r 's/", "/"\n"/g' \
      | sed -r 's/^(.*"),$/\1/' \
      | inline_local_workspace_folder \
      | inline_container_workspace_folder \
      | inline_local_workspace_folder_basename \
      | translate_local_envvars_to_shell_syntax
    }

    # Read workspaceFolder
    local WORKSPACE_FOLDER="$(
        json_string "workspaceFolder" < "${path}/devcontainer.json"
    )"

    # Read image
    local DOCKER_IMAGE="$(
        json_string "image" < "${path}/devcontainer.json"
    )"

    # Read runArgs
    local -a RUN_ARGS="($(
        json_array "runArgs" < "${path}/devcontainer.json"
    ))"

    for flag in rm init; do
        if [[ " ${RUN_ARGS[*]} " != *" --${flag} "* ]]; then
            RUN_ARGS+=("--${flag}")
        fi
    done

    # Read initializeCommand
    local -a INITIALIZE_COMMAND="($(
        json_array "initializeCommand" < "${path}/devcontainer.json"
    ))"

    local -a ENV_VARS="($(
        json_map "containerEnv" < "${path}/devcontainer.json" \
      | sed -r 's/(.*)=(.*)/--env \1=\2/'
    ))";

    ENV_VARS+=(--env REMOTE_USER=coder)
    ENV_VARS+=(--env NEW_UID="$(id -u)")
    ENV_VARS+=(--env NEW_GID="$(id -g)")

    local -a MOUNTS="($(
        tee < "${path}/devcontainer.json"   \
            1>/dev/null                     \
            >(json_array "mounts")          \
            >(json_string "workspaceMount") \
      | xargs -r -I% echo --mount '%'
    ))"

    if test -n "${SSH_AUTH_SOCK:-}"; then
        ENV_VARS+=(--env "SSH_AUTH_SOCK=/tmp/ssh-auth-sock")
        MOUNTS+=(--mount "source=${SSH_AUTH_SOCK},target=/tmp/ssh-auth-sock,type=bind")
    fi

    if test "${#INITIALIZE_COMMAND[@]}" -gt 0; then
        eval "${INITIALIZE_COMMAND[*]@Q}"
    fi

    exec docker run \
        -it \
        -u root:root \
        --pull always \
        --workdir "${WORKSPACE_FOLDER:-/home/coder/cccl}" \
        --entrypoint /home/coder/cccl/.devcontainer/docker-entrypoint.sh \
        "${RUN_ARGS[@]}" \
        "${ENV_VARS[@]}" \
        "${MOUNTS[@]}" \
        "${DOCKER_IMAGE}" \
        "$@"
}

launch_vscode() {
    local -;
    set -euo pipefail;
    # Since Visual Studio Code allows only one instance per `devcontainer.json`,
    # this code prepares a unique temporary directory structure for each launch of a devcontainer.
    # By doing so, it ensures that multiple instances of the same environment can be run
    # simultaneously. The script replicates the `devcontainer.json` from the desired CUDA
    # and compiler environment into this temporary directory, adjusting paths to ensure the
    # correct workspace is loaded. A special URL is then generated to instruct VSCode to
    # launch the development container using this temporary configuration.
    local workspace="$(basename "$(pwd)")"
    local tmpdir="$(mktemp -d)/${workspace}"
    mkdir -p "${tmpdir}"
    mkdir -p "${tmpdir}/.devcontainer"
    cp -arL "${path}/devcontainer.json" "${tmpdir}/.devcontainer"
    sed -i "s@\${localWorkspaceFolder}@$(pwd)@g" "${tmpdir}/.devcontainer/devcontainer.json"
    local path="${tmpdir}"
    local hash="$(echo -n "${path}" | xxd -pu - | tr -d '[:space:]')"
    local url="vscode://vscode-remote/dev-container+${hash}/home/coder/cccl"

    local launch=""
    if type open >/dev/null 2>&1; then
        launch="open"
    elif type xdg-open >/dev/null 2>&1; then
        launch="xdg-open"
    fi

    if [ -n "${launch}" ]; then
        echo "Launching VSCode Dev Container URL: ${url}"
        code --new-window "${tmpdir}"
        exec "${launch}" "${url}" >/dev/null 2>&1
    fi
}

main() {
    local -a unparsed;
    parse_options "$@" unparsed;
    set -- "${unparsed[@]}";

    # If no CTK/Host compiler are provided, just use the default environment
    if [[ -z ${cuda_version:-} ]] && [[ -z ${host_compiler:-} ]]; then
        path=".devcontainer"
    else
        path=".devcontainer/cuda${cuda_version}-${host_compiler}"
        if [[ ! -f "${path}/devcontainer.json" ]]; then
            echo "Unknown CUDA [${cuda_version}] compiler [${host_compiler}] combination"
            echo "Requested devcontainer ${path}/devcontainer.json does not exist"
            exit 1
        fi
    fi

    if ${docker_mode:-'false'}; then
        launch_docker "$@"
    else
        launch_vscode
    fi
}

main "$@"
