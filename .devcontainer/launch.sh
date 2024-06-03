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
    echo "  -c, --cuda               Specify the CUDA version. E.g., 12.2"
    echo "  -H, --host               Specify the host compiler. E.g., gcc12"
    echo "  -d, --docker             Launch the development environment in Docker directly without using VSCode."
    echo "  --gpus gpu-request       GPU devices to add to the container ('all' to pass all GPUs)."
    echo "  -e, --env list           Set additional container environment variables."
    echo "  -h, --help               Display this help message and exit."
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

    local OPTIONS=c:e:H:dh
    local LONG_OPTIONS=cuda:,env:,host:,gpus:,docker,help
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
            -e|--env)
                env_vars+=("$1" "$2")
                shift 2
                ;;
            -H|--host)
                host_compiler="$2"
                shift 2
                ;;
            --gpus)
                gpu_request="$2"
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

    no_empty_lines() {
        grep -v -e '^$' || [ "$?" == "1" ]
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
        remove_leading_char "$1" \
      | remove_trailing_char "$2"
    }

    read_docker_image_metadata() {
        docker inspect --type image --format '{{json .Config.Labels}}' "$DOCKER_IMAGE" \
      | sed -r 's/.*"devcontainer.metadata":[ ]*"(.*[]\)])",?.*/\1/g' \
      | sed -r 's/\\\"/"/g'
    }

    json_map_key_to_shell_syntax() {
        sed -r 's/^"(.*)":[ ]*?([^,]*),?$/\1=\2/g'
    }

    translate_local_envvars_to_shell_syntax() {
        sed -r 's/\$\{localEnv:([^\:]*):?(.*)\}/${\1:-\2}/g'
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

    transform_to_one_line() {
        tabs_to_spaces \
      | trim_leading_whitespace \
      | trim_trailing_whitespace \
      | inline_local_workspace_folder \
      | inline_container_workspace_folder \
      | inline_local_workspace_folder_basename \
      | translate_local_envvars_to_shell_syntax \
      | tr -s '\n' '\t'
    }

    json_string() {
        transform_to_one_line \
      | grep -Po "\"$1\":\s*\"(.*)\"" \
      | sed -r "s/.*\"$1\":[ ]*\"([^\"]*)\",?.*/\1/g" \
      | no_empty_lines
    }

    json_map() {
        transform_to_one_line \
      | grep -Po "\"$1\":\s*{(.*?)\s*?}[^\"]" \
      | strip_enclosing_chars '{' '}' \
      | tr -s '\t' '\n' \
      | json_map_key_to_shell_syntax \
      | no_empty_lines
    }

    json_array() {
        transform_to_one_line \
      | grep -Po "\"$1\":\s*\[(.*?)\s*?\][^\"]" \
      | strip_enclosing_chars '[' ']' \
      | tr -s '\t' '\n' \
      | sed -r 's/", "/"\n"/g' \
      | sed -r 's/^(.*"),$/\1/' \
      | no_empty_lines
    }

    # Read image
    local DOCKER_IMAGE="$(
        json_string "image" < "${path}/devcontainer.json"
    )"

    # Read workspaceFolder
    local WORKSPACE_FOLDER="$(
        json_string "workspaceFolder" < "${path}/devcontainer.json"
    )"

    # Read remoteUser
    local REMOTE_USER="$(
        json_string "remoteUser" < "${path}/devcontainer.json"
    )"

    if test -z "${REMOTE_USER:-}"; then
        REMOTE_USER="$(
            # Read remoteUser from image metadata
            read_docker_image_metadata \
          | json_string "remoteUser"
        )"
    fi

    # Read runArgs
    local -a RUN_ARGS="($(
        json_array "runArgs" < "${path}/devcontainer.json"
    ))"

    # Read initializeCommand
    local -a INITIALIZE_COMMAND="($(
        json_array "initializeCommand" < "${path}/devcontainer.json"
    ))"

    local -a ENV_VARS="($(
        json_map "containerEnv" < "${path}/devcontainer.json" \
      | sed -r 's/(.*)=(.*)/--env \1=\2/'
    ))";

    local -a MOUNTS="($(
        tee < "${path}/devcontainer.json"   \
            1>/dev/null                     \
            >(json_array "mounts")          \
            >(json_string "workspaceMount") \
      | xargs -r -I% echo --mount '%'
    ))"

    # Update run args and env vars

    for flag in rm init; do
        if [[ " ${RUN_ARGS[*]} " != *" --${flag} "* ]]; then
            RUN_ARGS+=("--${flag}")
        fi
    done

    if [[ " ${RUN_ARGS[*]} " != *" --pull"* ]]; then
        RUN_ARGS+=(--pull always)
    fi

    if test -n "${gpu_request:-}"; then
        RUN_ARGS+=(--gpus "${gpu_request}")
    else
        # Read hostRequirements.gpu
        local GPU_REQUEST="$(
            json_map "hostRequirements" < "${path}/devcontainer.json" \
          | grep 'gpu=' \
          | sed -r 's/(.*)=(.*)/\2/' \
          | xargs
        )"
        if test "${GPU_REQUEST:-false}" = true; then
            RUN_ARGS+=(--gpus all)
        elif test "${GPU_REQUEST:-false}" = optional && \
             command -v nvidia-container-runtime >/dev/null 2>&1; then
            RUN_ARGS+=(--gpus all)
        fi
    fi

    RUN_ARGS+=(--workdir "${WORKSPACE_FOLDER:-/home/coder/cccl}")

    if test -n "${REMOTE_USER:-}"; then
        ENV_VARS+=(--env NEW_UID="$(id -u)")
        ENV_VARS+=(--env NEW_GID="$(id -g)")
        ENV_VARS+=(--env REMOTE_USER="$REMOTE_USER")
        RUN_ARGS+=(-u root:root)
        RUN_ARGS+=(--entrypoint "${WORKSPACE_FOLDER:-/home/coder/cccl}/.devcontainer/docker-entrypoint.sh")
    fi

    if test -n "${SSH_AUTH_SOCK:-}"; then
        ENV_VARS+=(--env "SSH_AUTH_SOCK=/tmp/ssh-auth-sock")
        MOUNTS+=(--mount "source=${SSH_AUTH_SOCK},target=/tmp/ssh-auth-sock,type=bind")
    fi

    if test -v env_vars && test ${#env_vars[@]} -gt 0; then
        ENV_VARS+=("${env_vars[@]}")
    fi

    if test "${#INITIALIZE_COMMAND[@]}" -gt 0; then
        eval "${INITIALIZE_COMMAND[*]@Q}"
    fi

    exec docker run -it \
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
