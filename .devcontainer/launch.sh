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
    echo "  -v, --volume list        Bind mount a volume."
    echo "  -h, --help               Display this help message and exit."
}

# Assign variable one scope above the caller
# Usage: local "$1" && _upvar $1 "value(s)"
# Param: $1  Variable name to assign value to
# Param: $*  Value(s) to assign.  If multiple values, an array is
#            assigned, otherwise a single value is assigned.
# See: http://fvue.nl/wiki/Bash:_Passing_variables_by_reference
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

    # Read the name of the variable in which to return unparsed arguments
    local UNPARSED="${!#}";
    # Splice the unparsed arguments variable name from the arguments list
    set -- "${@:1:$#-1}";

    local OPTIONS=c:e:H:dhv:
    local LONG_OPTIONS=cuda:,env:,host:,gpus:,volume:,docker,help
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
            -v|--volume)
                volumes+=("$1" "$2")
                shift 2
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
    set -euo pipefail

    inline_vars() {
        cat - \
        `# inline local workspace folder` \
      | sed "s@\${localWorkspaceFolder}@$(pwd)@g" \
        `# inline local workspace folder basename` \
      | sed "s@\${localWorkspaceFolderBasename}@$(basename "$(pwd)")@g" \
        `# inline container workspace folder` \
      | sed "s@\${containerWorkspaceFolder}@${WORKSPACE_FOLDER:-}@g" \
        `# inline container workspace folder basename` \
      | sed "s@\${containerWorkspaceFolderBasename}@$(basename "${WORKSPACE_FOLDER:-}")@g" \
        `# translate local envvars to shell syntax` \
      | sed -r 's/\$\{localEnv:([^\:]*):?(.*)\}/${\1:-\2}/g'
    }

    args_to_path() {
        local -a keys=("${@}")
        keys=("${keys[@]/#/[}")
        keys=("${keys[@]/%/]}")
        echo "$(IFS=; echo "${keys[*]}")"
    }

    json_string() {
        python3 -c "import json,sys; print(json.load(sys.stdin)$(args_to_path "${@}"))" 2>/dev/null | inline_vars
    }

    json_array() {
        python3 -c "import json,sys; [print(f'\"{x}\"') for x in json.load(sys.stdin)$(args_to_path "${@}")]" 2>/dev/null | inline_vars
    }

    json_map() {
        python3 -c "import json,sys; [print(f'{k}=\"{v}\"') for k,v in json.load(sys.stdin)$(args_to_path "${@}").items()]" 2>/dev/null | inline_vars
    }

    devcontainer_metadata_json() {
        docker inspect --type image --format '{{json .Config.Labels}}' "$DOCKER_IMAGE" \
      | json_string '"devcontainer.metadata"'
    }

    ###
    # Read relevant values from devcontainer.json
    ###

    local devcontainer_json="${path}/devcontainer.json";

    # Read image
    local DOCKER_IMAGE="$(json_string '"image"' < "${devcontainer_json}")"
    # Always pull the latest copy of the image
    docker pull "$DOCKER_IMAGE"

    # Read workspaceFolder
    local WORKSPACE_FOLDER="$(json_string '"workspaceFolder"' < "${devcontainer_json}")"
    # Read remoteUser
    local REMOTE_USER="$(json_string '"remoteUser"' < "${devcontainer_json}")"
    # If remoteUser isn't in our devcontainer.json, read it from the image's "devcontainer.metadata" label
    if test -z "${REMOTE_USER:-}"; then
        REMOTE_USER="$(devcontainer_metadata_json | json_string "-1" '"remoteUser"')"
    fi
    # Read runArgs
    local -a RUN_ARGS="($(json_array '"runArgs"' < "${devcontainer_json}"))"
    # Read initializeCommand
    local -a INITIALIZE_COMMAND="($(json_array '"initializeCommand"' < "${devcontainer_json}"))"
    # Read containerEnv
    local -a ENV_VARS="($(json_map '"containerEnv"' < "${devcontainer_json}" | sed -r 's/(.*)=(.*)/--env \1=\2/'))"
    # Read mounts
    local -a MOUNTS="($(
        tee < "${devcontainer_json}"          \
            1>/dev/null                       \
            >(json_array '"mounts"')          \
            >(json_string '"workspaceMount"') \
      | xargs -r -I% echo --mount '%'
    ))"

    ###
    # Update run arguments and container environment variables
    ###

    # Only pass `-it` if the shell is a tty
    if ! ${CI:-'false'} && tty >/dev/null 2>&1 && (exec </dev/tty); then
        RUN_ARGS+=("-it")
    fi

    for flag in rm init; do
        if [[ " ${RUN_ARGS[*]} " != *" --${flag} "* ]]; then
            RUN_ARGS+=("--${flag}")
        fi
    done

    # Prefer the user-provided --gpus argument
    if test -n "${gpu_request:-}"; then
        RUN_ARGS+=(--gpus "${gpu_request}")
    else
        # Otherwise read and infer from hostRequirements.gpu
        local GPU_REQUEST="$(json_string '"hostRequirements"' '"gpu"' < "${devcontainer_json}")"
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

    if test -n "${SSH_AUTH_SOCK:-}" && test -e "${SSH_AUTH_SOCK:-}"; then
        ENV_VARS+=(--env "SSH_AUTH_SOCK=/tmp/ssh-auth-sock")
        MOUNTS+=(--mount "source=${SSH_AUTH_SOCK},target=/tmp/ssh-auth-sock,type=bind")
    fi

    # Append user-provided volumes
    if test -v volumes && test ${#volumes[@]} -gt 0; then
        MOUNTS+=("${volumes[@]}")
    fi

    # Append user-provided envvars
    if test -v env_vars && test ${#env_vars[@]} -gt 0; then
        ENV_VARS+=("${env_vars[@]}")
    fi

    # Run the initialize command before starting the container
    if test "${#INITIALIZE_COMMAND[@]}" -gt 0; then
        eval "${INITIALIZE_COMMAND[*]@Q}"
    fi

    exec docker run \
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
