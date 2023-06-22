#!/usr/bin/env bash

set -e

nvrtcdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
libcudacxxdir="$(cd "${nvrtcdir}/../../.." && pwd)"

logdir=$(mktemp --tmpdir=${XDG_RUNTIME_DIR} -d libcudacxx.build.XXXXXXXXXX)

nvcc=$(echo $1 | sed 's/^[[:space:]]*//')
shift

echo "${nvcc}" >> ${logdir}/log

original_flags=${@}
echo "original flags: ${original_flags[@]}" >> ${logdir}/log

original_flags=("${original_flags[@]}" -D__LIBCUDACXX_NVRTC_TEST__=1)

declare -a modified_flags
declare -a gpu_archs
declare -a includes

input=""
input_type=""
compile=0

while [[ $# -ne 0 ]]
do
    case "$1" in
        -E)
            "${nvcc}" ${original_flags[@]} 2>>${logdir}/error_log
            exit $?
            ;;

        -c)
            compile=1
            ;;

        -I*)
            modified_flags=("${modified_flags[@]}" "$1")
            includes=("${includes[@]}" "$1")
            ;;

        -I)
            modified_flags=("${modified_flags[@]}" "$1" "$2")
            includes=("${includes[@]}" "$1" "$2")
            shift
            ;;

        -include|-isystem|-o|-ccbin)
            modified_flags=("${modified_flags[@]}" "$1" "$2")
            shift
            ;;

        -x)
            input_type="-x $2"
            shift
            ;;

        -gencode=*)
            # head -n1 to handle "compute_70,compute_70"
            gpu_archs=("${gpu_archs[@]}" "$(echo $1 | egrep -o 'compute_[0-9]+' | egrep -o '[0-9]+' | head -n1)")
            modified_flags=("${modified_flags[@]}" "$1")
            ;;

        -?*|\"-?*)
            modified_flags=("${modified_flags[@]}" "$1")
            ;;

        all-warnings)
            modified_flags=("${modified_flags[@]}" "$1")
            ;;

        *)
            if [[ "${input}" != "" ]]
            then
                echo "spurious argument interpreted as positional: ${1}" >> ${logdir}/log
                echo "in: ${original_flags[@]}" >> ${logdir}/log
                exit 1
            fi
            input="$1"

            ;;
    esac

    shift
done

echo "${includes[@]}"

if [[ $compile -eq 0 ]] || [[ "${input_type}" != "-x cu" ]]
then
    "${nvcc}" ${original_flags[@]} -lnvrtc -lcuda 2> >(tee -a ${logdir}/error_log)
    exit $?
fi

cudart_include_dir=$(
    echo '#include <cuda_pipeline_primitives.h>' \
        | ${nvcc} -x cu - -M -E "${includes[@]}" \
        | grep -e ' /.*/cuda_pipeline_primitives\.h' -o \
        | xargs dirname)
ext_include_dir=$(
    echo '#include <cuda/pipeline>' \
        | ${nvcc} -x cu - -M -E "${includes[@]}" -arch sm_70 -std=c++11 \
        | grep -e ' /.*/cuda/pipeline' -o \
        | xargs dirname | xargs dirname)

echo "detected input file: ${input}" >> ${logdir}/log
echo "modified flags: ${modified_flags[@]}" >> ${logdir}/log

tempfile=$(mktemp --tmpdir -t XXXXXXXXX.cu)

finish() {
    if [[ "${FAUX_NVRTC_KEEP_TMP}" == "YES" ]]
    then
        echo "${tempfile}" >> ${logdir}/tmp_log
    else
        rm "${tempfile}"
    fi
}
trap finish EXIT

thread_count=$(cat "${input}" | egrep 'cuda_thread_count = [0-9]+' | egrep -o '[0-9]+' || echo 1)
shmem_size=$(cat "${input}" | egrep 'cuda_block_shmem_size = [0-9]+' | egrep -o '[0-9]+' || echo 0)

gpu_archs=($(printf "%s\n" "${gpu_archs[@]}" | sort -un | tr '\n' ' '))

cat "${nvrtcdir}/head.cu.in" >> "${tempfile}"
cat "${input}" >> "${tempfile}"
cat "${nvrtcdir}/middle.cu.in" >> "${tempfile}"
echo '        // BEGIN SCRIPT GENERATED OPTIONS' >> "${tempfile}"
echo '        "-I'"${libcudacxxdir}/include"'",' >> "${tempfile}"
echo '        "-I'"${libcudacxxdir}/test/support"'",' >> "${tempfile}"
echo '        "-I'"${cudart_include_dir}"'",' >> "${tempfile}"
echo '        "-I'"${ext_include_dir}"'",' >> "${tempfile}"
echo '        "--pre-include='"${libcudacxxdir}/test/support/nvrtc_limit_macros.h"'",' >> "${tempfile}"
echo '        "--device-int128",' >> "${tempfile}"
# The line below intentionally only uses the first element of ${gpu_archs[@]}.
# They are sorted numerically above, and this selects the lowest of the requested
# values.
echo '        "--gpu-architecture='compute_"${gpu_archs}"'",' >> "${tempfile}"
echo '        // END SCRIPT GENERATED OPTIONS' >> "${tempfile}"
cat "${nvrtcdir}/tail.cu.in" >> "${tempfile}"
echo '        '"${thread_count}, 1, 1," >> "${tempfile}"
echo '        '"${shmem_size}," >> "${tempfile}"
cat "${nvrtcdir}/post_tail.cu.in" >> "${tempfile}"

cat "${tempfile}" > ${logdir}/generated_file

input_dir=$(dirname "${input}")

echo "invoking: ${nvcc} -c ${input_type} ${tempfile} -I${input_dir} ${modified_flags[@]}" >> ${logdir}/log
"${nvcc}" -c ${input_type} "${tempfile}" "-I${input_dir}" "${modified_flags[@]}" 2> >(tee -a ${logdir}/error_log)

if [[ "${LIBCUDACXX_NVRTC_KEEP_LOG}" != "YES" ]]
then
    rm -rf $logdir
fi