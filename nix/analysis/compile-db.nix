# Generate compile_commands.json for the CCCL repo.
#
# Strategy:
#   1. Attempt cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#   2. If cmake fails (no CUDA toolkit), generate synthetic fallback
#
# The synthetic fallback generates minimal compile_commands.json entries
# that are sufficient for clang-tidy, cppcheck, and iwyu to parse most code
# without full dependency resolution.
#
{ pkgs, lib, src }:

let
  # Python script to generate a synthetic compile_commands.json
  # for when cmake configure fails (missing CUDA toolkit, etc.).
  syntheticDbScript = pkgs.writeText "synthetic-compile-db.py" ''
    """Generate synthetic compile_commands.json from source file discovery."""
    import json
    import os
    import sys

    source_dir = sys.argv[1]
    output_file = sys.argv[2]

    extensions = {
        '.c': 'c',
        '.cc': 'c++',
        '.cpp': 'c++',
        '.cxx': 'c++',
        '.cu': 'cuda',
    }

    # Directories to skip
    skip_dirs = {'build', '.git', '_deps', '.devcontainer', 'lib', '__pycache__', 'wheelhouse'}

    # CCCL include paths
    include_flags = ' '.join([
        f'-I{source_dir}/libcudacxx/include',
        f'-I{source_dir}/cub',
        f'-I{source_dir}/thrust',
        f'-I{source_dir}/cudax/include',
        f'-I{source_dir}/c/include',
    ])

    # CCCL defines
    defines = ' '.join([
        '-D__CUDACC__',
        '-D_CCCL_COMPILER_NVCC',
    ])

    entries = []
    for root, dirs, files in os.walk(source_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for f in files:
            _, ext = os.path.splitext(f)
            if ext not in extensions:
                continue

            filepath = os.path.join(root, f)
            lang = extensions[ext]

            if lang == 'c':
                std_flag = '-std=c11'
                lang_flag = '-x c'
            elif lang == 'cuda':
                std_flag = '-std=c++17'
                lang_flag = '-x cuda'
            else:
                std_flag = '-std=c++17'
                lang_flag = '-x c++'

            entries.append({
                'directory': source_dir,
                'file': filepath,
                'command': f'clang++ {lang_flag} {std_flag} {include_flags} {defines} -c {filepath}',
            })

    with open(output_file, 'w') as fh:
        json.dump(entries, fh, indent=2)

    print(f"Generated synthetic compile_commands.json with {len(entries)} entries")
  '';

  # Use CUDA toolkit from nixpkgs if available (requires config.allowUnfree = true).
  hasCuda = pkgs ? cudaPackages && pkgs.cudaPackages ? cuda_nvcc;
  cudaDeps = lib.optionals hasCuda (with pkgs.cudaPackages; [
    cuda_nvcc
    cuda_cudart
    cuda_cccl
  ]);

  compileDb = pkgs.runCommand "compile-db-cccl" {
    nativeBuildInputs = with pkgs; [ cmake python3 gnumake gcc ] ++ cudaDeps;
  } ''
    mkdir -p $out

    echo "=== Generating compile_commands.json for CCCL ==="

    # Copy source to writable directory (cmake needs to write)
    cp -r ${src} source
    chmod -R u+w source

    # Attempt cmake configure (with CUDA if available)
    if cmake -B source/build \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_C_COMPILER=${pkgs.gcc}/bin/gcc \
      -DCMAKE_CXX_COMPILER=${pkgs.gcc}/bin/g++ \
      -S source 2>&1 | tee configure.log; then

      if [ -f source/build/compile_commands.json ] && \
         ${pkgs.python3}/bin/python3 -c "
import json, sys
with open('source/build/compile_commands.json') as f:
    db = json.load(f)
sys.exit(0 if len(db) > 0 else 1)
" 2>/dev/null; then
        # Rewrite paths: cmake generates entries with /build/source as the
        # source root, but analysis tools run against the nix store source.
        ${pkgs.python3}/bin/python3 -c "
import json, sys
src_store = '${src}'
with open('source/build/compile_commands.json') as f:
    db = json.load(f)
for entry in db:
    entry['directory'] = entry['directory'].replace('/build/source/build/', src_store + '/').replace('/build/source', src_store)
    entry['file'] = entry['file'].replace('/build/source', src_store)
    entry['command'] = entry['command'].replace('/build/source', src_store)
with open(sys.argv[1], 'w') as f:
    json.dump(db, f, indent=2)
print(f'Rewrote {len(db)} entries: /build/source -> {src_store}')
" $out/compile_commands.json
        echo "cmake" > $out/method.txt
        echo "Generated compile_commands.json via cmake"
      else
        echo "cmake produced empty or no compile_commands.json, using synthetic"
        ${pkgs.python3}/bin/python3 ${syntheticDbScript} \
          ${src} $out/compile_commands.json
        echo "synthetic" > $out/method.txt
      fi
    else
      echo "cmake configure failed, using synthetic compile_commands.json"
      ${pkgs.python3}/bin/python3 ${syntheticDbScript} \
        ${src} $out/compile_commands.json
      echo "synthetic" > $out/method.txt
    fi
  '';

in
compileDb
