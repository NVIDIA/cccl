#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

import platform
import os
import libcxx.util


class CXXCompiler(object):
    CM_Default = 0
    CM_PreProcess = 1
    CM_Compile = 2
    CM_Link = 3

    def __init__(self, path, first_arg,
                 flags=None, compile_flags=None, link_flags=None,
                 warning_flags=None, verify_supported=None,
                 verify_flags=None, use_verify=False,
                 modules_flags=None, use_modules=False,
                 use_ccache=False, use_warnings=False, compile_env=None,
                 cxx_type=None, cxx_version=None):
        self.source_lang = 'c++'
        self.path = path
        self.first_arg = first_arg or ''
        self.flags = list(flags or [])
        self.compile_flags = list(compile_flags or [])
        self.link_flags = list(link_flags or [])
        self.warning_flags = list(warning_flags or [])
        self.verify_supported = verify_supported
        self.use_verify = use_verify
        self.verify_flags = list(verify_flags or [])
        assert not use_verify or verify_supported
        assert not use_verify or verify_flags is not None
        self.modules_flags = list(modules_flags or [])
        self.use_modules = use_modules
        assert not use_modules or modules_flags is not None
        self.use_ccache = use_ccache
        self.use_warnings = use_warnings
        if compile_env is not None:
            self.compile_env = dict(compile_env)
        else:
            self.compile_env = None
        self.type = cxx_type
        self.version = cxx_version
        if self.type is None or self.version is None:
            self._initTypeAndVersion()

    def isVerifySupported(self):
        if self.verify_supported is None:
            self.verify_supported = self.hasCompileFlag(['-Xclang',
                                        '-verify-ignore-unexpected'])
            if self.verify_supported:
                self.verify_flags = [
                    '-Xclang', '-verify',
                    '-Xclang', '-verify-ignore-unexpected=note',
                    '-ferror-limit=1024'
                ]
        return self.verify_supported

    def useVerify(self, value=True):
        self.use_verify = value
        assert not self.use_verify or self.verify_flags is not None

    def useModules(self, value=True):
        self.use_modules = value
        assert not self.use_modules or self.modules_flags is not None

    def useCCache(self, value=True):
        self.use_ccache = value

    def useWarnings(self, value=True):
        self.use_warnings = value

    def _initTypeAndVersion(self):
        # Get compiler type and version
        try:
          macros = self.dumpMacros()
          compiler_type = None
          major_ver = minor_ver = patchlevel = None
          self.is_nvrtc = False

          if '__NVCC__' in macros.keys():
              compiler_type = 'nvcc'
              major_ver = macros['__CUDACC_VER_MAJOR__']
              minor_ver = macros['__CUDACC_VER_MINOR__']
              patchlevel = macros['__CUDACC_VER_BUILD__']
              if '__LIBCUDACXX_NVRTC_TEST__' in macros.keys():
                self.is_nvrtc = True
          elif '__NVCOMPILER' in macros.keys():
              compiler_type = "nvhpc"
              # nvhpc, unfortunately, adds an extra space between the macro name
              # and macro value in their macro dump mode.
              major_ver = macros['__NVCOMPILER'].strip()
              minor_ver = macros['___NVCOMPILER_MINOR__'].strip()
              patchlevel = macros['___NVCOMPILER_PATCHLEVEL__'].strip()
          elif '__INTEL_COMPILER' in macros.keys():
              compiler_type = "icc"
              major_ver = int(macros['__INTEL_COMPILER']) / 100
              minor_ver = (int(macros['__INTEL_COMPILER']) % 100) / 10
              patchlevel = int(macros['__INTEL_COMPILER']) % 10
          elif '__clang__' in macros.keys():
              compiler_type = 'clang'
              # Treat Apple's LLVM fork differently.
              if '__apple_build_version__' in macros.keys():
                  compiler_type = 'apple-clang'
              major_ver = macros['__clang_major__']
              minor_ver = macros['__clang_minor__']
              patchlevel = macros['__clang_patchlevel__']
          elif '__GNUC__' in macros.keys():
              compiler_type = 'gcc'
              major_ver = macros['__GNUC__']
              minor_ver = macros['__GNUC_MINOR__']
              patchlevel = macros['__GNUC_PATCHLEVEL__']

          if '__cplusplus' in macros.keys():
            cplusplus = macros['__cplusplus']
            if cplusplus[-1] == 'L':
              cplusplus = cplusplus[:-1]
            cpp_standard = int(cplusplus)

            if cpp_standard <= 199711:
              default_dialect = "c++03"
            elif cpp_standard <= 201103:
              default_dialect = "c++11"
            elif cpp_standard <= 201402:
              default_dialect = "c++14"
            elif cpp_standard <= 201703:
              default_dialect = "c++17"
            else:
              default_dialect = "c++20"
          else:
            default_dialect = "c++03"

          self.type = compiler_type
          self.version = (major_ver, minor_ver, patchlevel)
          self.default_dialect = default_dialect
        except:
          (self.type, self.version, self.default_dialect, self.is_nvrtc) = \
              self.dumpVersion()

        if self.type == 'nvcc':
            # Treat C++ as CUDA when the compiler is NVCC.
            self.source_lang = 'cu'

    def _basicCmd(self, source_files, out, mode=CM_Default, flags=[],
                  input_is_cxx=False):
        cmd = []
        if self.use_ccache \
                and not mode == self.CM_Link \
                and not mode == self.CM_PreProcess:
            cmd += [os.environ.get('CMAKE_CXX_COMPILER_LAUNCHER')]
        cmd += [self.path] + ([self.first_arg] if self.first_arg != '' else [])
        if out is not None:
            cmd += ['-o', out]
        if input_is_cxx:
            cmd += ['-x', self.source_lang]
        if isinstance(source_files, list):
            cmd += source_files
        elif isinstance(source_files, str):
            cmd += [source_files]
        else:
            raise TypeError('source_files must be a string or list')
        if mode == self.CM_PreProcess:
            cmd += ['-E']
        elif mode == self.CM_Compile:
            cmd += ['-c']
        cmd += self.flags
        if self.use_verify:
            cmd += self.verify_flags
            assert mode in [self.CM_Default, self.CM_Compile]
        if self.use_modules:
            cmd += self.modules_flags
        if mode != self.CM_Link:
            cmd += self.compile_flags
            if self.use_warnings:
                cmd += self.warning_flags
        if mode != self.CM_PreProcess and mode != self.CM_Compile:
            cmd += self.link_flags
        cmd += flags
        return cmd

    def preprocessCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags,
                             mode=self.CM_PreProcess,
                             input_is_cxx=True)

    def compileCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags,
                             mode=self.CM_Compile,
                             input_is_cxx=True) + ['-c']

    def linkCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags,
                              mode=self.CM_Link)

    def compileLinkCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags)

    def preprocess(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.preprocessCmd(source_files, out, flags)
        out, err, rc = libcxx.util.executeCommand(cmd, env=self.compile_env,
                                                  cwd=cwd)
        return cmd, out, err, rc

    def compile(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.compileCmd(source_files, out, flags)
        out, err, rc = libcxx.util.executeCommand(cmd, env=self.compile_env,
                                                  cwd=cwd)
        return cmd, out, err, rc

    def link(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.linkCmd(source_files, out, flags)
        out, err, rc = libcxx.util.executeCommand(cmd, env=self.compile_env,
                                                  cwd=cwd)
        return cmd, out, err, rc

    def compileLink(self, source_files, out=None, flags=[],
                    cwd=None):
        cmd = self.compileLinkCmd(source_files, out, flags)
        out, err, rc = libcxx.util.executeCommand(cmd, env=self.compile_env,
                                                  cwd=cwd)
        return cmd, out, err, rc

    def compileLinkTwoSteps(self, source_file, out=None, object_file=None,
                            flags=[], cwd=None):
        if not isinstance(source_file, str):
            raise TypeError('This function only accepts a single input file')
        if object_file is None:
            # Create, use and delete a temporary object file if none is given.
            with_fn = lambda: libcxx.util.guardedTempFilename(suffix='.o')
        else:
            # Otherwise wrap the filename in a context manager function.
            with_fn = lambda: libcxx.util.nullContext(object_file)
        with with_fn() as object_file:
            cc_cmd, cc_stdout, cc_stderr, rc = self.compile(
                source_file, object_file, flags=flags, cwd=cwd)
            if rc != 0:
                return cc_cmd, cc_stdout, cc_stderr, rc
            link_cmd, link_stdout, link_stderr, rc = self.link(
                object_file, out=out, flags=flags, cwd=cwd)
            return (cc_cmd + ['&&'] + link_cmd, cc_stdout + link_stdout,
                    cc_stderr + link_stderr, rc)

    def dumpVersion(self, flags=[], cwd=None):
        dumpversion_cpp = os.path.join(
          os.path.dirname(os.path.abspath(__file__)), "dumpversion.cpp")
        with_fn = lambda: libcxx.util.guardedTempFilename(suffix=".exe")
        with with_fn() as exe:
          cmd, out, err, rc = self.compileLink([dumpversion_cpp], out=exe,
                                               flags=flags, cwd=cwd)
          if rc != 0:
            return ("unknown", (0, 0, 0), "c++03", False)
          out, err, rc = libcxx.util.executeCommand(exe, env=self.compile_env,
                                                    cwd=cwd)
          version = None
          try:
            version = eval(out)
          except:
            pass
          if not (isinstance(version, tuple) and 4 == len(version)):
            version = ("unknown", (0, 0, 0), "c++03", False)
        return version

    def dumpMacros(self, source_files=None, flags=[], cwd=None):
        if source_files is None:
            source_files = os.devnull
        flags = ['-dM'] + flags
        cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)
        if rc != 0:
            flags = ['-Xcompiler'] + flags
            cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)
            if rc != 0:
                return cmd, out, err, rc
        parsed_macros = {}
        lines = [l.strip() for l in out.split('\n') if l.strip()]
        for l in lines:
            # NVHPC also outputs the file contents from -E -dM for some reason; handle that
            if not l.startswith('#define '):
                if '__NVCOMPILER' not in parsed_macros.keys():
                    assert False, "a line not starting with '#define' encountered in predefined macro dump"
                else:
                    continue
            l = l[len('#define '):]
            macro, _, value = l.partition(' ')
            parsed_macros[macro] = value
        return parsed_macros

    def getTriple(self):
        if self.type == "msvc":
            return "x86_64-pc-windows-msvc"
        cmd = [self.path] + self.flags + ['-dumpmachine']
        return libcxx.util.capture(cmd).strip()

    def hasCompileFlag(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]

        # Add -Werror to ensure that an unrecognized flag causes a non-zero
        # exit code. -Werror is supported on all known non-nvcc compiler types.
        if self.type is not None and self.type != 'nvcc' and self.type != 'msvc':
            flags += ['-Werror', '-fsyntax-only']
        empty_cpp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty.cpp")
        cmd, out, err, rc = self.compile(empty_cpp, out=os.devnull,
                                         flags=flags)
        if out.find('flag is not supported with the configured host compiler') != -1:
            return False
        if err.find('flag is not supported with the configured host compiler') != -1:
            return False
        return rc == 0

    def addFlagIfSupported(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]
        if self.hasCompileFlag(flags):
            self.flags += flags
            return True
        else:
            return False

    def addCompileFlagIfSupported(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]
        if self.hasCompileFlag(flags):
            self.compile_flags += flags
            return True
        else:
            return False

    def hasWarningFlag(self, flag):
        """
        hasWarningFlag - Test if the compiler supports a given warning flag.
        Unlike addCompileFlagIfSupported, this function detects when
        "-Wno-<warning>" flags are unsupported. If flag is a
        "-Wno-<warning>" GCC will not emit an unknown option diagnostic unless
        another error is triggered during compilation.
        """
        assert isinstance(flag, str)
        assert flag.startswith('-W')
        if not flag.startswith('-Wno-'):
            return self.hasCompileFlag(flag)
        flags = ['-Werror', flag]
        old_use_warnings = self.use_warnings
        self.useWarnings(False)
        cmd = self.compileCmd('-', os.devnull, flags)
        self.useWarnings(old_use_warnings)
        # Remove '-v' because it will cause the command line invocation
        # to be printed as part of the error output.
        # TODO(EricWF): Are there other flags we need to worry about?
        if '-v' in cmd:
            cmd.remove('-v')
        out, err, rc = libcxx.util.executeCommand(
            cmd, input=libcxx.util.to_bytes('#error\n'))

        assert rc != 0
        if flag in err:
            return False
        return True

    def addWarningFlagIfSupported(self, flag):
        if self.hasWarningFlag(flag):
            if flag not in self.warning_flags:
                self.warning_flags += [flag]
            return True
        return False
