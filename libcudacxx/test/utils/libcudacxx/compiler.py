# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import os
import platform

import libcudacxx.util


class CXXCompiler(object):
    CM_Default = 0
    CM_PreProcess = 1
    CM_CheckCompileFlag = 2
    CM_Compile = 3
    CM_Link = 4

    def __init__(
        self,
        path,
        first_arg,
        flags=None,
        compile_flags=None,
        link_flags=None,
        warning_flags=None,
        verify_supported=None,
        verify_flags=None,
        use_verify=False,
        modules_flags=None,
        use_modules=False,
        use_ccache=False,
        use_warnings=False,
        compile_env=None,
        cxx_type=None,
        cxx_version=None,
        cuda_path=None,
    ):
        self.source_lang = "c++"
        self.path = path
        self.first_arg = first_arg or ""
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
        self.cuda_path = cuda_path
        if self.type is None or self.version is None:
            self._initTypeAndVersion()

    def isVerifySupported(self):
        if self.verify_supported is None:
            self.verify_supported = self.hasCompileFlag(
                ["-Xclang", "-verify-ignore-unexpected"]
            )
            if self.verify_supported:
                self.verify_flags = [
                    "-Xclang",
                    "-verify",
                    "-Xclang",
                    "-verify-ignore-unexpected=note",
                    "-ferror-limit=1024",
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

            if "__NVCC__" in macros.keys():
                compiler_type = "nvcc"
                major_ver = int(macros["__CUDACC_VER_MAJOR__"])
                minor_ver = int(macros["__CUDACC_VER_MINOR__"])
                patchlevel = int(macros["__CUDACC_VER_BUILD__"])
            elif "__NVCOMPILER" in macros.keys():
                compiler_type = "nvhpc"
                # NVHPC, unfortunately, adds an extra space between the macro name
                # and macro value in their macro dump mode.
                major_ver = int(macros["__NVCOMPILER_MAJOR__"].strip())
                minor_ver = int(macros["__NVCOMPILER_MINOR__"].strip())
                patchlevel = int(macros["__NVCOMPILER_PATCHLEVEL__"].strip())
            elif "__clang__" in macros.keys():
                compiler_type = "clang"
                # Treat Apple's LLVM fork differently.
                if "__apple_build_version__" in macros.keys():
                    compiler_type = "apple-clang"
                major_ver = int(macros["__clang_major__"])
                minor_ver = int(macros["__clang_minor__"])
                patchlevel = int(macros["__clang_patchlevel__"])
            elif "__GNUC__" in macros.keys():
                compiler_type = "gcc"
                major_ver = int(macros["__GNUC__"])
                minor_ver = int(macros["__GNUC_MINOR__"])
                patchlevel = int(macros["__GNUC_PATCHLEVEL__"])
            elif "_MSC_VER" in macros.keys():
                compiler_type = "msvc"
                major_ver = int(macros["_MSC_FULL_VER"]) // 10000000
                minor_ver = int(macros["_MSC_FULL_VER"]) // 100000 % 100
                patchlevel = int(macros["_MSC_FULL_VER"]) % 100000

            if "__cplusplus" in macros.keys():
                if "_MSVC_LANG" in macros.keys():
                    msvc_lang = macros["_MSVC_LANG"]
                    if msvc_lang[-1] == "L":
                        msvc_lang = msvc_lang[:-1]
                    msvc_lang = int(msvc_lang)
                    if msvc_lang <= 201103:
                        default_dialect = "c++11"
                    elif msvc_lang <= 201402:
                        default_dialect = "c++14"
                    elif msvc_lang <= 201703:
                        default_dialect = "c++17"
                    elif msvc_lang > 201703:
                        default_dialect = "c++20"
                else:
                    cplusplus = macros["__cplusplus"]
                    if cplusplus[-1] == "L":
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
        except Exception:
            (self.type, self.version, self.default_dialect) = self.dumpVersion()

        if self.type == "nvcc":
            # Treat C++ as CUDA when the compiler is NVCC.
            self.source_lang = "cu"
        elif self.type == "clang":
            # Treat C++ as clang-cuda when the compiler is Clang.
            self.source_lang = "cu"

    def _basicCmdCl(
        self, source_files, out, mode=CM_Default, flags=[], input_is_cxx=False
    ):
        cmd = []

        if (
            self.use_ccache
            and not mode == self.CM_Link
            and not mode == self.CM_PreProcess
            and not mode == self.CM_CheckCompileFlag
        ):
            cmd += [os.environ.get("CMAKE_CUDA_COMPILER_LAUNCHER")]

        cmd += [self.path] + ([self.first_arg] if self.first_arg != "" else [])

        if isinstance(source_files, list):
            cmd += source_files
        elif isinstance(source_files, str):
            cmd += [source_files]
        else:
            raise TypeError("source_files must be a string or list")

        if mode == self.CM_PreProcess or mode == self.CM_CheckCompileFlag:
            cmd += ["/Zs", "/options:strict"]
        elif mode == self.CM_Compile:
            cmd += ["/c"]

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
        if (
            mode != self.CM_PreProcess
            and mode != self.CM_Compile
            and mode != self.CM_CheckCompileFlag
        ):
            cmd += self.link_flags
        cmd += flags
        if out is not None:
            cmd += ["/link", '/out:"{}"'.format(out)]
        return cmd

    def _basicCmd(
        self, source_files, out, mode=CM_Default, flags=[], input_is_cxx=False
    ):
        if self.path.startswith("cl") and not self.path.startswith("clang"):
            return self._basicCmdCl(source_files, out, mode, flags)

        cmd = []

        if (
            self.use_ccache
            and not mode == self.CM_Link
            and not mode == self.CM_PreProcess
            and not mode == self.CM_CheckCompileFlag
        ):
            cmd += [os.environ.get("CMAKE_CUDA_COMPILER_LAUNCHER")]
        cmd += [self.path] + ([self.first_arg] if self.first_arg != "" else [])
        if out is not None:
            cmd += ["-o", out]
        if input_is_cxx:
            cmd += ["-x", self.source_lang]
        if (
            self.type == "clang"
            and self.source_lang == "cu"
            and self.cuda_path is not None
        ):
            cmd += ["--cuda-path=" + self.cuda_path]
        if isinstance(source_files, list):
            cmd += source_files
        elif isinstance(source_files, str):
            cmd += [source_files]
        else:
            raise TypeError("source_files must be a string or list")
        if mode == self.CM_PreProcess:
            cmd += ["-E"]
        elif mode == self.CM_Compile or mode == self.CM_CheckCompileFlag:
            cmd += ["-c"]
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
        if (
            mode != self.CM_PreProcess
            and mode != self.CM_Compile
            and mode != self.CM_CheckCompileFlag
        ):
            cmd += self.link_flags
        cmd += flags
        return cmd

    def preprocessCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(
            source_files, out, flags=flags, mode=self.CM_PreProcess, input_is_cxx=True
        )

    def compileCmd(self, source_files, out=None, flags=[], mode=CM_Compile):
        return self._basicCmd(
            source_files, out, flags=flags, mode=mode, input_is_cxx=True
        ) + ["-c"]

    def linkCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags, mode=self.CM_Link)

    def compileLinkCmd(self, source_files, out=None, flags=[]):
        return self._basicCmd(source_files, out, flags=flags)

    def preprocess(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.preprocessCmd(source_files, out, flags)
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, env=self.compile_env, cwd=cwd
        )
        return cmd, out, err, rc

    def checkCompileFlag(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.compileCmd(source_files, out, flags, self.CM_CheckCompileFlag)
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, env=self.compile_env, cwd=cwd
        )
        return cmd, out, err, rc

    def compile(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.compileCmd(source_files, out, flags, self.CM_Compile)
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, env=self.compile_env, cwd=cwd
        )
        return cmd, out, err, rc

    def link(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.linkCmd(source_files, out, flags)
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, env=self.compile_env, cwd=cwd
        )
        return cmd, out, err, rc

    def compileLink(self, source_files, out=None, flags=[], cwd=None):
        cmd = self.compileLinkCmd(source_files, out, flags)
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, env=self.compile_env, cwd=cwd
        )
        return cmd, out, err, rc

    def compileLinkTwoSteps(
        self, source_file, out=None, object_file=None, flags=[], cwd=None
    ):
        if not isinstance(source_file, str):
            raise TypeError("This function only accepts a single input file")
        if object_file is None:
            # Create, use and delete a temporary object file if none is given.
            def with_fn():
                return libcudacxx.util.guardedTempFilename(suffix=".o")
        else:
            # Otherwise wrap the filename in a context manager function.
            def with_fn():
                return libcudacxx.util.nullContext(object_file)

        with with_fn() as object_file:
            cc_cmd, cc_stdout, cc_stderr, rc = self.compile(
                source_file, object_file, flags=flags, cwd=cwd
            )
            if rc != 0:
                return cc_cmd, cc_stdout, cc_stderr, rc
            link_cmd, link_stdout, link_stderr, rc = self.link(
                object_file, out=out, flags=flags, cwd=cwd
            )
            return (
                cc_cmd + ["&&"] + link_cmd,
                cc_stdout + link_stdout,
                cc_stderr + link_stderr,
                rc,
            )

    def dumpVersion(self, flags=[], cwd=None):
        dumpversion_cpp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dumpversion.cpp"
        )

        def with_fn():
            return libcudacxx.util.guardedTempFilename(suffix=".exe")

        with with_fn() as exe:
            cmd, out, err, rc = self.compileLink(
                [dumpversion_cpp], out=exe, flags=flags, cwd=cwd
            )
            if rc != 0:
                return ("unknown", (0, 0, 0), "c++03")
            out, err, rc = libcudacxx.util.executeCommand(
                exe, env=self.compile_env, cwd=cwd
            )
            version = None
            try:
                version = eval(out)
            except Exception:
                pass

            if not (isinstance(version, tuple) and 3 == len(version)):
                version = ("unknown", (0, 0, 0), "c++03")
            return version

    def dumpMacros(self, source_files=None, flags=[], cwd=None):
        if source_files is None:
            source_files = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "empty.cpp"
            )

        old_flags = flags
        # Assume MSVC flags on Windows
        if platform.system() == "Windows":
            flags = ["/Zc:preprocessor", "/PD"] + old_flags
            cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)
            # Older MSVC does not support dumping macros
            if err.find("D9002") > 0:
                raise RuntimeError("Cannot be dumped on old MSVC")
            if rc != 0:
                flags = [
                    "-Xcompiler",
                    "/Zc:preprocessor",
                    "-Xcompiler",
                    "/PD",
                ] + old_flags
                cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)
                if err.find("D9002") > 0:
                    raise RuntimeError("Cannot be dumped on old MSVC")
        else:
            flags = ["-dM"] + flags
            cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)
            if rc != 0:
                flags = ["-Xcompiler"] + flags
                cmd, out, err, rc = self.preprocess(source_files, flags=flags, cwd=cwd)

        if rc != 0:
            raise RuntimeError("Macros failed to dump")

        parsed_macros = {}
        lines = [line.strip() for line in out.split("\n") if line.strip()]
        for line in lines:
            # NVHPC also outputs the file contents from -E -dM for some reason; handle that
            if not line.startswith("#define "):
                continue
            line = line[len("#define ") :]
            macro, _, value = line.partition(" ")
            parsed_macros[macro] = value
        return parsed_macros

    def getTriple(self):
        if self.type == "msvc":
            return "x86_64-pc-windows-msvc"
        cmd = [self.path] + self.flags + ["-dumpmachine"]
        return libcudacxx.util.capture(cmd).strip()

    def hasCompileFlag(self, flag):
        if isinstance(flag, list):
            flags = list(flag)
        else:
            flags = [flag]

        # Add -Werror to ensure that an unrecognized flag causes a non-zero
        # exit code. -Werror is supported on all known non-nvcc compiler types.
        if self.type is not None and self.type != "nvcc" and self.type != "msvc":
            flags += ["-Werror", "-fsyntax-only"]
        if self.type == "clang" and self.source_lang == "cu":
            flags += ["-Wno-unused-command-line-argument"]

        empty_cpp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "empty.cpp"
        )
        cmd, out, err, rc = self.checkCompileFlag(
            empty_cpp, out=os.devnull, flags=flags
        )
        if out.find("flag is not supported with the configured host compiler") != -1:
            return False
        if err.find("flag is not supported with the configured host compiler") != -1:
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
        assert flag.startswith("-W")
        if not flag.startswith("-Wno-"):
            return self.hasCompileFlag(flag)
        flags = ["-Werror", flag]
        old_use_warnings = self.use_warnings
        self.useWarnings(False)
        cmd = self.compileCmd("-", os.devnull, flags)
        self.useWarnings(old_use_warnings)
        # Remove '-v' because it will cause the command line invocation
        # to be printed as part of the error output.
        # TODO(EricWF): Are there other flags we need to worry about?
        if "-v" in cmd:
            cmd.remove("-v")
        out, err, rc = libcudacxx.util.executeCommand(
            cmd, input=libcudacxx.util.to_bytes("#error\n")
        )
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
