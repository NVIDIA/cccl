// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Native ABI boundary. Library handles outlive every copied entry point.

use crate::{
    Error, Layout, Result,
    headers::{Headers, hash_field},
};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs,
    path::{Path, PathBuf},
    ptr,
};

type Handle = *mut c_void;
type Status = c_int;
type Create = unsafe extern "C" fn(
    *mut Handle,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> Status;
type Destroy = unsafe extern "C" fn(*mut Handle) -> Status;
type AddExpression = unsafe extern "C" fn(Handle, *const c_char) -> Status;
type Compile = unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> Status;
type GetSize = unsafe extern "C" fn(Handle, *mut usize) -> Status;
type GetBytes = unsafe extern "C" fn(Handle, *mut c_char) -> Status;
type GetName = unsafe extern "C" fn(Handle, *const c_char, *mut *const c_char) -> Status;
type GetError = unsafe extern "C" fn(Status) -> *const c_char;

pub(crate) struct Compiler {
    _library: Library,
    _builtins: Library,
    pub(crate) identity: [u8; 32],
    create: Create,
    destroy: Destroy,
    add_expression: AddExpression,
    compile_program: Compile,
    log_size: GetSize,
    log: GetBytes,
    lto_size: GetSize,
    lto: GetBytes,
    name: GetName,
    error: GetError,
}
pub(crate) struct Output {
    pub(crate) ltoir: Vec<u8>,
    pub(crate) layouts: Vec<Layout>,
    pub(crate) log: String,
}

impl Compiler {
    pub(crate) fn input_paths(toolkit: &Path) -> [PathBuf; 2] {
        ["libnvrtc-builtins.so", "libnvrtc.so"].map(|name| toolkit.join("lib64").join(name))
    }

    pub(crate) fn load(toolkit: &Path) -> Result<Self> {
        if !cfg!(target_os = "linux") {
            return Err(Error::message(
                "the native provider currently supports Linux hosts",
            ));
        }
        let mut hash = Sha256::new();
        let mut load = |input: &Path| -> Result<Library> {
            let path = input.canonicalize().map_err(|e| Error::input(input, e))?;
            let name = input
                .file_name()
                .and_then(|name| name.to_str())
                .expect("fixed library name");
            hash_field(&mut hash, name.as_bytes());
            hash_field(
                &mut hash,
                &fs::read(&path).map_err(|e| Error::input(&path, e))?,
            );
            // SAFETY: the explicit toolkit is trusted native executable input;
            // only documented NVRTC entry points are called with their C ABI.
            unsafe { Library::new(&path) }
                .map_err(|e| Error::message(format!("cannot load {}: {e}", path.display())))
        };
        let [builtins_path, library_path] = Self::input_paths(toolkit);
        let builtins = load(&builtins_path)?;
        let library = load(&library_path)?;
        // SAFETY: symbols use signatures from nvrtc.h. Missing entry points
        // return errors before a program is created; handles remain owned.
        unsafe {
            let version = library
                .get::<unsafe extern "C" fn(*mut c_int, *mut c_int) -> Status>(b"nvrtcVersion\0")
                .map_err(|e| Error::message(format!("missing NVRTC entry point: {e}")))?;
            let (mut major, mut minor) = (0, 0);
            if version(&mut major, &mut minor) != 0 {
                return Err(Error::message("nvrtcVersion failed"));
            }
            hash_field(&mut hash, &major.to_le_bytes());
            hash_field(&mut hash, &minor.to_le_bytes());
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|e| {
                            Error::message(format!("missing NVRTC entry point {}: {e}", $name))
                        })?
                };
            }
            Ok(Self {
                create: symbol!("nvrtcCreateProgram", Create),
                destroy: symbol!("nvrtcDestroyProgram", Destroy),
                add_expression: symbol!("nvrtcAddNameExpression", AddExpression),
                compile_program: symbol!("nvrtcCompileProgram", Compile),
                log_size: symbol!("nvrtcGetProgramLogSize", GetSize),
                log: symbol!("nvrtcGetProgramLog", GetBytes),
                lto_size: symbol!("nvrtcGetLTOIRSize", GetSize),
                lto: symbol!("nvrtcGetLTOIR", GetBytes),
                name: symbol!("nvrtcGetLoweredName", GetName),
                error: symbol!("nvrtcGetErrorString", GetError),
                identity: hash.finalize().into(),
                _library: library,
                _builtins: builtins,
            })
        }
    }
    fn check(&self, status: Status, operation: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: NVRTC returns a non-null static NUL-terminated error string.
        let message = unsafe { CStr::from_ptr((self.error)(status)) }.to_string_lossy();
        Err(Error::message(format!("{operation}: {message} ({status})")))
    }
    pub(crate) fn compile(
        &self,
        source: &str,
        options: &[String],
        expressions: &[String],
        headers: &Headers,
    ) -> Result<Output> {
        let source = CString::new(source)?;
        let options = options
            .iter()
            .map(|s| CString::new(s.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let option_pointers: Vec<_> = options.iter().map(|s| s.as_ptr()).collect();
        let header_names: Vec<_> = headers
            .entries
            .iter()
            .map(|(name, _)| name.as_ptr())
            .collect();
        let header_contents: Vec<_> = headers
            .entries
            .iter()
            .map(|(_, contents)| contents.as_ptr())
            .collect();
        let count = c_int::try_from(headers.entries.len())
            .map_err(|_| Error::message("too many input headers"))?;
        let mut program = Program {
            raw: ptr::null_mut(),
            compiler: self,
        };
        // SAFETY: C strings and arrays remain live until after program destroy;
        // NVRTC initializes the uniquely owned handle.
        self.check(
            unsafe {
                (self.create)(
                    &mut program.raw,
                    source.as_ptr(),
                    c"cuda_coop.cu".as_ptr(),
                    count,
                    header_contents.as_ptr(),
                    header_names.as_ptr(),
                )
            },
            "nvrtcCreateProgram",
        )?;
        let expressions = expressions
            .iter()
            .map(|s| CString::new(s.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for expression in &expressions {
            // SAFETY: program and terminated expression remain live through compilation.
            self.check(
                unsafe { (self.add_expression)(program.raw, expression.as_ptr()) },
                "nvrtcAddNameExpression",
            )?;
        }
        // SAFETY: program and option strings are live for this synchronous call.
        let status = unsafe {
            (self.compile_program)(
                program.raw,
                option_pointers.len() as c_int,
                option_pointers.as_ptr(),
            )
        };
        let mut log = self.bytes(program.raw, self.log_size, self.log)?;
        if log.last() == Some(&0) {
            log.pop();
        }
        let log = String::from_utf8_lossy(&log).into_owned();
        self.check(status, "nvrtcCompileProgram")
            .map_err(|e| Error::message(format!("{e}\n{log}")))?;
        let mut layouts = Vec::with_capacity(expressions.len());
        for expression in &expressions {
            let mut lowered = ptr::null();
            // SAFETY: successful compilation owns the returned name until destroy.
            self.check(
                unsafe { (self.name)(program.raw, expression.as_ptr(), &mut lowered) },
                "nvrtcGetLoweredName",
            )?;
            if lowered.is_null() {
                return Err(Error::message("NVRTC returned a null layout name"));
            }
            // SAFETY: successful query returned a program-owned C string.
            layouts.push(decode_layout(
                unsafe { CStr::from_ptr(lowered) }
                    .to_str()
                    .map_err(|_| Error::message("non-UTF-8 layout name"))?,
            )?);
        }
        let ltoir = self.bytes(program.raw, self.lto_size, self.lto)?;
        if ltoir.is_empty() {
            return Err(Error::message("NVRTC emitted empty LTO-IR"));
        }
        Ok(Output {
            ltoir,
            layouts,
            log,
        })
    }
    fn bytes(&self, program: Handle, size: GetSize, get: GetBytes) -> Result<Vec<u8>> {
        let mut length = 0;
        // SAFETY: program is live; NVRTC writes one size_t.
        self.check(unsafe { size(program, &mut length) }, "NVRTC output size")?;
        let mut bytes = vec![0; length];
        if length != 0 {
            // SAFETY: exact reported byte capacity is allocated and writable.
            self.check(
                unsafe { get(program, bytes.as_mut_ptr().cast()) },
                "NVRTC output",
            )?;
        }
        Ok(bytes)
    }
}
struct Program<'a> {
    raw: Handle,
    compiler: &'a Compiler,
}
impl Drop for Program<'_> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: unique owner; the compiler library remains loaded.
            unsafe {
                (self.compiler.destroy)(&mut self.raw);
            }
        }
    }
}
fn decode_layout(name: &str) -> Result<Layout> {
    let invalid = || Error::message(format!("unsupported NVRTC layout name: {name}"));
    let values = name
        .strip_prefix("_Z12layout_probeILy")
        .and_then(|s| s.strip_suffix("EE"))
        .ok_or_else(invalid)?;
    let (size, alignment) = values.split_once("ELy").ok_or_else(invalid)?;
    let size: usize = size.parse().map_err(|_| invalid())?;
    let alignment: usize = alignment.parse().map_err(|_| invalid())?;
    if size == 0 || !alignment.is_power_of_two() || size % alignment != 0 {
        return Err(invalid());
    }
    Ok(Layout { size, alignment })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_literals_and_rejection() {
        assert_eq!(
            decode_layout("_Z12layout_probeILy1ELy1EE").unwrap(),
            Layout {
                size: 1,
                alignment: 1
            }
        );
        assert_eq!(
            decode_layout("_Z12layout_probeILy2064ELy16EE").unwrap(),
            Layout {
                size: 2064,
                alignment: 16
            }
        );
        for invalid in [
            "_Z12layout_probeILy0ELy1EE",
            "_Z12layout_probeILy32ELy3EE",
            "_Z12layout_probeILy33ELy16EE",
            "_Z12layout_probeILy32ELy0EE",
            "_Z12layout_probeILy18446744073709551616ELy1EE",
            "other",
            "_Z12layout_probeILy32ELy16EE.extra",
        ] {
            assert!(decode_layout(invalid).is_err(), "{invalid}");
        }
    }
}
