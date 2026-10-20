// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Version-one host transport for the Oxide library-operation marker.
//!
//! A dependency build script can serve as its already-built provider executable:
//! normal Cargo execution embeds [`registration`] in a device crate constant;
//! `--oxide-library-request-v1` execution passes stdin/stdout to [`serve`];
//! `--oxide-library-inputs-v1` passes them to [`serve_inputs`] during macro expansion.
//! Use `cuda_device::library::registration!` in the getter to track these inputs.
//! The compiler reads the registration getter's constant MIR and invokes this
//! host process. This module never invokes Cargo or executes device Rust code.
//! The protocol supports the four i32 Load/Store forms documented by [`Operation`].
//! Device calls retain all participation, validity and scratch obligations of
//! [`crate::Specialization`]. No allocation or synchronization policy is supplied.

use crate::{Algorithm, Artifact, Error, Layout, Operation, Provider, Request, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"OXDLIB01";
const MAX_HEADER: usize = 1024 * 1024;
const MAX_LTO: usize = 256 * 1024 * 1024;

#[derive(Serialize)]
struct Registration<'a> {
    version: u32,
    provider: &'a str,
    executable: &'a Path,
    operation: &'a str,
    algorithm: &'a str,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    content = "bits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Scalar {
    Signed(u16),
    Unsigned(u16),
    Float(u16),
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    content = "scalar",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Argument {
    ConstPointer(Scalar),
    MutPointer(Scalar),
    Value(Scalar),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchRequest {
    version: u32,
    target: String,
    cuda_root: PathBuf,
    requests: Vec<SpecializationRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecializationRequest {
    operation: String,
    algorithm: String,
    dtype: Scalar,
    block: [u32; 3],
    items_per_thread: u32,
}

impl SpecializationRequest {
    fn native(&self) -> Result<Request> {
        if self.dtype != Scalar::Signed(32) {
            return Err(Error::message(
                "this provider supports signed i32 elements only",
            ));
        }
        if self.block[1..] != [1, 1] {
            return Err(Error::message(
                "this provider requires the exact block shape [B,1,1]",
            ));
        }
        let operation = match self.operation.as_str() {
            "load_full" => Operation::LoadFull,
            "load_filled" => Operation::LoadFilled,
            "store_full" => Operation::StoreFull,
            "store_partial" => Operation::StorePartial,
            _ => {
                return Err(Error::message(format!(
                    "unsupported operation {:?}",
                    self.operation
                )));
            }
        };
        let algorithm = match self.algorithm.as_str() {
            "direct" => Algorithm::Direct,
            "transpose" => Algorithm::Transpose,
            _ => {
                return Err(Error::message(format!(
                    "unsupported algorithm {:?}",
                    self.algorithm
                )));
            }
        };
        Request::new(operation, algorithm, self.block[0], self.items_per_thread)
    }
}

#[derive(Serialize)]
struct WireLayout {
    size: u64,
    alignment: u64,
}

impl TryFrom<Layout> for WireLayout {
    type Error = Error;
    fn try_from(layout: Layout) -> Result<Self> {
        Ok(Self {
            size: u64::try_from(layout.size)
                .map_err(|_| Error::message("layout size exceeds u64"))?,
            alignment: u64::try_from(layout.alignment)
                .map_err(|_| Error::message("layout alignment exceeds u64"))?,
        })
    }
}

#[derive(Serialize)]
struct SpecializationResponse {
    request_index: u32,
    identity: [u8; 32],
    symbol: String,
    arguments: Vec<Argument>,
    cub_layout: WireLayout,
    scratch: WireLayout,
    scratch_free: bool,
    mutates_payload: bool,
}

#[derive(Serialize)]
struct BatchResponse {
    version: u32,
    batch_identity: [u8; 32],
    specializations: Vec<SpecializationResponse>,
}

fn arguments(operation: Operation) -> Vec<Argument> {
    let mut arguments = vec![
        if operation.is_load() {
            Argument::ConstPointer(Scalar::Signed(32))
        } else {
            Argument::MutPointer(Scalar::Signed(32))
        },
        Argument::MutPointer(Scalar::Signed(32)),
    ];
    if matches!(operation, Operation::LoadFilled | Operation::StorePartial) {
        arguments.push(Argument::Value(Scalar::Signed(32)));
    }
    if operation == Operation::LoadFilled {
        arguments.push(Argument::Value(Scalar::Signed(32)));
    }
    arguments
}

/// Encode a constant registration for one native recipe.
///
/// `executable` is the live, already-built dependency build script, normally
/// obtained from `std::env::current_exe()`. Embed the returned string in the
/// device crate through `cargo::rustc-env`; a named getter must return that
/// constant directly. The marker uses the getter's function-item type.
///
/// # Errors
///
/// Rejects a nonabsolute executable or a path that cannot be encoded as JSON.
pub fn registration(
    executable: &Path,
    operation: Operation,
    algorithm: Algorithm,
) -> Result<String> {
    if !executable.is_absolute() {
        return Err(Error::message("provider executable path must be absolute"));
    }
    serde_json::to_string(&Registration {
        version: VERSION,
        provider: "cuda_coop",
        executable,
        operation: operation.token(),
        algorithm: match algorithm {
            Algorithm::Direct => "direct",
            Algorithm::Transpose => "transpose",
        },
    })
    .map_err(|error| Error::message(format!("cannot encode provider registration: {error}")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRequest {
    version: u32,
    cuda_root: PathBuf,
    target: String,
}

#[derive(Serialize)]
struct InputResponse {
    version: u32,
    paths: Vec<PathBuf>,
    environment: Vec<String>,
}

/// Declare native inputs for Oxide's registration macro without loading CUDA.
///
/// `checkout` selects the same CCCL tree as [`serve`]. The bounded version-one
/// JSON request contains `cuda_root` and `target`; the response lists header
/// directories and NVRTC libraries. The macro tracks files and directories
/// recursively before rustc emits dep-info, including new header files.
/// No GPU, native compiler invocation or library loading occurs here.
/// Cargo's timestamp freshness rules apply; replacements with unchanged or older
/// timestamps can remain undetected. Rebuild after restoring or replacing inputs
/// this way, including a symlink switched to an older installation.
///
/// # Errors
///
/// Rejects malformed, oversized or unsupported requests, nonabsolute toolkit
/// paths, invalid targets, and inaccessible checkout paths.
pub fn serve_inputs(checkout: &Path, input: impl Read, mut output: impl Write) -> Result<()> {
    let mut bytes = Vec::new();
    input
        .take((MAX_HEADER + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_HEADER {
        return Err(Error::message("provider input request exceeds 1 MiB"));
    }
    let request: InputRequest = serde_json::from_slice(&bytes)
        .map_err(|error| Error::message(format!("invalid provider input request: {error}")))?;
    if request.version != VERSION
        || !request.cuda_root.is_absolute()
        || !request.target.starts_with("compute_")
        || request.target.len() <= 8
        || !request.target[8..]
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    {
        return Err(Error::message(
            "provider inputs require version 1, an absolute toolkit path and compute_XX target",
        ));
    }
    let checkout = checkout
        .canonicalize()
        .map_err(|error| Error::input(checkout, error))?;
    let paths = crate::headers::Headers::roots(&checkout, &request.cuda_root)
        .into_iter()
        .map(|(path, _)| path)
        .chain(crate::nvrtc::Compiler::input_paths(&request.cuda_root))
        .collect();
    let response = serde_json::to_vec(&InputResponse {
        version: VERSION,
        paths,
        environment: Vec::new(),
    })
    .map_err(|error| Error::message(format!("cannot encode provider inputs: {error}")))?;
    if response.len() > MAX_HEADER {
        return Err(Error::message("provider input response exceeds 1 MiB"));
    }
    output.write_all(&response)?;
    Ok(())
}

/// Compile one versioned stdin request and write its framed native response.
///
/// `checkout` selects the CCCL checkout whose headers the provider captures.
/// The request supplies an explicit toolkit path and target. Stdout contains
/// only the frame; the caller reports errors on stderr and exits nonzero.
/// The 24-byte prefix is `OXDLIB01`, JSON header length, and raw LTO length
/// (both lengths are little-endian u64), followed by the two byte buffers.
/// Requests and response headers are limited to 1 MiB; LTO is limited to 256 MiB.
/// Runtime arguments are a scalar/raw-pointer tuple; scratch is inserted at
/// parameter two and uses a generic byte pointer after shared address conversion.
///
/// # Errors
///
/// Rejects malformed/version-mismatched requests, unsupported types/recipes,
/// invalid geometry, and duplicate specializations. Native compiler errors
/// retain NVRTC diagnostics. No response is written until compilation succeeds.
pub fn serve(checkout: &Path, input: impl Read, mut output: impl Write) -> Result<()> {
    let mut bytes = Vec::new();
    input
        .take((MAX_HEADER + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_HEADER {
        return Err(Error::message("provider request exceeds 1 MiB"));
    }
    let request: BatchRequest = serde_json::from_slice(&bytes)
        .map_err(|error| Error::message(format!("invalid provider request: {error}")))?;
    if request.version != VERSION {
        return Err(Error::message(format!(
            "unsupported provider protocol {}; expected {VERSION}",
            request.version
        )));
    }
    let requests: Vec<_> = request
        .requests
        .iter()
        .map(SpecializationRequest::native)
        .collect::<Result<_>>()?;
    let artifact = Provider::from_checkout(checkout, &request.cuda_root)?
        .compile(&requests, &request.target)?;
    write_response(&requests, artifact, &mut output)
}

fn write_response(requests: &[Request], artifact: Artifact, output: &mut impl Write) -> Result<()> {
    if artifact.ltoir.is_empty() || artifact.ltoir.len() > MAX_LTO {
        return Err(Error::message(
            "native LTO buffer must contain 1..=256 MiB bytes",
        ));
    }
    let specializations = artifact
        .specializations
        .into_iter()
        .zip(requests)
        .enumerate()
        .map(|(index, (entry, request))| {
            Ok(SpecializationResponse {
                request_index: u32::try_from(index)
                    .map_err(|_| Error::message("request index exceeds u32"))?,
                identity: entry.identity,
                symbol: entry.symbol,
                arguments: arguments(request.operation()),
                cub_layout: entry.cub_layout.try_into()?,
                scratch: entry.scratch.try_into()?,
                scratch_free: entry.scratch_free,
                mutates_payload: entry.mutates_payload,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = serde_json::to_vec(&BatchResponse {
        version: VERSION,
        batch_identity: artifact.identity,
        specializations,
    })
    .map_err(|error| Error::message(format!("cannot encode provider response: {error}")))?;
    if header.len() > MAX_HEADER {
        return Err(Error::message("provider response header exceeds 1 MiB"));
    }
    output.write_all(MAGIC)?;
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&(artifact.ltoir.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&artifact.ltoir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_query_needs_no_installed_toolkit() {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"));
        let toolkit = std::env::temp_dir().join("cuda-coop-input-query-no-toolkit");
        let request = serde_json::json!({"version":1,"cuda_root":toolkit,"target":"compute_120"});
        let mut bytes = Vec::new();
        serve_inputs(checkout, request.to_string().as_bytes(), &mut bytes).unwrap();
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["version"], 1);
        assert_eq!(response["paths"].as_array().unwrap().len(), 6);
        assert!(
            response["paths"]
                .as_array()
                .unwrap()
                .iter()
                .all(|path| Path::new(path.as_str().unwrap()).is_absolute())
        );
        assert!(response["environment"].as_array().unwrap().is_empty());
        for request in [
            serde_json::json!({"version":2,"cuda_root":toolkit,"target":"compute_120"}),
            serde_json::json!({"version":1,"cuda_root":"relative","target":"compute_120"}),
            serde_json::json!({"version":1,"cuda_root":toolkit,"target":"invalid"}),
        ] {
            assert!(serve_inputs(checkout, request.to_string().as_bytes(), Vec::new()).is_err());
        }
        assert!(serve_inputs(checkout, vec![b' '; MAX_HEADER + 1].as_slice(), Vec::new()).is_err());
    }

    #[test]
    fn registration_binds_native_recipe() {
        let value: serde_json::Value = serde_json::from_str(
            &registration(
                Path::new("/provider"),
                Operation::LoadFilled,
                Algorithm::Transpose,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["operation"], "load_filled");
        assert_eq!(value["algorithm"], "transpose");
        assert!(
            registration(
                Path::new("relative"),
                Operation::LoadFull,
                Algorithm::Direct
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_request_before_opening_toolkit() {
        let base = serde_json::json!({"version":1,"target":"compute_120","cuda_root":"/absent", "requests":[{
            "operation":"load_full","algorithm":"direct","dtype":{"kind":"signed","bits":32},"block":[128,1,1],"items_per_thread":4
        }]});
        for (path, value, diagnostic) in [
            (
                "/version",
                serde_json::json!(2),
                "unsupported provider protocol",
            ),
            (
                "/requests/0/dtype/bits",
                serde_json::json!(64),
                "signed i32",
            ),
            (
                "/requests/0/block/1",
                serde_json::json!(2),
                "exact block shape",
            ),
            (
                "/requests/0/operation",
                serde_json::json!("unknown"),
                "unsupported operation",
            ),
        ] {
            let mut request = base.clone();
            *request.pointer_mut(path).unwrap() = value;
            let mut output = Vec::new();
            let error = serve(
                Path::new("/absent"),
                serde_json::to_vec(&request).unwrap().as_slice(),
                &mut output,
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
            assert!(output.is_empty());
        }
    }
}
