// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Snapshot compiler inputs so identity describes the actual compilation bytes.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs,
    path::{Path, PathBuf},
};

pub(crate) struct Headers {
    pub(crate) entries: Vec<(CString, CString)>,
    pub(crate) digest: [u8; 32],
}

impl Headers {
    pub(crate) fn roots(checkout: &Path, toolkit: &Path) -> [(PathBuf, &'static str); 4] {
        [
            (checkout.join("cub/cub"), "cub/"),
            (checkout.join("libcudacxx/include"), ""),
            (checkout.join("thrust/thrust"), "thrust/"),
            (toolkit.join("include"), ""),
        ]
    }

    pub(crate) fn read(checkout: &Path, toolkit: &Path) -> Result<Self> {
        let mut files = BTreeMap::new();
        // First entry wins: the selected checkout always owns CCCL headers,
        // including when the toolkit also bundles an older CCCL installation.
        for (root, prefix) in Self::roots(checkout, toolkit) {
            visit(&root, &root, prefix, &mut files)?;
        }
        let mut hash = Sha256::new();
        let mut entries = Vec::with_capacity(files.len());
        for (name, contents) in files {
            hash_field(&mut hash, name.as_bytes());
            hash_field(&mut hash, &contents);
            let content = CString::new(contents)
                .map_err(|_| Error::message(format!("NUL byte in header: {name}")))?;
            entries.push((CString::new(name)?, content));
        }
        Ok(Self {
            entries,
            digest: hash.finalize().into(),
        })
    }
}

fn visit(
    root: &Path,
    directory: &Path,
    prefix: &str,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    for entry in fs::read_dir(directory).map_err(|e| Error::input(directory, e))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Header-directory symlinks would need cycle handling. The include
            // root itself may be a symlink (as in normal CUDA installations).
            if entry.file_type()?.is_symlink() {
                return Err(Error::message(format!(
                    "header directory symlink is unsupported: {}",
                    path.display()
                )));
            }
            visit(root, &path, prefix, files)?;
        } else if path.is_file() {
            let relative = path.strip_prefix(root).expect("child of include root");
            let name = format!(
                "{prefix}{}",
                relative
                    .to_str()
                    .ok_or_else(|| Error::message(format!(
                        "non-UTF-8 header path: {}",
                        path.display()
                    )))?
                    .replace('\\', "/")
            );
            if let std::collections::btree_map::Entry::Vacant(slot) = files.entry(name) {
                slot.insert(fs::read(&path).map_err(|e| Error::input(&path, e))?);
            }
        }
    }
    Ok(())
}

pub(crate) fn hash_field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
