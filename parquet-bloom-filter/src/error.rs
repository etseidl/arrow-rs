// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;

/// Errors constructing split block Bloom filters.
#[derive(Debug, Clone, PartialEq)]
pub enum SbbfError {
    /// The requested false positive probability is outside `[0.0, 1.0)` or NaN.
    InvalidFpp(f64),
}

impl fmt::Display for SbbfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFpp(fpp) => write!(
                f,
                "False positive probability must be between 0.0 and 1.0, got {fpp}"
            ),
        }
    }
}

impl std::error::Error for SbbfError {}
