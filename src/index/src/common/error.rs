// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Common error types shared by probabilistic filters

use std::any::Any;

use common_error::ext::{BoxedError, ErrorExt};
use common_error::status_code::StatusCode;
use common_macro::stack_trace_debug;
use snafu::{Location, Snafu};

/// Common error variants shared by all filter implementations
#[derive(Snafu)]
#[snafu(visibility(pub))]
#[stack_trace_debug]
pub enum CommonFilterError {
    #[snafu(display("IO error"))]
    Io {
        #[snafu(source)]
        error: std::io::Error,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display("Failed to decode protobuf"))]
    DecodeProto {
        #[snafu(source)]
        error: prost::DecodeError,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display("Intermediate error"))]
    Intermediate {
        source: crate::error::Error,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display("File size {size} is too small for filter"))]
    FileSizeTooSmall {
        size: u64,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display(
        "Unexpected filter meta size: max {max_meta_size}, actual {actual_meta_size}"
    ))]
    UnexpectedMetaSize {
        max_meta_size: u64,
        actual_meta_size: u64,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display("Invalid intermediate magic"))]
    InvalidIntermediateMagic {
        invalid: Vec<u8>,
        #[snafu(implicit)]
        location: Location,
    },

    #[snafu(display("External error"))]
    External {
        source: BoxedError,
        #[snafu(implicit)]
        location: Location,
    },
}

impl ErrorExt for CommonFilterError {
    fn status_code(&self) -> StatusCode {
        use CommonFilterError::*;

        match self {
            Io { .. }
            | FileSizeTooSmall { .. }
            | UnexpectedMetaSize { .. }
            | DecodeProto { .. }
            | InvalidIntermediateMagic { .. } => StatusCode::Unexpected,

            Intermediate { source, .. } => source.status_code(),
            External { source, .. } => source.status_code(),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub type CommonResult<T> = std::result::Result<T, CommonFilterError>;
