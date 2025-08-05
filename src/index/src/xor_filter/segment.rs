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

use crate::common::creator::{FinalizedSegment, SegmentBuilder};
use crate::common::CommonResult;
use crate::xor_filter::XorFilter;

/// Segment builder for XOR filters.
#[derive(Debug)]
pub struct XorFilterSegmentBuilder {
    /// Keys in the current segment.
    keys: Vec<u64>,
}

impl XorFilterSegmentBuilder {
    /// Create a new XOR filter segment builder.
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }

    /// Get the key count.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for XorFilterSegmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentBuilder for XorFilterSegmentBuilder {
    type Input = Vec<u64>;
    type Segment = FinalizedXorFilterSegment;

    fn add_data(&mut self, data: Self::Input) -> CommonResult<()> {
        self.keys.extend(data);
        Ok(())
    }

    fn finalize(self) -> Self::Segment {
        FinalizedXorFilterSegment { keys: self.keys }
    }

    fn memory_usage(&self) -> usize {
        self.keys.len() * std::mem::size_of::<u64>()
    }
}

/// A finalized XOR filter segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedXorFilterSegment {
    /// The keys in this segment.
    pub keys: Vec<u64>,
}

impl FinalizedXorFilterSegment {
    /// Create a new finalized segment.
    pub fn new(keys: Vec<u64>) -> Self {
        Self { keys }
    }
}

impl FinalizedSegment for FinalizedXorFilterSegment {
    fn memory_usage(&self) -> usize {
        self.keys.len() * std::mem::size_of::<u64>()
    }

    fn serialize_for_storage(&self) -> CommonResult<Vec<u8>> {
        let mut bytes = Vec::new();

        // Write key count
        bytes.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());

        // Write keys
        for &key in &self.keys {
            bytes.extend_from_slice(&key.to_le_bytes());
        }

        Ok(bytes)
    }

    fn deserialize_from_storage(bytes: &[u8]) -> CommonResult<Self> {
        if bytes.len() < 8 {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Not enough bytes for key count",
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

        // Read key count
        let key_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

        let expected_size = 8 + key_count * 8;
        if bytes.len() < expected_size {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Not enough bytes for keys",
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

        // Read keys
        let mut keys = Vec::with_capacity(key_count);
        for i in 0..key_count {
            let start = 8 + i * 8;
            let end = start + 8;
            let key = u64::from_le_bytes(bytes[start..end].try_into().unwrap());
            keys.push(key);
        }

        Ok(Self { keys })
    }

    fn build_filter(&self) -> CommonResult<Vec<u8>> {
        if self.keys.is_empty() {
            return Ok(Vec::new());
        }

        let filter = XorFilter::from(&self.keys)?;
        let bytes = filter
            .serialize()
            .map_err(|e| crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(e),
                location: snafu::Location::new(file!(), line!(), 0),
            })?;
        Ok(bytes.to_vec())
    }

    fn element_count(&self) -> usize {
        self.keys.len()
    }
}
