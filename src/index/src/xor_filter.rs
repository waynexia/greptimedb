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

//! XOR filter implementation for GreptimeDB.
//!
//! The XOR filter is a probabilistic data structure for membership testing.
//! It's similar to a Bloom filter but has some advantages - it's more compact
//! and provides faster lookups, with the trade-off of being immutable.
//!
//! This module provides functionality to create, serialize, deserialize, and use XOR filters.

pub mod applier;
pub mod creator;
pub mod creator_v2;
pub mod error;
pub mod reader;
pub mod segment;

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use common_base::bytes::Bytes;
use snafu::ResultExt;
use xorf::{BinaryFuse8, Filter};

use crate::common::{CommonResult, ProbabilisticFilter};
use crate::xor_filter::error::{
    CreateXorFilterSnafu, DeserializeXorFilterSnafu, Result, SerializeXorFilterSnafu,
};

/// Skipping index based on XOR filter.
#[derive(Debug)]
pub struct XorFilter {
    filter: BinaryFuse8,
}

impl XorFilter {
    /// Create a new XOR filter by building from keys.
    pub fn create_from_keys(keys: &[u64]) -> Result<Self> {
        Ok(Self {
            filter: BinaryFuse8::try_from(keys)
                .map_err(|reason| CreateXorFilterSnafu { reason }.build())?,
        })
    }
    
    /// Create a new XOR filter from keys (convenience method for common result).
    pub fn from(keys: &[u64]) -> CommonResult<Self> {
        Self::create_from_keys(keys).map_err(|e| {
            crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(e),
                location: snafu::Location::new(file!(), line!(), 0),
            }
        })
    }

    /// Serialize the XOR filter to bytes.
    pub fn serialize(&self) -> Result<Bytes> {
        let bytes = bincode::serialize(&self.filter).context(SerializeXorFilterSnafu)?;
        Ok(Bytes::from(bytes))
    }

    /// Deserialize the XOR filter from bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let filter: BinaryFuse8 = bincode::deserialize(bytes).context(DeserializeXorFilterSnafu)?;
        Ok(Self { filter })
    }

    /// Check if the key may be in the set.
    ///
    /// False positives are possible, but false negatives are not.
    pub fn contains(&self, key: u64) -> bool {
        self.filter.contains(&key)
    }

    /// Returns the memory usage of the XOR filter in bytes.
    pub fn memory_usage(&self) -> usize {
        // The BinaryFuse8 stores 3 arrays of fingerprints and a seed
        // fingerprints size * 3 + seed size
        std::mem::size_of::<BinaryFuse8>()
    }

    /// Get a reference to the inner XOR filter.
    pub fn inner(&self) -> &BinaryFuse8 {
        &self.filter
    }
}

impl Display for XorFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "XorFilter")
    }
}

impl PartialEq for XorFilter {
    fn eq(&self, other: &Self) -> bool {
        self.serialize().unwrap() == other.serialize().unwrap()
    }
}

impl AsRef<dyn Any> for XorFilter {
    fn as_ref(&self) -> &dyn Any {
        self
    }
}

impl ProbabilisticFilter for XorFilter {
    fn contains(&self, key: u64) -> bool {
        // Call the inner filter's contains method directly
        self.filter.contains(&key)
    }

    fn serialize(&self) -> CommonResult<Bytes> {
        // Use bincode directly to avoid recursive calls
        let bytes = bincode::serialize(&self.filter).map_err(|e| {
            crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(
                    crate::xor_filter::error::Error::SerializeXorFilter {
                        error: e,
                        location: snafu::Location::new(file!(), line!(), 0),
                    }
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            }
        })?;
        Ok(Bytes::from(bytes))
    }

    fn deserialize(bytes: &[u8]) -> CommonResult<Self> {
        // Use bincode directly to avoid recursive calls
        let filter: BinaryFuse8 = bincode::deserialize(bytes).map_err(|e| {
            crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(
                    crate::xor_filter::error::Error::DeserializeXorFilter {
                        error: e,
                        location: snafu::Location::new(file!(), line!(), 0),
                    }
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            }
        })?;
        Ok(Self { filter })
    }

    fn memory_usage(&self) -> usize {
        // Calculate memory usage directly to avoid recursive calls
        std::mem::size_of::<BinaryFuse8>()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_filter_basic() {
        let keys = vec![1, 2, 3, 4, 5];
        let filter = XorFilter::create_from_keys(&keys).unwrap();

        for key in keys {
            assert!(filter.contains(key));
        }

        // Non-existent keys
        assert!(!filter.contains(100));
        assert!(!filter.contains(200));
    }

    #[test]
    fn test_xor_filter_serialization() {
        let keys = vec![1, 2, 3, 4, 5];
        let filter = XorFilter::create_from_keys(&keys).unwrap();

        let serialized = filter.serialize().unwrap();
        let deserialized = XorFilter::deserialize(&serialized).unwrap();

        // Test both filters contain the same keys
        for key in keys {
            assert!(deserialized.contains(key));
        }

        // Non-existent keys should also behave the same
        assert!(!filter.contains(100));
        assert!(!deserialized.contains(100));

        // Test equality
        assert_eq!(filter, deserialized);
    }

    #[test]
    fn test_xor_filter_memory_usage() {
        let keys = vec![1, 2, 3, 4, 5];
        let filter = XorFilter::create_from_keys(&keys).unwrap();

        // Just ensure it returns a non-zero value
        assert!(filter.memory_usage() > 0);
    }
}
