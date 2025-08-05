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

pub mod applier;
pub mod creator;
pub mod error;
pub mod reader;

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};

use common_base::bytes::Bytes;
use fastbloom::BloomFilter as FastBloomFilter;
use snafu::ResultExt;

use crate::bloom_filter::error::{Result, SerializeBloomFilterSnafu, DeserializeBloomFilterSnafu};
use crate::common::{CommonResult, ProbabilisticFilter};

/// The seed used for the Bloom filter.
pub const SEED: u128 = 42;

/// Skipping index based on Bloom filter.
/// 
/// This wrapper provides a unified interface for bloom filters while maintaining
/// compatibility with the existing fastbloom implementation.
#[derive(Debug)]
pub struct BloomFilter {
    filter: FastBloomFilter,
    element_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter from a FastBloomFilter and element count.
    pub fn new(filter: FastBloomFilter, element_count: usize) -> Self {
        Self {
            filter,
            element_count,
        }
    }

    /// Create a new Bloom filter from a vector and element count.
    pub fn from_vec(vec: Vec<u64>, element_count: usize) -> Self {
        let filter = FastBloomFilter::from_vec(vec)
            .seed(&SEED)
            .expected_items(element_count);
        Self::new(filter, element_count)
    }

    /// Check if the key may be in the set.
    ///
    /// False positives are possible, but false negatives are not.
    pub fn contains<T: AsRef<[u8]> + std::hash::Hash>(&self, key: &T) -> bool {
        self.filter.contains(key)
    }

    /// Check if the key (as u64) may be in the set.
    pub fn contains_u64(&self, key: u64) -> bool {
        self.filter.contains(&key.to_le_bytes())
    }

    /// Get a reference to the inner bloom filter.
    pub fn inner(&self) -> &FastBloomFilter {
        &self.filter
    }

    /// Serialize the bloom filter to bytes.
    pub fn serialize(&self) -> Result<Bytes> {
        // FastBloomFilter doesn't support serde, so we serialize as Vec<u64> from slice
        let bits_slice = self.filter.as_slice();
        let bits: Vec<u64> = bits_slice.to_vec();
        let bytes = bincode::serialize(&bits).context(SerializeBloomFilterSnafu)?;
        Ok(Bytes::from(bytes))
    }

    /// Deserialize the bloom filter from bytes.
    pub fn deserialize(bytes: &[u8], element_count: usize) -> Result<Self> {
        // Deserialize as Vec<u64> and reconstruct filter
        let bits: Vec<u64> = bincode::deserialize(bytes).context(DeserializeBloomFilterSnafu)?;
        let filter = FastBloomFilter::from_vec(bits)
            .seed(&SEED)
            .expected_items(element_count);
        Ok(Self::new(filter, element_count))
    }

    /// Returns the memory usage of the bloom filter in bytes.
    pub fn memory_usage(&self) -> usize {
        // Estimate based on the number of bits in the filter
        std::mem::size_of::<FastBloomFilter>() + self.filter.as_slice().len() * 8
    }

    /// Get the element count.
    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

impl Display for BloomFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "BloomFilter(elements={})", self.element_count)
    }
}

impl PartialEq for BloomFilter {
    fn eq(&self, other: &Self) -> bool {
        self.element_count == other.element_count &&
        self.filter.as_slice() == other.filter.as_slice()
    }
}

impl AsRef<dyn Any> for BloomFilter {
    fn as_ref(&self) -> &dyn Any {
        self
    }
}

impl ProbabilisticFilter for BloomFilter {
    fn contains(&self, key: u64) -> bool {
        self.contains_u64(key)
    }

    fn serialize(&self) -> CommonResult<Bytes> {
        let bits_slice = self.filter.as_slice();
        let bits: Vec<u64> = bits_slice.to_vec();
        let bytes = bincode::serialize(&bits).map_err(|e| {
            crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(
                    crate::bloom_filter::error::Error::SerializeBloomFilter {
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
        // Note: We need element_count for proper deserialization, but the trait doesn't provide it.
        // For now, we'll use a default value and let the caller handle proper initialization.
        let bits: Vec<u64> = bincode::deserialize(bytes).map_err(|e| {
            crate::common::CommonFilterError::External {
                source: common_error::ext::BoxedError::new(
                    crate::bloom_filter::error::Error::DeserializeBloomFilter {
                        error: e,
                        location: snafu::Location::new(file!(), line!(), 0),
                    }
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            }
        })?;
        let filter = FastBloomFilter::from_vec(bits)
            .seed(&SEED)
            .expected_items(1); // Default value, caller should reinitialize properly
        Ok(Self::new(filter, 0))  // element_count will need to be set by caller
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<FastBloomFilter>() + self.filter.as_slice().len() * 8
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
