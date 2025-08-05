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

use std::collections::HashSet;

use fastbloom::BloomFilter as FastBloomFilter;

use crate::bloom_filter::SEED;
use crate::common::creator::{SegmentBuilder, FinalizedSegment};
use crate::common::CommonResult;
use crate::Bytes;

/// Segment builder for bloom filters.
#[derive(Debug)]
pub struct BloomFilterSegmentBuilder {
    /// Distinct elements in the current segment.
    distinct_elems: HashSet<Bytes>,
    
    /// False positive rate for the bloom filter.
    false_positive_rate: f64,
}

impl BloomFilterSegmentBuilder {
    /// Create a new bloom filter segment builder.
    pub fn new(false_positive_rate: f64) -> Self {
        Self {
            distinct_elems: HashSet::new(),
            false_positive_rate,
        }
    }
    
    /// Get the element count.
    pub fn element_count(&self) -> usize {
        self.distinct_elems.len()
    }
}

impl SegmentBuilder for BloomFilterSegmentBuilder {
    type Input = Vec<Bytes>;
    type Segment = FinalizedBloomFilterSegment;
    
    fn add_data(&mut self, data: Self::Input) -> CommonResult<()> {
        for elem in data {
            self.distinct_elems.insert(elem);
        }
        Ok(())
    }
    
    fn finalize(self) -> Self::Segment {
        let element_count = self.distinct_elems.len();
        
        if element_count == 0 {
            // Handle empty segment
            return FinalizedBloomFilterSegment {
                bloom_filter_bytes: Vec::new(),
                element_count: 0,
            };
        }
        
        let mut bf = FastBloomFilter::with_false_pos(self.false_positive_rate)
            .seed(&SEED)
            .expected_items(element_count);
        
        for elem in self.distinct_elems {
            bf.insert(&elem);
        }
        
        // Convert to bytes
        let bf_slice = bf.as_slice();
        let mut bloom_filter_bytes = Vec::with_capacity(std::mem::size_of_val(bf_slice));
        for &x in bf_slice {
            bloom_filter_bytes.extend_from_slice(&x.to_le_bytes());
        }
        
        FinalizedBloomFilterSegment {
            bloom_filter_bytes,
            element_count,
        }
    }
    
    fn memory_usage(&self) -> usize {
        self.distinct_elems.iter().map(|elem| elem.len()).sum::<usize>()
            + self.distinct_elems.capacity() * std::mem::size_of::<Bytes>()
    }
}

/// A finalized bloom filter segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedBloomFilterSegment {
    /// The underlying bloom filter bytes.
    pub bloom_filter_bytes: Vec<u8>,
    
    /// The number of elements in the bloom filter.
    pub element_count: usize,
}

impl FinalizedSegment for FinalizedBloomFilterSegment {
    fn memory_usage(&self) -> usize {
        self.bloom_filter_bytes.len()
    }
    
    fn serialize_for_storage(&self) -> CommonResult<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Write element count
        bytes.extend_from_slice(&(self.element_count as u64).to_le_bytes());
        
        // Write bloom filter bytes length
        bytes.extend_from_slice(&(self.bloom_filter_bytes.len() as u64).to_le_bytes());
        
        // Write bloom filter bytes
        bytes.extend_from_slice(&self.bloom_filter_bytes);
        
        Ok(bytes)
    }
    
    fn deserialize_from_storage(bytes: &[u8]) -> CommonResult<Self> {
        if bytes.len() < 16 {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(std::io::ErrorKind::InvalidData, "Not enough bytes for header"),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }
        
        // Read element count
        let element_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        
        // Read bloom filter bytes length
        let bf_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        
        if bytes.len() < 16 + bf_len {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(std::io::ErrorKind::InvalidData, "Not enough bytes for bloom filter data"),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }
        
        // Read bloom filter bytes
        let bloom_filter_bytes = bytes[16..16 + bf_len].to_vec();
        
        Ok(Self {
            bloom_filter_bytes,
            element_count,
        })
    }
    
    fn build_filter(&self) -> CommonResult<Vec<u8>> {
        // For bloom filters, the stored bytes ARE the filter bytes
        Ok(self.bloom_filter_bytes.clone())
    }
    
    fn element_count(&self) -> usize {
        self.element_count
    }
}