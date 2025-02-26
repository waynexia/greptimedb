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

use crate::error::Result;
use crate::xor_filter::XorFilter;

/// A finalized segment of XOR filter.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedXorFilterSegment {
    pub keys: Vec<u64>,
}

impl FinalizedXorFilterSegment {
    /// Create a new finalized segment.
    pub fn new(keys: Vec<u64>) -> Self {
        Self { keys }
    }

    /// Build a XOR filter from the finalized segment.
    pub fn build_filter(&self) -> Result<XorFilter> {
        XorFilter::create_from_keys(&self.keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finalized_xor_filter_segment() {
        let keys = vec![1, 2, 3, 4, 5];
        let segment = FinalizedXorFilterSegment::new(keys.clone());

        assert_eq!(segment.keys, keys);

        let filter = segment.build_filter().unwrap();
        for key in keys {
            assert!(filter.contains(key));
        }

        // Non-existent keys
        assert!(!filter.contains(100));
    }
}
