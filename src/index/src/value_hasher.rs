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

use std::hash::{BuildHasher, Hasher};

use ahash::random_state::RandomState;
use arrow_array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};

/// A hash utility to convert [`Array`] to a list of hash value.
#[derive(Debug)]
pub struct ArrayValueHasher {
    // Random state for creating hashers
    random_state: RandomState,
}

impl ArrayValueHasher {
    /// Create a RandomState with four fixed seeds
    fn create_random_state() -> RandomState {
        RandomState::with_seeds(
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
        )
    }
}

impl Default for ArrayValueHasher {
    fn default() -> Self {
        Self {
            random_state: Self::create_random_state(),
        }
    }
}

impl ArrayValueHasher {
    /// Hash [`Array`] into a list of hash values (`u64`).
    ///
    /// For numeric types, the original values are converted into `u64` without hash.
    /// For null values, it will skip it. So the returned value list might have
    /// different length than the input array.
    ///
    /// Returns `None` if the array type is not supported.
    pub fn hash_values(&self, values: &dyn Array) -> Option<Vec<u64>> {
        let len = values.len();
        let mut result = Vec::with_capacity(len);

        // Handle numeric types by preserving original values
        if let Some(array) = values.as_any().downcast_ref::<UInt64Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i));
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<UInt32Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<UInt16Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<UInt8Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Int64Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Int32Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Int16Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Int8Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i) as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Float32Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i).to_bits() as u64);
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<Float64Array>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                result.push(array.value(i).to_bits() as u64);
            }
            return Some(result);
        }

        // Hash all other types using the RandomState
        if let Some(array) = values.as_any().downcast_ref::<BooleanArray>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                let mut hasher = self.random_state.build_hasher();
                let value = array.value(i) as u8;
                hasher.write_u8(value);
                result.push(hasher.finish());
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<StringArray>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                let mut hasher = self.random_state.build_hasher();
                let bytes = array.value(i).as_bytes();
                hasher.write(bytes);
                result.push(hasher.finish());
            }
            return Some(result);
        } else if let Some(array) = values.as_any().downcast_ref::<BinaryArray>() {
            for i in 0..len {
                if array.is_null(i) {
                    continue;
                }
                let mut hasher = self.random_state.build_hasher();
                let bytes = array.value(i);
                hasher.write(bytes);
                result.push(hasher.finish());
            }
            return Some(result);
        }

        // Return None for unsupported types
        None
    }
}
