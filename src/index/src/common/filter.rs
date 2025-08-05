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

//! Common trait definitions for probabilistic filters

use std::any::Any;
use std::fmt::{Debug, Display};

use common_base::bytes::Bytes;

use crate::common::CommonResult;

/// Common interface for all probabilistic filters (Bloom, XOR, etc.)
/// 
/// This trait defines the core operations that all probabilistic membership 
/// filters should support: membership testing, serialization, and introspection.
pub trait ProbabilisticFilter: Debug + Display + PartialEq + Send + Sync {
    /// Check if the key may be in the set.
    /// 
    /// False positives are possible, but false negatives are not.
    /// This is the core operation for probabilistic membership testing.
    fn contains(&self, key: u64) -> bool;

    /// Serialize the filter to bytes for storage.
    fn serialize(&self) -> CommonResult<Bytes>;

    /// Deserialize the filter from bytes.
    fn deserialize(bytes: &[u8]) -> CommonResult<Self>
    where
        Self: Sized;

    /// Returns the memory usage of the filter in bytes.
    /// 
    /// This is used for memory tracking and optimization decisions.
    fn memory_usage(&self) -> usize;

    /// Get a reference to the filter as `Any` for downcasting.
    /// 
    /// This enables type-safe access to filter-specific methods when needed.
    fn as_any(&self) -> &dyn Any;
}

/// Marker trait for filters that can be created from a collection of keys.
/// 
/// This separates the creation interface from the core filter interface,
/// allowing for different construction strategies across filter types.
pub trait FilterBuilder<F: ProbabilisticFilter> {
    type Error;

    /// Create a new filter from a collection of keys.
    fn create_from_keys(keys: &[u64]) -> Result<F, Self::Error>;
}