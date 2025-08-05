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

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use futures::AsyncWrite;
use snafu::ResultExt;

use crate::common::creator::{FilterCreator, FilterCreatorImpl, FinalizedSegment, SegmentBuilder};
use crate::common::CommonResult;
use crate::external_provider::ExternalTempFileProvider;
use crate::xor_filter::error::{CommonSnafu, Result};

/// XOR filter segment builder implementation.
#[derive(Debug, Default)]
pub struct XorFilterSegmentBuilderNew {
    /// Keys in this segment.
    keys: Vec<u64>,
}

impl XorFilterSegmentBuilderNew {
    /// Create a new XOR filter segment builder.
    pub fn new() -> Self {
        Self { keys: Vec::new() }
    }
}

impl SegmentBuilder for XorFilterSegmentBuilderNew {
    type Input = Vec<u64>;
    type Segment = FinalizedXorFilterSegment;

    fn add_data(&mut self, data: Self::Input) -> CommonResult<()> {
        self.keys.extend(data);
        Ok(())
    }

    fn finalize(self) -> Self::Segment {
        FinalizedXorFilterSegment::new(self.keys)
    }

    fn memory_usage(&self) -> usize {
        self.keys.len() * std::mem::size_of::<u64>()
    }
}

/// Finalized XOR filter segment.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedXorFilterSegment {
    /// Keys in this segment.
    pub keys: Vec<u64>,
}

impl FinalizedXorFilterSegment {
    /// Create a new finalized XOR filter segment.
    pub fn new(keys: Vec<u64>) -> Self {
        Self { keys }
    }

    /// Build the actual XOR filter from the keys.
    pub fn build_xor_filter(&self) -> Result<crate::xor_filter::XorFilter> {
        if self.keys.is_empty() {
            // Create an empty filter using create_from_keys
            return crate::xor_filter::XorFilter::create_from_keys(&[]).map_err(|e| {
                crate::xor_filter::error::Error::CreateXorFilter {
                    reason: format!("Failed to create empty XOR filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), 0),
                }
            });
        }

        let mut unique_keys = self.keys.clone();
        unique_keys.sort_unstable();
        unique_keys.dedup();

        crate::xor_filter::XorFilter::create_from_keys(&unique_keys).map_err(|e| {
            crate::xor_filter::error::Error::CreateXorFilter {
                reason: format!("Failed to create XOR filter: {}", e),
                location: snafu::Location::new(file!(), line!(), 0),
            }
        })
    }
}

impl FinalizedSegment for FinalizedXorFilterSegment {
    fn memory_usage(&self) -> usize {
        self.keys.len() * std::mem::size_of::<u64>()
    }

    fn serialize_for_storage(&self) -> CommonResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        for key in &self.keys {
            data.extend_from_slice(&key.to_le_bytes());
        }
        Ok(data)
    }

    fn deserialize_from_storage(bytes: &[u8]) -> CommonResult<Self> {
        if bytes.len() < 8 {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid segment data"),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

        let key_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

        if bytes.len() != 8 + key_count * 8 {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid segment data size",
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

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
        let filter =
            self.build_xor_filter()
                .map_err(|e| crate::common::CommonFilterError::External {
                    source: common_error::ext::BoxedError::new(e),
                    location: snafu::Location::new(file!(), line!(), 0),
                })?;

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

/// Refactored XOR filter creator using the common infrastructure.
pub struct XorFilterCreatorV2 {
    /// The underlying generic creator implementation.
    inner: FilterCreatorImpl<XorFilterSegmentBuilderNew>,
}

impl XorFilterCreatorV2 {
    /// Creates a new `XorFilterCreatorV2` with the specified number of rows per segment.
    pub fn new(
        rows_per_segment: usize,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        let initial_builder = XorFilterSegmentBuilderNew::new();
        let inner = FilterCreatorImpl::new(
            rows_per_segment,
            intermediate_provider,
            global_memory_usage,
            global_memory_usage_threshold,
            initial_builder,
        );

        Self { inner }
    }

    /// Returns the memory usage of the creating XOR filter.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

impl FilterCreator for XorFilterCreatorV2 {
    type Builder = XorFilterSegmentBuilderNew;
    type ElementInput = u64;

    async fn push_n_row_elems(
        &mut self,
        nrows: usize,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        // Convert iterator to Vec to handle the repeated usage across segments
        let elems_vec: Vec<u64> = elems.into_iter().collect();

        // For XOR filters, we add the same keys to each row
        // We call add_data once with the total row count
        self.inner
            .add_data(elems_vec, nrows, || XorFilterSegmentBuilderNew::new())
            .await
    }

    async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        let elems_vec: Vec<u64> = elems.into_iter().collect();
        self.inner
            .add_data(elems_vec, 1, || XorFilterSegmentBuilderNew::new())
            .await
    }

    async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> CommonResult<()> {
        self.inner
            .finish(writer, || XorFilterSegmentBuilderNew::new())
            .await
    }

    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

/// A segment builder for XOR filter.
#[derive(Debug, Default)]
pub struct XorFilterSegmentBuilder {
    keys: Vec<u64>,
}

impl XorFilterSegmentBuilder {
    /// Create a new XOR filter segment builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single key to the builder.
    pub fn add_key(&mut self, key: u64) -> Result<()> {
        self.keys.push(key);
        Ok(())
    }

    /// Add keys directly to the builder.
    pub fn add_keys<I>(&mut self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        self.keys.extend(keys);
        Ok(())
    }
}

/// `XorFilterCreator` is responsible for creating and managing XOR filters
/// for a set of elements. It divides the rows into segments and creates
/// XOR filters for each segment.
pub struct XorFilterCreator {
    /// Internal implementation using the new common infrastructure.
    inner: XorFilterCreatorV2,
}

impl XorFilterCreator {
    /// Creates a new `XorFilterCreator` with the specified number of rows per segment.
    ///
    /// # PANICS
    ///
    /// `rows_per_segment` <= 0
    pub fn new(
        rows_per_segment: usize,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        assert!(
            rows_per_segment > 0,
            "rows_per_segment must be greater than 0"
        );

        Self {
            inner: XorFilterCreatorV2::new(
                rows_per_segment,
                intermediate_provider,
                global_memory_usage,
                global_memory_usage_threshold,
            ),
        }
    }

    /// Add a segment directly to the creator.
    pub async fn add_segment(&mut self, segment_builder: XorFilterSegmentBuilder) -> Result<()> {
        // Convert old segment builder to the new format and add as single row
        let keys = segment_builder.keys;
        self.inner.push_row_elems(keys).await.context(CommonSnafu)
    }

    /// Adds multiple rows of keys to the XOR filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_n_row_elems<I>(&mut self, nrows: usize, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = u64> + Send,
    {
        self.inner
            .push_n_row_elems(nrows, keys)
            .await
            .context(CommonSnafu)
    }

    /// Adds a row of keys to the XOR filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_row_elems<I>(&mut self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = u64> + Send,
    {
        self.inner.push_row_elems(keys).await.context(CommonSnafu)
    }

    /// Finalizes any remaining segments and writes the XOR filters and metadata to the provided writer.
    pub async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> Result<()> {
        self.inner.finish(writer).await.context(CommonSnafu)
    }

    /// Returns the memory usage of the creating XOR filter.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use arrow_array::UInt32Array;
    use futures::io::Cursor;
    use prost::Message;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::value_hasher::ArrayValueHasher;
    use crate::xor_filter::XorFilter;

    #[tokio::test]
    async fn test_xor_filter_creator() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        let hasher = ArrayValueHasher::default();
        let array1: arrow_array::PrimitiveArray<arrow_array::types::UInt32Type> =
            UInt32Array::from(vec![1, 2]);
        let array2 = UInt32Array::from(vec![3, 4]);
        let array3 = UInt32Array::from(vec![5, 6]);

        // Using push_row_elems instead of add
        creator
            .push_row_elems(hasher.hash_values(&array1).unwrap())
            .await
            .unwrap();
        creator
            .push_row_elems(hasher.hash_values(&array2).unwrap())
            .await
            .unwrap();
        creator
            .push_row_elems(hasher.hash_values(&array3).unwrap())
            .await
            .unwrap();

        assert!(creator.memory_usage() > 0);

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = greptime_proto::v1::index::BloomFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 2);
        assert_eq!(meta.row_count, 3);
        assert_eq!(
            meta.bloom_filter_size as usize + meta_bytes.len() + 4,
            total_size
        );

        // Verify filters work by reconstructing them
        for (i, segment_loc) in meta.bloom_filter_locs.iter().enumerate() {
            let filter_bytes = &bytes
                [segment_loc.offset as usize..(segment_loc.offset + segment_loc.size) as usize];

            if !filter_bytes.is_empty() {
                let xf = XorFilter::deserialize(filter_bytes).unwrap();
                let segment_idx = meta.segment_loc_indices[i] as usize;

                if segment_idx == 0 {
                    // First segment should contain keys from first two rows
                    for key in hasher.hash_values(&array1).unwrap() {
                        assert!(xf.contains(key));
                    }
                    for key in hasher.hash_values(&array2).unwrap() {
                        assert!(xf.contains(key));
                    }
                } else {
                    // Second segment should contain keys from third row
                    for key in hasher.hash_values(&array3).unwrap() {
                        assert!(xf.contains(key));
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_xor_filter_creator_batch_push() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator: XorFilterCreator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        let hasher = ArrayValueHasher::default();
        let array = UInt32Array::from(vec![1, 2, 3]);

        creator
            .push_n_row_elems(5, hasher.hash_values(&array).unwrap())
            .await
            .unwrap();

        assert!(creator.memory_usage() > 0);

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = greptime_proto::v1::index::BloomFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 3); // 5 rows / 2 rows per segment = 3 segments
        assert_eq!(meta.row_count, 5);

        // All segments should contain the same keys since we pushed the same data to each row
        for segment_loc in &meta.bloom_filter_locs {
            let filter_bytes = &bytes
                [segment_loc.offset as usize..(segment_loc.offset + segment_loc.size) as usize];

            if !filter_bytes.is_empty() {
                let xf = XorFilter::deserialize(filter_bytes).unwrap();
                for key in hasher.hash_values(&array).unwrap() {
                    assert!(xf.contains(key));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_xor_filter_segment_builder() {
        let mut builder = XorFilterSegmentBuilder::new();

        builder.add_key(42).unwrap();
        builder.add_keys(vec![1, 2, 3]).unwrap();

        assert_eq!(builder.keys, vec![42, 1, 2, 3]);
    }
}
