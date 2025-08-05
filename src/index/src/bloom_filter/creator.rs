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
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use fastbloom::BloomFilter as FastBloomFilter;
use futures::AsyncWrite;
use snafu::ResultExt;

use crate::bloom_filter::error::{CommonSnafu, Result};
use crate::common::creator::{FilterCreator, FilterCreatorImpl, FinalizedSegment, SegmentBuilder};
use crate::common::CommonResult;
use crate::external_provider::ExternalTempFileProvider;
use crate::Bytes;

/// Bloom filter segment builder implementation.
#[derive(Debug)]
pub struct BloomFilterSegmentBuilder {
    /// Distinct elements in this segment.
    distinct_elems: HashSet<Bytes>,
    /// False positive rate for creating the bloom filter.
    false_positive_rate: f64,
}

impl BloomFilterSegmentBuilder {
    /// Create a new bloom filter segment builder.
    pub fn new(false_positive_rate: f64) -> Self {
        Self {
            distinct_elems: HashSet::default(),
            false_positive_rate,
        }
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
        FinalizedBloomFilterSegment::new(self.distinct_elems, self.false_positive_rate)
    }

    fn memory_usage(&self) -> usize {
        self.distinct_elems.iter().map(|elem| elem.len()).sum()
    }
}

/// Finalized bloom filter segment.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedBloomFilterSegment {
    /// Bloom filter bytes.
    pub bloom_filter_bytes: Vec<u8>,
    /// Number of elements in this segment.
    pub element_count: usize,
}

impl FinalizedBloomFilterSegment {
    /// Create a new finalized bloom filter segment.
    pub fn new(distinct_elems: HashSet<Bytes>, false_positive_rate: f64) -> Self {
        let element_count = distinct_elems.len();

        if element_count == 0 {
            return Self {
                bloom_filter_bytes: Vec::new(),
                element_count: 0,
            };
        }

        let mut filter = FastBloomFilter::with_false_pos(false_positive_rate)
            .seed(&crate::bloom_filter::SEED)
            .expected_items(element_count);

        for elem in distinct_elems {
            filter.insert(&elem);
        }

        let bloom_filter_bytes: Vec<u8> = filter
            .as_slice()
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();

        Self {
            bloom_filter_bytes,
            element_count,
        }
    }
}

impl FinalizedSegment for FinalizedBloomFilterSegment {
    fn memory_usage(&self) -> usize {
        self.bloom_filter_bytes.len() + std::mem::size_of::<usize>()
    }

    fn serialize_for_storage(&self) -> CommonResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(&(self.element_count as u64).to_le_bytes());
        data.extend_from_slice(&(self.bloom_filter_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(&self.bloom_filter_bytes);
        Ok(data)
    }

    fn deserialize_from_storage(bytes: &[u8]) -> CommonResult<Self> {
        if bytes.len() < 16 {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid segment data"),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

        let element_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let filter_size = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;

        if bytes.len() != 16 + filter_size {
            return Err(crate::common::CommonFilterError::Io {
                error: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid segment data size",
                ),
                location: snafu::Location::new(file!(), line!(), 0),
            });
        }

        let bloom_filter_bytes = bytes[16..].to_vec();

        Ok(Self {
            bloom_filter_bytes,
            element_count,
        })
    }

    fn build_filter(&self) -> CommonResult<Vec<u8>> {
        Ok(self.bloom_filter_bytes.clone())
    }

    fn element_count(&self) -> usize {
        self.element_count
    }
}

/// Refactored bloom filter creator using the common infrastructure.
pub struct BloomFilterCreatorV2 {
    /// The underlying generic creator implementation.
    inner: FilterCreatorImpl<BloomFilterSegmentBuilder>,

    /// False positive rate for creating new builders.
    false_positive_rate: f64,
}

impl BloomFilterCreatorV2 {
    /// Creates a new `BloomFilterCreatorV2` with the specified number of rows per segment.
    pub fn new(
        rows_per_segment: usize,
        false_positive_rate: f64,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        let initial_builder = BloomFilterSegmentBuilder::new(false_positive_rate);
        let inner = FilterCreatorImpl::new(
            rows_per_segment,
            intermediate_provider,
            global_memory_usage,
            global_memory_usage_threshold,
            initial_builder,
        );

        Self {
            inner,
            false_positive_rate,
        }
    }

    /// Returns the memory usage of the creating bloom filter.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

impl FilterCreator for BloomFilterCreatorV2 {
    type Builder = BloomFilterSegmentBuilder;
    type ElementInput = Bytes;

    async fn push_n_row_elems(
        &mut self,
        nrows: usize,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        // Convert iterator to Vec to handle the repeated usage across segments
        let elems_vec: Vec<Bytes> = elems.into_iter().collect();

        // For bloom filters, we add the same elements to each row
        // We call add_data once with the total row count
        let false_positive_rate = self.false_positive_rate;
        self.inner
            .add_data(elems_vec, nrows, move || {
                BloomFilterSegmentBuilder::new(false_positive_rate)
            })
            .await
    }

    async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        let elems_vec: Vec<Bytes> = elems.into_iter().collect();
        let false_positive_rate = self.false_positive_rate;
        self.inner
            .add_data(elems_vec, 1, move || {
                BloomFilterSegmentBuilder::new(false_positive_rate)
            })
            .await
    }

    async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> CommonResult<()> {
        let false_positive_rate = self.false_positive_rate;
        self.inner
            .finish(writer, move || {
                BloomFilterSegmentBuilder::new(false_positive_rate)
            })
            .await
    }

    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

/// `BloomFilterCreator` is responsible for creating and managing bloom filters
/// for a set of elements. It divides the rows into segments and creates
/// bloom filters for each segment.
///
/// # Format
///
/// The bloom filter creator writes the following format to the writer:
///
/// ```text
/// +--------------------+--------------------+-----+----------------------+----------------------+
/// | Bloom filter 0     | Bloom filter 1     | ... | BloomFilterMeta      | Meta size            |
/// +--------------------+--------------------+-----+----------------------+----------------------+
/// |<- bytes (size 0) ->|<- bytes (size 1) ->| ... |<- json (meta size) ->|<- u32 LE (4 bytes) ->|
/// ```
///
pub struct BloomFilterCreator {
    /// Internal implementation using the new common infrastructure.
    inner: BloomFilterCreatorV2,
}

impl BloomFilterCreator {
    /// Creates a new `BloomFilterCreator` with the specified number of rows per segment.
    ///
    /// # PANICS
    ///
    /// `rows_per_segment` <= 0
    pub fn new(
        rows_per_segment: usize,
        false_positive_rate: f64,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        assert!(
            rows_per_segment > 0,
            "rows_per_segment must be greater than 0"
        );

        Self {
            inner: BloomFilterCreatorV2::new(
                rows_per_segment,
                false_positive_rate,
                intermediate_provider,
                global_memory_usage,
                global_memory_usage_threshold,
            ),
        }
    }

    /// Adds multiple rows of elements to the bloom filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_n_row_elems(
        &mut self,
        nrows: usize,
        elems: impl IntoIterator<Item = Bytes> + Send,
    ) -> Result<()> {
        self.inner
            .push_n_row_elems(nrows, elems)
            .await
            .context(CommonSnafu)
    }

    /// Adds a row of elements to the bloom filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Bytes> + Send,
    ) -> Result<()> {
        self.inner.push_row_elems(elems).await.context(CommonSnafu)
    }

    /// Finalizes any remaining segments and writes the bloom filters and metadata to the provided writer.
    pub async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> Result<()> {
        self.inner.finish(writer).await.context(CommonSnafu)
    }

    /// Returns the memory usage of the creating bloom filter.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use fastbloom::BloomFilter;
    use futures::io::Cursor;
    use prost::Message;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;

    /// Converts a slice of bytes to a vector of `u64`.
    pub fn u64_vec_from_bytes(bytes: &[u8]) -> Vec<u64> {
        bytes
            .chunks_exact(std::mem::size_of::<u64>())
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[tokio::test]
    async fn test_bloom_filter_creator() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = BloomFilterCreator::new(
            2,
            0.01,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        creator
            .push_row_elems(vec![b"a".to_vec(), b"b".to_vec()])
            .await
            .unwrap();
        assert!(creator.memory_usage() > 0);

        creator
            .push_row_elems(vec![b"c".to_vec(), b"d".to_vec()])
            .await
            .unwrap();
        // Finalize the first segment
        assert!(creator.memory_usage() > 0);

        creator
            .push_row_elems(vec![b"e".to_vec(), b"f".to_vec()])
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

        let mut bfs = Vec::new();
        for segment in meta.bloom_filter_locs {
            let bloom_filter_bytes =
                &bytes[segment.offset as usize..(segment.offset + segment.size) as usize];
            let v = u64_vec_from_bytes(bloom_filter_bytes);
            let bloom_filter = BloomFilter::from_vec(v)
                .seed(&crate::bloom_filter::SEED)
                .expected_items(segment.element_count as usize);
            bfs.push(bloom_filter);
        }

        assert_eq!(meta.segment_loc_indices.len(), 2);

        let bf0 = &bfs[meta.segment_loc_indices[0] as usize];
        assert!(bf0.contains(&b"a"));
        assert!(bf0.contains(&b"b"));
        assert!(bf0.contains(&b"c"));
        assert!(bf0.contains(&b"d"));

        let bf1 = &bfs[meta.segment_loc_indices[1] as usize];
        assert!(bf1.contains(&b"e"));
        assert!(bf1.contains(&b"f"));
    }

    #[tokio::test]
    async fn test_bloom_filter_creator_batch_push() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator: BloomFilterCreator = BloomFilterCreator::new(
            2,
            0.01,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        creator
            .push_n_row_elems(5, vec![b"a".to_vec(), b"b".to_vec()])
            .await
            .unwrap();
        assert!(creator.memory_usage() > 0);

        creator
            .push_n_row_elems(5, vec![b"c".to_vec(), b"d".to_vec()])
            .await
            .unwrap();
        assert!(creator.memory_usage() > 0);

        creator
            .push_n_row_elems(10, vec![b"e".to_vec(), b"f".to_vec()])
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
        assert_eq!(meta.segment_count, 10);
        assert_eq!(meta.row_count, 20);
        assert_eq!(
            meta.bloom_filter_size as usize + meta_bytes.len() + 4,
            total_size
        );

        let mut bfs = Vec::new();
        for segment in meta.bloom_filter_locs {
            let bloom_filter_bytes =
                &bytes[segment.offset as usize..(segment.offset + segment.size) as usize];
            let v = u64_vec_from_bytes(bloom_filter_bytes);
            let bloom_filter = BloomFilter::from_vec(v)
                .seed(&crate::bloom_filter::SEED)
                .expected_items(segment.element_count as _);
            bfs.push(bloom_filter);
        }

        // 4 bloom filters to serve 10 segments
        assert_eq!(bfs.len(), 4);
        assert_eq!(meta.segment_loc_indices.len(), 10);

        for idx in meta.segment_loc_indices.iter().take(3) {
            let bf = &bfs[*idx as usize];
            assert!(bf.contains(&b"a"));
            assert!(bf.contains(&b"b"));
        }
        for idx in meta.segment_loc_indices.iter().take(5).skip(2) {
            let bf = &bfs[*idx as usize];
            assert!(bf.contains(&b"c"));
            assert!(bf.contains(&b"d"));
        }
        for idx in meta.segment_loc_indices.iter().take(10).skip(5) {
            let bf = &bfs[*idx as usize];
            assert!(bf.contains(&b"e"));
            assert!(bf.contains(&b"f"));
        }
    }

    #[tokio::test]
    async fn test_final_seg_all_null() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = BloomFilterCreator::new(
            2,
            0.01,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        creator
            .push_n_row_elems(4, vec![b"a".to_vec(), b"b".to_vec()])
            .await
            .unwrap();
        creator.push_row_elems(Vec::new()).await.unwrap();

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = greptime_proto::v1::index::BloomFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 3);
        assert_eq!(meta.row_count, 5);
    }
}
