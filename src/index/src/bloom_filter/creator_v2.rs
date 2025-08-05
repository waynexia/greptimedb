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

use crate::bloom_filter::segment::BloomFilterSegmentBuilder;
use crate::common::creator::{FilterCreator, FilterCreatorImpl};
use crate::common::CommonResult;
use crate::external_provider::ExternalTempFileProvider;
use crate::Bytes;

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
        self.inner.add_data(elems_vec, nrows, move || BloomFilterSegmentBuilder::new(false_positive_rate)).await
    }
    
    async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        let elems_vec: Vec<Bytes> = elems.into_iter().collect();
        let false_positive_rate = self.false_positive_rate;
        self.inner.add_data(elems_vec, 1, move || BloomFilterSegmentBuilder::new(false_positive_rate)).await
    }
    
    async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> CommonResult<()> {
        let false_positive_rate = self.false_positive_rate;
        self.inner.finish(writer, move || BloomFilterSegmentBuilder::new(false_positive_rate)).await
    }
    
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use futures::io::Cursor;
    use greptime_proto::v1::index::BloomFilterMeta;
    use prost::Message;

    use super::*;
    use crate::bloom_filter::BloomFilter;
    use crate::external_provider::MockExternalTempFileProvider;

    #[tokio::test]
    async fn test_bloom_filter_creator_v2() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = BloomFilterCreatorV2::new(
            2, // rows per segment
            0.01,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Add data for 3 rows, which should create 2 segments
        creator
            .push_row_elems(vec![b"a".to_vec(), b"b".to_vec()])
            .await
            .unwrap();
        creator
            .push_row_elems(vec![b"c".to_vec(), b"d".to_vec()])
            .await
            .unwrap();
        creator
            .push_row_elems(vec![b"e".to_vec(), b"f".to_vec()])
            .await
            .unwrap();

        assert!(creator.memory_usage() > 0);

        creator.finish(&mut writer).await.unwrap();

        // Verify the structure
        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = BloomFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 2);
        assert_eq!(meta.row_count, 3);
        
        // Verify filters work by reconstructing them
        for (i, segment_loc) in meta.bloom_filter_locs.iter().enumerate() {
            let filter_bytes = &bytes[segment_loc.offset as usize..(segment_loc.offset + segment_loc.size) as usize];
            
            // Convert bytes back to u64 vector
            let vec = filter_bytes
                .chunks_exact(std::mem::size_of::<u64>())
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            
            let bf = BloomFilter::from_vec(vec, segment_loc.element_count as usize);
            
            // Verify contents based on segment
            if i == 0 {
                // First segment should contain elements from first two rows
                assert!(bf.contains(&b"a"));
                assert!(bf.contains(&b"b"));
                assert!(bf.contains(&b"c"));
                assert!(bf.contains(&b"d"));
            } else {
                // Second segment should contain elements from third row
                assert!(bf.contains(&b"e"));
                assert!(bf.contains(&b"f"));
            }
        }
    }
    
    #[tokio::test]
    async fn test_bloom_filter_creator_v2_push_n_rows() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = BloomFilterCreatorV2::new(
            2,
            0.01,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Push 5 rows with the same elements
        creator
            .push_n_row_elems(5, vec![b"test".to_vec()])
            .await
            .unwrap();

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = BloomFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 3); // 5 rows / 2 rows per segment = 3 segments
        assert_eq!(meta.row_count, 5);
        
        // All segments should contain "test"
        for segment_loc in &meta.bloom_filter_locs {
            let filter_bytes = &bytes[segment_loc.offset as usize..(segment_loc.offset + segment_loc.size) as usize];
            
            let vec = filter_bytes
                .chunks_exact(std::mem::size_of::<u64>())
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            
            let bf = BloomFilter::from_vec(vec, segment_loc.element_count as usize);
            assert!(bf.contains(&b"test"));
        }
    }
}