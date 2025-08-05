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

use crate::xor_filter::segment::XorFilterSegmentBuilder;
use crate::common::creator::{FilterCreator, FilterCreatorImpl};
use crate::common::CommonResult;
use crate::external_provider::ExternalTempFileProvider;

/// Refactored XOR filter creator using the common infrastructure.
pub struct XorFilterCreatorV2 {
    /// The underlying generic creator implementation.
    inner: FilterCreatorImpl<XorFilterSegmentBuilder>,
}

impl XorFilterCreatorV2 {
    /// Creates a new `XorFilterCreatorV2` with the specified number of rows per segment.
    pub fn new(
        rows_per_segment: usize,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        let initial_builder = XorFilterSegmentBuilder::new();
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
    type Builder = XorFilterSegmentBuilder;
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
        self.inner.add_data(elems_vec, nrows, || XorFilterSegmentBuilder::new()).await
    }
    
    async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()> {
        let elems_vec: Vec<u64> = elems.into_iter().collect();
        self.inner.add_data(elems_vec, 1, || XorFilterSegmentBuilder::new()).await
    }
    
    async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> CommonResult<()> {
        self.inner.finish(writer, || XorFilterSegmentBuilder::new()).await
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
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::xor_filter::XorFilter;

    #[tokio::test]
    async fn test_xor_filter_creator_v2() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreatorV2::new(
            2, // rows per segment
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Add data for 3 rows, which should create 2 segments
        creator
            .push_row_elems(vec![1, 2])
            .await
            .unwrap();
        creator
            .push_row_elems(vec![3, 4])
            .await
            .unwrap();
        creator
            .push_row_elems(vec![5, 6])
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
            
            if !filter_bytes.is_empty() {
                let xf = XorFilter::deserialize(filter_bytes).unwrap();
                
                // Verify contents based on segment
                if i == 0 {
                    // First segment should contain keys from first two rows
                    assert!(xf.contains(1));
                    assert!(xf.contains(2));
                    assert!(xf.contains(3));
                    assert!(xf.contains(4));
                } else {
                    // Second segment should contain keys from third row
                    assert!(xf.contains(5));
                    assert!(xf.contains(6));
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_xor_filter_creator_v2_push_n_rows() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreatorV2::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Push 5 rows with the same keys
        creator
            .push_n_row_elems(5, vec![42])
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
        
        // All segments should contain key 42
        for segment_loc in &meta.bloom_filter_locs {
            let filter_bytes = &bytes[segment_loc.offset as usize..(segment_loc.offset + segment_loc.size) as usize];
            
            if !filter_bytes.is_empty() {
                let xf = XorFilter::deserialize(filter_bytes).unwrap();
                assert!(xf.contains(42));
            }
        }
    }
}