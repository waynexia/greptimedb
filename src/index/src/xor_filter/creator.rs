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

pub mod finalize_segment;
pub mod intermediate_codec;

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use asynchronous_codec::{FramedRead, FramedWrite};
use futures::{stream, AsyncWrite, AsyncWriteExt, Stream, StreamExt};
use greptime_proto::v1::index::{XorFilterLoc, XorFilterMeta};
use prost::Message;
use snafu::ResultExt;

use crate::error::Result;
use crate::external_provider::ExternalTempFileProvider;
use crate::value_hasher::ValueHasher;
use crate::xor_filter::creator::finalize_segment::FinalizedXorFilterSegment;
use crate::xor_filter::creator::intermediate_codec::IntermediateXorFilterCodecV1;
use crate::xor_filter::error::{IntermediateSnafu, IoSnafu};
use crate::xor_filter::XorFilter;

/// The minimum memory usage threshold for flushing in-memory XOR filters to disk.
const MIN_MEMORY_USAGE_THRESHOLD: usize = 1024 * 1024; // 1MB

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

    /// Finalize the segment builder and return a finalized segment.
    pub fn finalize(self) -> FinalizedXorFilterSegment {
        FinalizedXorFilterSegment::new(self.keys)
    }
}

/// Storage for finalized XOR filters.
pub struct FinalizedXorFilterStorage {
    /// Indices of the segments in the sequence of finalized XOR filters.
    segment_indices: Vec<usize>,

    /// XOR filters that are stored in memory.
    in_memory: Vec<FinalizedXorFilterSegment>,

    /// Used to generate unique file IDs for intermediate XOR filters.
    intermediate_file_id_counter: usize,

    /// Prefix for intermediate XOR filter files.
    intermediate_prefix: String,

    /// The provider for intermediate XOR filter files.
    intermediate_provider: Arc<dyn ExternalTempFileProvider>,

    /// The memory usage of the in-memory XOR filters.
    memory_usage: usize,

    /// The global memory usage provided by the user to track the
    /// total memory usage of the creating XOR filters.
    global_memory_usage: Arc<AtomicUsize>,

    /// The threshold of the global memory usage of the creating XOR filters.
    global_memory_usage_threshold: Option<usize>,

    /// Records the number of flushed segments.
    flushed_seg_count: usize,
}

impl FinalizedXorFilterStorage {
    /// Creates a new `FinalizedXorFilterStorage`.
    pub fn new(
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        let external_prefix = format!("intm-xor-filters-{}", uuid::Uuid::new_v4());
        Self {
            segment_indices: Vec::new(),
            in_memory: Vec::new(),
            intermediate_file_id_counter: 0,
            intermediate_prefix: external_prefix,
            intermediate_provider,
            memory_usage: 0,
            global_memory_usage,
            global_memory_usage_threshold,
            flushed_seg_count: 0,
        }
    }

    /// Returns the memory usage of the storage.
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Adds a finalized segment to the storage.
    ///
    /// If the memory usage exceeds the threshold, flushes the in-memory XOR filters to disk.
    pub async fn add(&mut self, segment: FinalizedXorFilterSegment) -> Result<usize> {
        // Reuse the last segment if it is the same as the current one.
        if self
            .in_memory
            .last()
            .map(|s| s.keys == segment.keys)
            .unwrap_or(false)
        {
            let idx = self.flushed_seg_count + self.in_memory.len() - 1;
            self.segment_indices.push(idx);
            return Ok(idx);
        }

        // Update memory usage.
        let memory_diff = segment.keys.len() * std::mem::size_of::<u64>();
        self.memory_usage += memory_diff;
        self.global_memory_usage
            .fetch_add(memory_diff, Ordering::Relaxed);

        // Add the finalized XOR filter to the in-memory storage.
        self.in_memory.push(segment);
        let idx = self.flushed_seg_count + self.in_memory.len() - 1;
        self.segment_indices.push(idx);

        // Flush to disk if necessary.

        // Do not flush if memory usage is too low.
        if self.memory_usage < MIN_MEMORY_USAGE_THRESHOLD {
            return Ok(idx);
        }

        // Check if the global memory usage exceeds the threshold and flush to disk if necessary.
        if let Some(threshold) = self.global_memory_usage_threshold {
            let global = self.global_memory_usage.load(Ordering::Relaxed);

            if global > threshold {
                self.flush_in_memory_to_disk().await?;

                self.global_memory_usage
                    .fetch_sub(self.memory_usage, Ordering::Relaxed);
                self.memory_usage = 0;
            }
        }

        Ok(idx)
    }

    /// Drains the storage and returns indices of the segments and a stream of finalized XOR filters.
    pub async fn drain(
        &mut self,
    ) -> Result<(
        Vec<usize>,
        Pin<Box<dyn Stream<Item = Result<FinalizedXorFilterSegment>> + Send + '_>>,
    )> {
        // FAST PATH: memory only
        if self.intermediate_file_id_counter == 0 {
            return Ok((
                std::mem::take(&mut self.segment_indices),
                Box::pin(stream::iter(self.in_memory.drain(..).map(Ok))),
            ));
        }

        // SLOW PATH: memory + disk
        let mut on_disk = self
            .intermediate_provider
            .read_all(&self.intermediate_prefix)
            .await
            .context(IntermediateSnafu)?;
        on_disk.sort_unstable_by(|x, y| x.0.cmp(&y.0));

        let streams = on_disk
            .into_iter()
            .map(|(_, reader)| FramedRead::new(reader, IntermediateXorFilterCodecV1::default()));

        let in_memory_stream = stream::iter(self.in_memory.drain(..)).map(Ok);
        Ok((
            std::mem::take(&mut self.segment_indices),
            Box::pin(stream::iter(streams).flatten().chain(in_memory_stream)),
        ))
    }

    /// Flushes the in-memory XOR filters to disk.
    async fn flush_in_memory_to_disk(&mut self) -> Result<()> {
        let file_id = self.intermediate_file_id_counter;
        self.intermediate_file_id_counter += 1;
        self.flushed_seg_count += self.in_memory.len();

        let file_id = format!("{:08}", file_id);
        let mut writer = self
            .intermediate_provider
            .create(&self.intermediate_prefix, &file_id)
            .await
            .context(IntermediateSnafu)?;

        let fw = FramedWrite::new(&mut writer, IntermediateXorFilterCodecV1::default());
        // `forward()` will flush and close the writer when the stream ends
        if let Err(e) = stream::iter(self.in_memory.drain(..).map(Ok))
            .forward(fw)
            .await
        {
            writer.close().await.context(IoSnafu)?;
            writer.flush().await.context(IoSnafu)?;
            return Err(e);
        }

        Ok(())
    }
}

impl Drop for FinalizedXorFilterStorage {
    fn drop(&mut self) {
        self.global_memory_usage
            .fetch_sub(self.memory_usage, Ordering::Relaxed);
    }
}

/// `XorFilterCreator` is responsible for creating and managing XOR filters
/// for a set of elements. It divides the rows into segments and creates
/// XOR filters for each segment.
///
/// # Format
///
/// The XOR filter creator writes the following format to the writer:
///
/// ```text
/// +--------------------+--------------------+-----+----------------------+----------------------+
/// | XOR filter 0       | XOR filter 1       | ... | XorFilterMeta        | Meta size            |
/// +--------------------+--------------------+-----+----------------------+----------------------+
/// |<- bytes (size 0) ->|<- bytes (size 1) ->| ... |<- json (meta size) ->|<- u32 LE (4 bytes) ->|
/// ```
///
pub struct XorFilterCreator {
    /// The number of rows per segment set by the user.
    rows_per_segment: usize,

    /// Row count that added to the XOR filter so far.
    accumulated_row_count: usize,

    /// Current segment builder
    current_segment_builder: XorFilterSegmentBuilder,

    /// Storage for finalized XOR filters.
    finalized_xor_filters: FinalizedXorFilterStorage,

    /// Global memory usage of the XOR filter creator.
    global_memory_usage: Arc<AtomicUsize>,
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
            rows_per_segment,
            accumulated_row_count: 0,
            current_segment_builder: XorFilterSegmentBuilder::new(),
            global_memory_usage: global_memory_usage.clone(),
            finalized_xor_filters: FinalizedXorFilterStorage::new(
                intermediate_provider,
                global_memory_usage,
                global_memory_usage_threshold,
            ),
        }
    }

    /// Add a segment directly to the creator.
    pub async fn add_segment(&mut self, segment_builder: XorFilterSegmentBuilder) -> Result<()> {
        let segment = segment_builder.finalize();
        self.accumulated_row_count += 1;
        self.finalized_xor_filters.add(segment).await?;

        if self.accumulated_row_count % self.rows_per_segment == 0 {
            // Replace the current segment builder with a new one
            self.current_segment_builder = XorFilterSegmentBuilder::new();
        }
        Ok(())
    }

    /// Adds multiple rows of keys to the XOR filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_n_row_elems<I>(&mut self, mut nrows: usize, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        if nrows == 0 {
            return Ok(());
        }
        if nrows == 1 {
            return self.push_row_elems(keys).await;
        }

        let keys = keys.into_iter().collect::<Vec<_>>();
        while nrows > 0 {
            let rows_to_seg_end =
                self.rows_per_segment - (self.accumulated_row_count % self.rows_per_segment);
            let rows_to_push = nrows.min(rows_to_seg_end);
            nrows -= rows_to_push;

            self.accumulated_row_count += rows_to_push;

            self.current_segment_builder
                .add_keys(keys.iter().copied())?;

            if self.accumulated_row_count % self.rows_per_segment == 0 {
                // Finalize the current segment and start a new one
                let segment = std::mem::take(&mut self.current_segment_builder).finalize();
                self.finalized_xor_filters.add(segment).await?;
            }
        }

        Ok(())
    }

    /// Adds a row of keys to the XOR filter. If the number of accumulated rows
    /// reaches `rows_per_segment`, it finalizes the current segment.
    pub async fn push_row_elems<I>(&mut self, keys: I) -> Result<()>
    where
        I: IntoIterator<Item = u64>,
    {
        self.accumulated_row_count += 1;

        self.current_segment_builder.add_keys(keys)?;

        if self.accumulated_row_count % self.rows_per_segment == 0 {
            // Finalize the current segment and start a new one
            let segment = std::mem::take(&mut self.current_segment_builder).finalize();
            self.finalized_xor_filters.add(segment).await?;
        }

        Ok(())
    }

    /// Finalizes any remaining segments and writes the XOR filters and metadata to the provided writer.
    pub async fn finish(&mut self, mut writer: impl AsyncWrite + Unpin) -> Result<()> {
        if self.accumulated_row_count % self.rows_per_segment != 0 {
            // Finalize the current segment if there's any data
            let segment = std::mem::take(&mut self.current_segment_builder).finalize();
            if (!segment.keys.is_empty()) {
                self.finalized_xor_filters.add(segment).await?;
            }
        }

        let mut meta = XorFilterMeta {
            rows_per_segment: self.rows_per_segment as _,
            row_count: self.accumulated_row_count as _,
            ..Default::default()
        };

        let (indices, mut segs) = self.finalized_xor_filters.drain().await?;
        meta.segment_loc_indices = indices.into_iter().map(|i| i as u64).collect();
        meta.segment_count = meta.segment_loc_indices.len() as _;

        while let Some(segment) = segs.next().await {
            let segment = segment?;
            let filter = segment.build_filter()?;
            let bytes = filter.serialize()?;
            writer.write_all(&bytes).await.context(IoSnafu)?;

            let size = bytes.len() as u64;
            meta.xor_filter_locs.push(XorFilterLoc {
                offset: meta.xor_filter_size as _,
                size,
                element_count: segment.keys.len() as _,
            });
            meta.xor_filter_size += size;
        }

        let meta_bytes = meta.encode_to_vec();
        writer.write_all(&meta_bytes).await.context(IoSnafu)?;

        let meta_size = meta_bytes.len() as u32;
        writer
            .write_all(&meta_size.to_le_bytes())
            .await
            .context(IoSnafu)?;
        writer.flush().await.context(IoSnafu)?;

        Ok(())
    }

    /// Returns the memory usage of the creating XOR filter.
    pub fn memory_usage(&self) -> usize {
        self.current_segment_builder.keys.len() * std::mem::size_of::<u64>()
            + self.finalized_xor_filters.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use arrow_array::UInt32Array;
    use futures::io::Cursor;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::value_hasher::IntegerValueHasher;

    #[tokio::test]
    async fn test_xor_filter_creator() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        let hasher = IntegerValueHasher::new();
        let array1 = UInt32Array::from(vec![1, 2]);
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
        let meta = XorFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 2);
        assert_eq!(meta.row_count, 3);
        assert_eq!(
            meta.xor_filter_size as usize + meta_bytes.len() + 4,
            total_size
        );

        assert_eq!(meta.xor_filter_locs.len(), 2);

        // Verify the XOR filters by reading their locations and deserializing
        let xf0_bytes = &bytes[meta.xor_filter_locs[0].offset as usize
            ..(meta.xor_filter_locs[0].offset + meta.xor_filter_locs[0].size) as usize];
        let xf0 = XorFilter::deserialize(xf0_bytes).unwrap();
        assert!(xf0.contains(1));
        assert!(xf0.contains(2));
        assert!(xf0.contains(3));
        assert!(xf0.contains(4));

        let xf1_bytes = &bytes[meta.xor_filter_locs[1].offset as usize
            ..(meta.xor_filter_locs[1].offset + meta.xor_filter_locs[1].size) as usize];
        let xf1 = XorFilter::deserialize(xf1_bytes).unwrap();
        assert!(xf1.contains(5));
        assert!(xf1.contains(6));
    }

    #[tokio::test]
    async fn test_xor_filter_creator_add_segment() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Segment 1
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![1, 2]).unwrap();
        creator.add_segment(segment_builder).await.unwrap();

        // Segment 2
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![3, 4]).unwrap();
        creator.add_segment(segment_builder).await.unwrap();

        // Segment 3 with duplicate data - should be reused
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![3, 4]).unwrap();
        creator.add_segment(segment_builder).await.unwrap();

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = XorFilterMeta::decode(meta_bytes).unwrap();

        // We should have 3 segments but only 2 unique XOR filters
        assert_eq!(meta.segment_count, 3);
        assert_eq!(meta.xor_filter_locs.len(), 2);

        // The third segment should point to the second XOR filter
        assert_eq!(meta.segment_loc_indices[2], 1);
    }

    #[tokio::test]
    async fn test_xor_filter_creator_push_elems() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Push single rows
        creator.push_row_elems(vec![1, 2]).await.unwrap();
        creator.push_row_elems(vec![3, 4]).await.unwrap();
        creator.push_row_elems(vec![5, 6]).await.unwrap();

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = XorFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 2);
        assert_eq!(meta.row_count, 3);

        // Verify the XOR filters by reading their locations and deserializing
        let xf0_bytes = &bytes[meta.xor_filter_locs[0].offset as usize
            ..(meta.xor_filter_locs[0].offset + meta.xor_filter_locs[0].size) as usize];
        let xf0 = XorFilter::deserialize(xf0_bytes).unwrap();
        assert!(xf0.contains(1));
        assert!(xf0.contains(2));
        assert!(xf0.contains(3));
        assert!(xf0.contains(4));

        let xf1_bytes = &bytes[meta.xor_filter_locs[1].offset as usize
            ..(meta.xor_filter_locs[1].offset + meta.xor_filter_locs[1].size) as usize];
        let xf1 = XorFilter::deserialize(xf1_bytes).unwrap();
        assert!(xf1.contains(5));
        assert!(xf1.contains(6));
    }

    #[tokio::test]
    async fn test_xor_filter_creator_push_n_rows() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            2,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        // Push multiple rows at once
        creator.push_n_row_elems(5, vec![1, 2]).await.unwrap();
        creator.push_n_row_elems(5, vec![3, 4]).await.unwrap();

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let total_size = bytes.len();
        let meta_size_offset = total_size - 4;
        let meta_size = u32::from_le_bytes((&bytes[meta_size_offset..]).try_into().unwrap());

        let meta_bytes = &bytes[total_size - meta_size as usize - 4..total_size - 4];
        let meta = XorFilterMeta::decode(meta_bytes).unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 5); // 10 rows / 2 rows per segment = 5 segments
        assert_eq!(meta.row_count, 10);

        // Check that all segments contain the expected values
        let mut found_1_2 = false;
        let mut found_3_4 = false;

        for i in 0..meta.xor_filter_locs.len() {
            let xf_bytes = &bytes[meta.xor_filter_locs[i].offset as usize
                ..(meta.xor_filter_locs[i].offset + meta.xor_filter_locs[i].size) as usize];
            let xf = XorFilter::deserialize(xf_bytes).unwrap();

            if xf.contains(1) && xf.contains(2) {
                found_1_2 = true;
            }
            if xf.contains(3) && xf.contains(4) {
                found_3_4 = true;
            }
        }

        assert!(found_1_2);
        assert!(found_3_4);
    }
}
