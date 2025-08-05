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

use std::fmt::Debug;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::{AsyncReadExt, AsyncWrite, AsyncWriteExt, Stream, StreamExt};
use greptime_proto::v1::index::BloomFilterMeta;
use prost::Message;

use crate::common::{CommonFilterError, CommonResult};
use crate::external_provider::ExternalTempFileProvider;

/// A trait for building filter segments incrementally.
pub trait SegmentBuilder: Debug + Send + Sync {
    /// The type of data this builder accepts.
    type Input: Clone;

    /// The finalized segment type this builder produces.
    type Segment: FinalizedSegment;

    /// Add data to the segment builder.
    fn add_data(&mut self, data: Self::Input) -> CommonResult<()>;

    /// Finalize the segment and return the result.
    fn finalize(self) -> Self::Segment;

    /// Get the estimated memory usage of the builder.
    fn memory_usage(&self) -> usize;
}

/// A trait for finalized filter segments.
pub trait FinalizedSegment: Debug + Clone + PartialEq + Send + Sync {
    /// Get the memory usage of this segment.
    fn memory_usage(&self) -> usize;

    /// Serialize the segment to bytes for intermediate storage.
    fn serialize_for_storage(&self) -> CommonResult<Vec<u8>>;

    /// Deserialize the segment from bytes for intermediate storage.
    fn deserialize_from_storage(bytes: &[u8]) -> CommonResult<Self>;

    /// Build the final filter from this segment.
    fn build_filter(&self) -> CommonResult<Vec<u8>>;

    /// Get the element count for this segment.
    fn element_count(&self) -> usize;
}

/// Common interface for filter creators.
pub trait FilterCreator {
    /// The type of segment builder used.
    type Builder: SegmentBuilder;

    /// The input data type for elements.
    type ElementInput;

    /// Add elements for multiple rows at once.
    async fn push_n_row_elems(
        &mut self,
        nrows: usize,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()>;

    /// Add elements for a single row.
    async fn push_row_elems(
        &mut self,
        elems: impl IntoIterator<Item = Self::ElementInput> + Send,
    ) -> CommonResult<()>;

    /// Finalize and write the filter data to the provided writer.
    async fn finish(&mut self, writer: impl AsyncWrite + Unpin + Send) -> CommonResult<()>;

    /// Get the current memory usage.
    fn memory_usage(&self) -> usize;
}

/// Generic filter creator implementation that handles common segment-based creation logic.
pub struct FilterCreatorImpl<B: SegmentBuilder> {
    /// The number of rows per segment.
    rows_per_segment: usize,

    /// Row count accumulated so far.
    accumulated_row_count: usize,

    /// Current segment builder.
    current_builder: Option<B>,

    /// Storage for finalized segments.
    finalized_storage: FinalizedSegmentStorage<B::Segment>,

    /// Row count that has been finalized.
    finalized_row_count: usize,

    /// Global memory usage tracker.
    global_memory_usage: Arc<AtomicUsize>,

    /// Current builder memory usage.
    current_builder_memory: usize,
}

impl<B: SegmentBuilder> FilterCreatorImpl<B> {
    /// Create a new filter creator.
    pub fn new(
        rows_per_segment: usize,
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
        initial_builder: B,
    ) -> Self {
        assert!(
            rows_per_segment > 0,
            "rows_per_segment must be greater than 0"
        );

        Self {
            rows_per_segment,
            accumulated_row_count: 0,
            current_builder: Some(initial_builder),
            finalized_storage: FinalizedSegmentStorage::new(
                intermediate_provider,
                global_memory_usage.clone(),
                global_memory_usage_threshold,
            ),
            finalized_row_count: 0,
            global_memory_usage,
            current_builder_memory: 0,
        }
    }

    /// Add data to the current segment, finalizing if needed.
    pub async fn add_data(
        &mut self,
        data: B::Input,
        row_count: usize,
        mut create_builder: impl FnMut() -> B,
    ) -> CommonResult<()> {
        let mut remaining_rows = row_count;

        while remaining_rows > 0 {
            // Track memory usage
            let builder = self
                .current_builder
                .as_mut()
                .ok_or_else(|| CommonFilterError::Io {
                    error: std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "No current builder",
                    ),
                    location: snafu::Location::new(file!(), line!(), 0),
                })?;

            let old_memory = builder.memory_usage();
            builder.add_data(data.clone())?; // Clone the data for each segment
            let new_memory = builder.memory_usage();

            let memory_diff = new_memory.saturating_sub(old_memory);
            self.current_builder_memory = new_memory;
            self.global_memory_usage
                .fetch_add(memory_diff, Ordering::Relaxed);

            // Calculate how many rows until the next segment boundary
            let rows_to_segment_end =
                self.rows_per_segment - (self.accumulated_row_count % self.rows_per_segment);
            let rows_to_add = remaining_rows.min(rows_to_segment_end);

            self.accumulated_row_count += rows_to_add;
            remaining_rows -= rows_to_add;

            // Check if we need to finalize the segment
            if self.accumulated_row_count % self.rows_per_segment == 0 {
                self.finalize_current_segment(&mut create_builder).await?;
                self.finalized_row_count = self.accumulated_row_count;
            }
        }

        Ok(())
    }

    /// Finalize the creator and write to the provided writer.
    pub async fn finish<W>(
        &mut self,
        mut writer: W,
        create_builder: impl FnOnce() -> B,
    ) -> CommonResult<()>
    where
        W: AsyncWrite + Unpin + Send,
    {
        // Finalize any remaining data
        if self.accumulated_row_count > self.finalized_row_count {
            // For the final segment, we don't need a new builder
            if let Some(builder) = self.current_builder.take() {
                let segment = builder.finalize();
                self.finalized_storage.add(segment).await?;

                // Update memory tracking
                self.global_memory_usage
                    .fetch_sub(self.current_builder_memory, Ordering::Relaxed);
                self.current_builder_memory = 0;
            }
        }

        // Prepare metadata
        let mut meta = BloomFilterMeta {
            rows_per_segment: self.rows_per_segment as _,
            row_count: self.accumulated_row_count as _,
            ..Default::default()
        };

        // Drain segments and write them
        let (indices, mut segments) = self.finalized_storage.drain().await?;
        meta.segment_loc_indices = indices.into_iter().map(|i| i as u64).collect();
        meta.segment_count = meta.segment_loc_indices.len() as _;

        while let Some(segment) = segments.next().await {
            let segment = segment?;
            let filter_bytes = segment.build_filter()?;

            writer
                .write_all(&filter_bytes)
                .await
                .map_err(|e| CommonFilterError::Io {
                    error: e,
                    location: snafu::Location::new(file!(), line!(), 0),
                })?;

            let size = filter_bytes.len() as u64;
            meta.bloom_filter_locs
                .push(greptime_proto::v1::index::BloomFilterLoc {
                    offset: meta.bloom_filter_size as _,
                    size,
                    element_count: segment.element_count() as _,
                });
            meta.bloom_filter_size += size;
        }

        // Write metadata
        let meta_bytes = meta.encode_to_vec();
        writer
            .write_all(&meta_bytes)
            .await
            .map_err(|e| CommonFilterError::Io {
                error: e,
                location: snafu::Location::new(file!(), line!(), 0),
            })?;

        let meta_size = meta_bytes.len() as u32;
        writer
            .write_all(&meta_size.to_le_bytes())
            .await
            .map_err(|e| CommonFilterError::Io {
                error: e,
                location: snafu::Location::new(file!(), line!(), 0),
            })?;

        writer.flush().await.map_err(|e| CommonFilterError::Io {
            error: e,
            location: snafu::Location::new(file!(), line!(), 0),
        })?;

        // Reset builder for potential reuse
        self.current_builder = Some(create_builder());

        Ok(())
    }

    /// Get the total memory usage.
    pub fn memory_usage(&self) -> usize {
        self.current_builder_memory + self.finalized_storage.memory_usage()
    }

    async fn finalize_current_segment(
        &mut self,
        mut create_new_builder: impl FnMut() -> B,
    ) -> CommonResult<()> {
        if let Some(builder) = self.current_builder.take() {
            let segment = builder.finalize();
            self.finalized_storage.add(segment).await?;

            // Update memory tracking
            self.global_memory_usage
                .fetch_sub(self.current_builder_memory, Ordering::Relaxed);
            self.current_builder_memory = 0;

            // Create a new builder for the next segment
            self.current_builder = Some(create_new_builder());
        }
        Ok(())
    }
}

impl<B: SegmentBuilder> Drop for FilterCreatorImpl<B> {
    fn drop(&mut self) {
        self.global_memory_usage
            .fetch_sub(self.current_builder_memory, Ordering::Relaxed);
    }
}

/// Storage for finalized segments with memory management and disk spillover.
pub struct FinalizedSegmentStorage<S: FinalizedSegment> {
    /// Indices of segments in the sequence.
    segment_indices: Vec<usize>,

    /// Segments stored in memory.
    in_memory: Vec<S>,

    /// Counter for generating unique intermediate file IDs.
    intermediate_file_id_counter: usize,

    /// Prefix for intermediate files.
    intermediate_prefix: String,

    /// Provider for intermediate files.
    intermediate_provider: Arc<dyn ExternalTempFileProvider>,

    /// Memory usage of in-memory segments.
    memory_usage: usize,

    /// Global memory usage tracker.
    global_memory_usage: Arc<AtomicUsize>,

    /// Threshold for flushing to disk.
    global_memory_usage_threshold: Option<usize>,

    /// Number of segments flushed to disk.
    flushed_seg_count: usize,
}

impl<S: FinalizedSegment> FinalizedSegmentStorage<S> {
    /// Create new storage.
    pub fn new(
        intermediate_provider: Arc<dyn ExternalTempFileProvider>,
        global_memory_usage: Arc<AtomicUsize>,
        global_memory_usage_threshold: Option<usize>,
    ) -> Self {
        let prefix = format!("intm-filters-{}", uuid::Uuid::new_v4());
        Self {
            segment_indices: Vec::new(),
            in_memory: Vec::new(),
            intermediate_file_id_counter: 0,
            intermediate_prefix: prefix,
            intermediate_provider,
            memory_usage: 0,
            global_memory_usage,
            global_memory_usage_threshold,
            flushed_seg_count: 0,
        }
    }

    /// Add a segment, potentially flushing to disk.
    pub async fn add(&mut self, segment: S) -> CommonResult<usize> {
        // Check for deduplication
        if self.in_memory.last() == Some(&segment) {
            let idx = self.flushed_seg_count + self.in_memory.len() - 1;
            self.segment_indices.push(idx);
            return Ok(idx);
        }

        // Update memory usage
        let memory_diff = segment.memory_usage();
        self.memory_usage += memory_diff;
        self.global_memory_usage
            .fetch_add(memory_diff, Ordering::Relaxed);

        // Add segment
        self.in_memory.push(segment);
        let idx = self.flushed_seg_count + self.in_memory.len() - 1;
        self.segment_indices.push(idx);

        // Check if we need to flush to disk
        if self.memory_usage >= 1024 * 1024 {
            // 1MB threshold
            if let Some(threshold) = self.global_memory_usage_threshold {
                let global = self.global_memory_usage.load(Ordering::Relaxed);
                if global > threshold {
                    self.flush_to_disk().await?;
                    self.global_memory_usage
                        .fetch_sub(self.memory_usage, Ordering::Relaxed);
                    self.memory_usage = 0;
                }
            }
        }

        Ok(idx)
    }

    /// Get memory usage.
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Drain all segments.
    pub async fn drain(
        &mut self,
    ) -> CommonResult<(
        Vec<usize>,
        Pin<Box<dyn Stream<Item = CommonResult<S>> + Send + '_>>,
    )> {
        // Fast path: memory only
        if self.intermediate_file_id_counter == 0 {
            return Ok((
                std::mem::take(&mut self.segment_indices),
                Box::pin(futures::stream::iter(self.in_memory.drain(..).map(Ok))),
            ));
        }

        // Slow path: need to read from disk
        let mut on_disk = self
            .intermediate_provider
            .read_all(&self.intermediate_prefix)
            .await
            .map_err(|e| CommonFilterError::Intermediate {
                source: e,
                location: snafu::Location::new(file!(), line!(), 0),
            })?;
        on_disk.sort_unstable_by(|x, y| x.0.cmp(&y.0));

        let disk_stream = futures::stream::iter(on_disk).then(|(_, mut reader)| async move {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .await
                .map_err(|e| CommonFilterError::Io {
                    error: e,
                    location: snafu::Location::new(file!(), line!(), 0),
                })?;
            S::deserialize_from_storage(&bytes)
        });
        let memory_stream = futures::stream::iter(self.in_memory.drain(..).map(Ok));

        Ok((
            std::mem::take(&mut self.segment_indices),
            Box::pin(disk_stream.chain(memory_stream)),
        ))
    }

    async fn flush_to_disk(&mut self) -> CommonResult<()> {
        let file_id = format!("{:08}", self.intermediate_file_id_counter);
        self.intermediate_file_id_counter += 1;
        self.flushed_seg_count += self.in_memory.len();

        let mut writer = self
            .intermediate_provider
            .create(&self.intermediate_prefix, &file_id)
            .await
            .map_err(|e| CommonFilterError::Intermediate {
                source: e,
                location: snafu::Location::new(file!(), line!(), 0),
            })?;

        for segment in self.in_memory.drain(..) {
            let bytes = segment.serialize_for_storage()?;
            writer
                .write_all(&bytes)
                .await
                .map_err(|e| CommonFilterError::Io {
                    error: e,
                    location: snafu::Location::new(file!(), line!(), 0),
                })?;
        }

        writer.flush().await.map_err(|e| CommonFilterError::Io {
            error: e,
            location: snafu::Location::new(file!(), line!(), 0),
        })?;

        Ok(())
    }
}

impl<S: FinalizedSegment> Drop for FinalizedSegmentStorage<S> {
    fn drop(&mut self) {
        self.global_memory_usage
            .fetch_sub(self.memory_usage, Ordering::Relaxed);
    }
}
