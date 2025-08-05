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

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use common_base::range_read::RangeReader;
use greptime_proto::v1::index::{BloomFilterLoc as XorFilterLoc, BloomFilterMeta as XorFilterMeta};

use crate::common::reader::{FilterReader, FilterReaderImpl};
use crate::xor_filter::error::*;
use crate::xor_filter::XorFilter;

/// Default prefetch size of XOR filter meta.
pub const DEFAULT_PREFETCH_SIZE: u64 = 8192; // 8KiB

/// `XorFilterReader` reads the XOR filter from the file.
/// 
/// This trait provides backward compatibility with the existing XOR filter API
/// while delegating to the common FilterReader implementation.
#[async_trait]
pub trait XorFilterReader: Sync {
    /// Reads range of bytes from the file.
    async fn range_read(&self, offset: u64, size: u32) -> Result<Bytes>;

    /// Reads bunch of ranges from the file.
    async fn read_vec(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>>;

    /// Reads the meta information of the XOR filter.
    async fn metadata(&self) -> Result<XorFilterMeta>;

    /// Reads a XOR filter with the given location.
    async fn xor_filter(&self, loc: &XorFilterLoc) -> Result<XorFilter>;

    /// Reads multiple XOR filters with the given locations.
    async fn xor_filter_vec(&self, locs: &[XorFilterLoc]) -> Result<Vec<XorFilter>>;
}

/// `XorFilterReaderImpl` reads the XOR filter from the file.
/// 
/// This implementation delegates to the common FilterReaderImpl for shared functionality.
pub struct XorFilterReaderImpl<R: RangeReader> {
    /// The underlying generic reader.
    inner: FilterReaderImpl<R, XorFilter>,
}

impl<R: RangeReader> XorFilterReaderImpl<R> {
    /// Creates a new `XorFilterReaderImpl` with the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            inner: FilterReaderImpl::new(reader),
        }
    }
}

#[async_trait]
impl<R: RangeReader> XorFilterReader for XorFilterReaderImpl<R> {
    async fn range_read(&self, offset: u64, size: u32) -> Result<Bytes> {
        self.inner
            .range_read(offset, size)
            .await
            .map_err(convert_common_error)
    }

    async fn read_vec(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.inner
            .read_vec(ranges)
            .await
            .map_err(convert_common_error)
    }

    async fn metadata(&self) -> Result<XorFilterMeta> {
        self.inner
            .metadata()
            .await
            .map_err(convert_common_error)
    }

    async fn xor_filter(&self, loc: &XorFilterLoc) -> Result<XorFilter> {
        self.inner
            .filter(loc)
            .await
            .map_err(convert_common_error)
    }

    async fn xor_filter_vec(&self, locs: &[XorFilterLoc]) -> Result<Vec<XorFilter>> {
        self.inner
            .filter_vec(locs)
            .await
            .map_err(convert_common_error)
    }
}

/// Helper function to convert common filter errors to XOR filter errors
fn convert_common_error(e: crate::common::CommonFilterError) -> Error {
    match e {
        crate::common::CommonFilterError::Io { error, location } => Error::Io { error, location },
        crate::common::CommonFilterError::FileSizeTooSmall { size, location } => {
            Error::FileSizeTooSmall { size, location }
        }
        crate::common::CommonFilterError::UnexpectedMetaSize { max_meta_size, actual_meta_size, location } => {
            Error::UnexpectedMetaSize { max_meta_size, actual_meta_size, location }
        }
        crate::common::CommonFilterError::DecodeProto { error, location } => {
            Error::DecodeProto { error, location }
        }
        crate::common::CommonFilterError::InvalidIntermediateMagic { invalid, location } => {
            Error::InvalidIntermediateMagic { invalid, location }
        }
        crate::common::CommonFilterError::Intermediate { source, location } => {
            Error::Intermediate { source, location }
        }
        crate::common::CommonFilterError::External { source, location } => {
            Error::External { source, location }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use futures::io::Cursor;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::xor_filter::creator::{XorFilterCreator, XorFilterSegmentBuilder};

    async fn mock_xor_filter_bytes() -> Vec<u8> {
        let mut writer = Cursor::new(vec![]);
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

        // Segment 3
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![5, 6]).unwrap();
        creator.add_segment(segment_builder).await.unwrap();

        creator.finish(&mut writer).await.unwrap();

        writer.into_inner()
    }

    #[tokio::test]
    async fn test_xor_filter_meta_reader() {
        let bytes = mock_xor_filter_bytes().await;

        // Test the metadata reading functionality through XorFilterReaderImpl
        let reader = XorFilterReaderImpl::new(bytes);
        let meta = reader.metadata().await.unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 3);
        assert_eq!(meta.row_count, 3);
        assert_eq!(meta.bloom_filter_locs.len(), 3);

        assert_eq!(meta.bloom_filter_locs[0].offset, 0);
        assert_eq!(meta.bloom_filter_locs[0].element_count, 2);
        assert_eq!(
            meta.bloom_filter_locs[1].offset,
            meta.bloom_filter_locs[0].size
        );
        assert_eq!(meta.bloom_filter_locs[1].element_count, 2);
    }

    #[tokio::test]
    async fn test_xor_filter_reader() {
        let bytes = mock_xor_filter_bytes().await;

        let reader = XorFilterReaderImpl::new(bytes);
        let meta = reader.metadata().await.unwrap();

        assert_eq!(meta.bloom_filter_locs.len(), 3);
        let xf = reader.xor_filter(&meta.bloom_filter_locs[0]).await.unwrap();
        assert!(xf.contains(1));
        assert!(xf.contains(2));

        let xf = reader.xor_filter(&meta.bloom_filter_locs[1]).await.unwrap();
        assert!(xf.contains(3));
        assert!(xf.contains(4));

        let xf = reader.xor_filter(&meta.bloom_filter_locs[2]).await.unwrap();
        assert!(xf.contains(5));
        assert!(xf.contains(6));
    }
}
