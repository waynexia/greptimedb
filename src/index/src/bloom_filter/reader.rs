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
use greptime_proto::v1::index::{BloomFilterLoc, BloomFilterMeta};

use crate::bloom_filter::error::{Error, Result};
use crate::bloom_filter::BloomFilter;
use crate::common::reader::{FilterReader, FilterReaderImpl};

/// `BloomFilterReader` reads the bloom filter from the file.
#[async_trait]
pub trait BloomFilterReader: Sync {
    /// Reads range of bytes from the file.
    async fn range_read(&self, offset: u64, size: u32) -> Result<Bytes>;

    /// Reads bunch of ranges from the file.
    async fn read_vec(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let mut results = Vec::with_capacity(ranges.len());
        for range in ranges {
            let size = (range.end - range.start) as u32;
            let data = self.range_read(range.start, size).await?;
            results.push(data);
        }
        Ok(results)
    }

    /// Reads the meta information of the bloom filter.
    async fn metadata(&self) -> Result<BloomFilterMeta>;

    /// Reads a bloom filter with the given location.
    async fn bloom_filter(&self, loc: &BloomFilterLoc) -> Result<BloomFilter> {
        let bytes = self.range_read(loc.offset, loc.size as _).await?;
        let vec = bytes
            .chunks_exact(std::mem::size_of::<u64>())
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let bf = BloomFilter::from_vec(vec, loc.element_count as usize);
        Ok(bf)
    }

    async fn bloom_filter_vec(&self, locs: &[BloomFilterLoc]) -> Result<Vec<BloomFilter>> {
        let ranges = locs
            .iter()
            .map(|l| l.offset..l.offset + l.size)
            .collect::<Vec<_>>();
        let bss = self.read_vec(&ranges).await?;

        let mut result = Vec::with_capacity(bss.len());
        for (bs, loc) in bss.into_iter().zip(locs.iter()) {
            let vec = bs
                .chunks_exact(std::mem::size_of::<u64>())
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let bf = BloomFilter::from_vec(vec, loc.element_count as usize);
            result.push(bf);
        }

        Ok(result)
    }
}

/// `BloomFilterReaderImpl` reads the bloom filter from the file.
///
/// This implementation delegates to the common FilterReaderImpl for shared functionality
/// while maintaining backward compatibility with the existing BloomFilter API.
pub struct BloomFilterReaderImpl<R: RangeReader> {
    /// The underlying generic reader.
    inner: FilterReaderImpl<R, BloomFilter>,
}

impl<R: RangeReader> BloomFilterReaderImpl<R> {
    /// Creates a new `BloomFilterReaderImpl` with the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            inner: FilterReaderImpl::new(reader),
        }
    }
}

#[async_trait]
impl<R: RangeReader> BloomFilterReader for BloomFilterReaderImpl<R> {
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

    async fn metadata(&self) -> Result<BloomFilterMeta> {
        self.inner.metadata().await.map_err(convert_common_error)
    }

    async fn bloom_filter(&self, loc: &BloomFilterLoc) -> Result<BloomFilter> {
        // Use custom deserialization with element_count parameter
        self.inner
            .filter_with_params(loc, loc.element_count as usize, |bytes, element_count| {
                let vec = bytes
                    .chunks_exact(std::mem::size_of::<u64>())
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(BloomFilter::from_vec(vec, element_count))
            })
            .await
            .map_err(convert_common_error)
    }

    async fn bloom_filter_vec(&self, locs: &[BloomFilterLoc]) -> Result<Vec<BloomFilter>> {
        // Use custom vector deserialization with element_count parameter
        self.inner
            .filter_vec_with_params(
                locs,
                |loc| loc.element_count as usize,
                |bytes, element_count| {
                    let vec = bytes
                        .chunks_exact(std::mem::size_of::<u64>())
                        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    Ok(BloomFilter::from_vec(vec, element_count))
                },
            )
            .await
            .map_err(convert_common_error)
    }
}

/// Helper function to convert common filter errors to bloom filter errors
fn convert_common_error(e: crate::common::CommonFilterError) -> Error {
    match e {
        crate::common::CommonFilterError::Io { error, location } => Error::Io { error, location },
        crate::common::CommonFilterError::FileSizeTooSmall { size, location } => {
            Error::FileSizeTooSmall { size, location }
        }
        crate::common::CommonFilterError::UnexpectedMetaSize {
            max_meta_size,
            actual_meta_size,
            location,
        } => Error::UnexpectedMetaSize {
            max_meta_size,
            actual_meta_size,
            location,
        },
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
    use crate::bloom_filter::creator::BloomFilterCreator;
    use crate::external_provider::MockExternalTempFileProvider;

    async fn mock_bloom_filter_bytes() -> Vec<u8> {
        let mut writer = Cursor::new(vec![]);
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
        creator
            .push_row_elems(vec![b"c".to_vec(), b"d".to_vec()])
            .await
            .unwrap();
        creator
            .push_row_elems(vec![b"e".to_vec(), b"f".to_vec()])
            .await
            .unwrap();

        creator.finish(&mut writer).await.unwrap();

        writer.into_inner()
    }

    #[tokio::test]
    async fn test_bloom_filter_meta_reader() {
        let bytes = mock_bloom_filter_bytes().await;

        // Use BloomFilterReaderImpl instead of BloomFilterMetaReader
        let reader = BloomFilterReaderImpl::new(bytes);
        let meta = reader.metadata().await.unwrap();

        assert_eq!(meta.rows_per_segment, 2);
        assert_eq!(meta.segment_count, 2);
        assert_eq!(meta.row_count, 3);
        assert_eq!(meta.bloom_filter_locs.len(), 2);

        assert_eq!(meta.bloom_filter_locs[0].offset, 0);
        assert_eq!(meta.bloom_filter_locs[0].element_count, 4);
        assert_eq!(
            meta.bloom_filter_locs[1].offset,
            meta.bloom_filter_locs[0].size
        );
        assert_eq!(meta.bloom_filter_locs[1].element_count, 2);
    }

    #[tokio::test]
    async fn test_bloom_filter_reader() {
        let bytes = mock_bloom_filter_bytes().await;

        let reader = BloomFilterReaderImpl::new(bytes);
        let meta = reader.metadata().await.unwrap();

        assert_eq!(meta.bloom_filter_locs.len(), 2);
        let bf = reader
            .bloom_filter(&meta.bloom_filter_locs[0])
            .await
            .unwrap();
        assert!(bf.contains(&b"a"));
        assert!(bf.contains(&b"b"));
        assert!(bf.contains(&b"c"));
        assert!(bf.contains(&b"d"));

        let bf = reader
            .bloom_filter(&meta.bloom_filter_locs[1])
            .await
            .unwrap();
        assert!(bf.contains(&b"e"));
        assert!(bf.contains(&b"f"));
    }
}
