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
use greptime_proto::v1::index::{XorFilterLoc, XorFilterMeta};
use prost::Message;
use snafu::{ensure, ResultExt};

use crate::xor_filter::error::*;
use crate::xor_filter::XorFilter;

/// Minimum size of the XOR filter, which is the size of the length of the XOR filter.
const XOR_META_LEN_SIZE: u64 = 4;

/// Default prefetch size of XOR filter meta.
pub const DEFAULT_PREFETCH_SIZE: u64 = 8192; // 8KiB

/// `XorFilterReader` reads the XOR filter from the file.
#[async_trait]
pub trait XorFilterReader: Sync {
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

    /// Reads the meta information of the XOR filter.
    async fn metadata(&self) -> Result<XorFilterMeta>;

    /// Reads a XOR filter with the given location.
    async fn xor_filter(&self, loc: &XorFilterLoc) -> Result<XorFilter> {
        let bytes = self.range_read(loc.offset, loc.size as _).await?;
        XorFilter::deserialize(&bytes)
    }

    async fn xor_filter_vec(&self, locs: &[XorFilterLoc]) -> Result<Vec<XorFilter>> {
        let ranges = locs
            .iter()
            .map(|l| l.offset..l.offset + l.size)
            .collect::<Vec<_>>();
        let bss = self.read_vec(&ranges).await?;

        let mut result = Vec::with_capacity(bss.len());
        for (bs, _) in bss.into_iter().zip(locs.iter()) {
            let filter = XorFilter::deserialize(&bs)?;
            result.push(filter);
        }

        Ok(result)
    }
}

/// `XorFilterReaderImpl` reads the XOR filter from the file.
pub struct XorFilterReaderImpl<R: RangeReader> {
    /// The underlying reader.
    reader: R,
}

impl<R: RangeReader> XorFilterReaderImpl<R> {
    /// Creates a new `XorFilterReaderImpl` with the given reader.
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

#[async_trait]
impl<R: RangeReader> XorFilterReader for XorFilterReaderImpl<R> {
    async fn range_read(&self, offset: u64, size: u32) -> Result<Bytes> {
        self.reader
            .read(offset..offset + size as u64)
            .await
            .context(IoSnafu)
    }

    async fn read_vec(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.reader.read_vec(ranges).await.context(IoSnafu)
    }

    async fn metadata(&self) -> Result<XorFilterMeta> {
        let metadata = self.reader.metadata().await.context(IoSnafu)?;
        let file_size = metadata.content_length;

        let mut meta_reader =
            XorFilterMetaReader::new(&self.reader, file_size, Some(DEFAULT_PREFETCH_SIZE));
        meta_reader.metadata().await
    }
}

/// `XorFilterMetaReader` reads the metadata of the XOR filter.
struct XorFilterMetaReader<R: RangeReader> {
    reader: R,
    file_size: u64,
    prefetch_size: u64,
}

impl<R: RangeReader> XorFilterMetaReader<R> {
    pub fn new(reader: R, file_size: u64, prefetch_size: Option<u64>) -> Self {
        Self {
            reader,
            file_size,
            prefetch_size: prefetch_size
                .unwrap_or(XOR_META_LEN_SIZE)
                .max(XOR_META_LEN_SIZE),
        }
    }

    /// Reads the metadata of the XOR filter.
    ///
    /// It will first prefetch some bytes from the end of the file,
    /// then parse the metadata from the prefetch bytes.
    pub async fn metadata(&mut self) -> Result<XorFilterMeta> {
        ensure!(
            self.file_size >= XOR_META_LEN_SIZE,
            FileSizeTooSmallSnafu {
                size: self.file_size,
            }
        );

        let meta_start = self.file_size.saturating_sub(self.prefetch_size);
        let suffix = self
            .reader
            .read(meta_start..self.file_size)
            .await
            .context(IoSnafu)?;
        let suffix_len = suffix.len();
        let length = u32::from_le_bytes(Self::read_tailing_four_bytes(&suffix)?) as u64;
        self.validate_meta_size(length)?;

        if length > suffix_len as u64 - XOR_META_LEN_SIZE {
            let metadata_start = self.file_size - length - XOR_META_LEN_SIZE;
            let meta = self
                .reader
                .read(metadata_start..self.file_size - XOR_META_LEN_SIZE)
                .await
                .context(IoSnafu)?;
            XorFilterMeta::decode(meta).context(DecodeProtoSnafu)
        } else {
            let metadata_start = self.file_size - length - XOR_META_LEN_SIZE - meta_start;
            let meta = &suffix[metadata_start as usize..suffix_len - XOR_META_LEN_SIZE as usize];
            XorFilterMeta::decode(meta).context(DecodeProtoSnafu)
        }
    }

    fn read_tailing_four_bytes(suffix: &[u8]) -> Result<[u8; 4]> {
        let suffix_len = suffix.len();
        ensure!(
            suffix_len >= 4,
            FileSizeTooSmallSnafu {
                size: suffix_len as u64
            }
        );
        let mut bytes = [0; 4];
        bytes.copy_from_slice(&suffix[suffix_len - 4..suffix_len]);

        Ok(bytes)
    }

    fn validate_meta_size(&self, length: u64) -> Result<()> {
        let max_meta_size = self.file_size - XOR_META_LEN_SIZE;
        ensure!(
            length <= max_meta_size,
            UnexpectedMetaSizeSnafu {
                max_meta_size,
                actual_meta_size: length,
            }
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use futures::io::Cursor;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::value_hasher::IntegerValueHasher;
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
        creator.add_segment(segment_builder).unwrap();

        // Segment 2
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![3, 4]).unwrap();
        creator.add_segment(segment_builder).unwrap();

        // Segment 3
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![5, 6]).unwrap();
        creator.add_segment(segment_builder).unwrap();

        creator.finish(&mut writer).await.unwrap();

        writer.into_inner()
    }

    #[tokio::test]
    async fn test_xor_filter_meta_reader() {
        let bytes = mock_xor_filter_bytes().await;
        let file_size = bytes.len() as u64;

        for prefetch in [0u64, file_size / 2, file_size, file_size + 10] {
            let mut reader =
                XorFilterMetaReader::new(bytes.clone(), file_size as _, Some(prefetch));
            let meta = reader.metadata().await.unwrap();

            assert_eq!(meta.rows_per_segment, 2);
            assert_eq!(meta.segment_count, 3);
            assert_eq!(meta.row_count, 3);
            assert_eq!(meta.xor_filter_locs.len(), 3);

            assert_eq!(meta.xor_filter_locs[0].offset, 0);
            assert_eq!(meta.xor_filter_locs[0].element_count, 2);
            assert_eq!(meta.xor_filter_locs[1].offset, meta.xor_filter_locs[0].size);
            assert_eq!(meta.xor_filter_locs[1].element_count, 2);
        }
    }

    #[tokio::test]
    async fn test_xor_filter_reader() {
        let bytes = mock_xor_filter_bytes().await;

        let reader = XorFilterReaderImpl::new(bytes);
        let meta = reader.metadata().await.unwrap();

        assert_eq!(meta.xor_filter_locs.len(), 3);
        let xf = reader.xor_filter(&meta.xor_filter_locs[0]).await.unwrap();
        assert!(xf.contains(1));
        assert!(xf.contains(2));

        let xf = reader.xor_filter(&meta.xor_filter_locs[1]).await.unwrap();
        assert!(xf.contains(3));
        assert!(xf.contains(4));

        let xf = reader.xor_filter(&meta.xor_filter_locs[2]).await.unwrap();
        assert!(xf.contains(5));
        assert!(xf.contains(6));
    }
}
