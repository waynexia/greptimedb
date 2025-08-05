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

//! Generic reader framework for probabilistic filters

use std::marker::PhantomData;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use common_base::range_read::RangeReader;
use greptime_proto::v1::index::{BloomFilterLoc as FilterLoc, BloomFilterMeta as FilterMeta};
use prost::Message;
use snafu::{ensure, ResultExt};

use crate::common::error::{CommonResult, DecodeProtoSnafu, FileSizeTooSmallSnafu, IoSnafu, UnexpectedMetaSizeSnafu};
use crate::common::filter::ProbabilisticFilter;

/// Minimum size of the filter metadata (4 bytes for length)
const FILTER_META_LEN_SIZE: u64 = 4;

/// Default prefetch size for filter metadata
pub const DEFAULT_PREFETCH_SIZE: u64 = 8192; // 8KiB

/// Generic trait for reading probabilistic filters from storage
#[async_trait]
pub trait FilterReader<F: ProbabilisticFilter>: Sync {
    /// Reads range of bytes from the file.
    async fn range_read(&self, offset: u64, size: u32) -> CommonResult<Bytes>;

    /// Reads multiple ranges from the file.
    async fn read_vec(&self, ranges: &[Range<u64>]) -> CommonResult<Vec<Bytes>> {
        let mut results = Vec::with_capacity(ranges.len());
        for range in ranges {
            let size = (range.end - range.start) as u32;
            let data = self.range_read(range.start, size).await?;
            results.push(data);
        }
        Ok(results)
    }

    /// Reads the metadata information of the filter.
    async fn metadata(&self) -> CommonResult<FilterMeta>;

    /// Reads a single filter with the given location.
    async fn filter(&self, loc: &FilterLoc) -> CommonResult<F> {
        let bytes = self.range_read(loc.offset, loc.size as _).await?;
        F::deserialize(&bytes)
    }

    /// Reads multiple filters with the given locations.
    async fn filter_vec(&self, locs: &[FilterLoc]) -> CommonResult<Vec<F>> {
        let ranges = locs
            .iter()
            .map(|l| l.offset..l.offset + l.size)
            .collect::<Vec<_>>();
        let bss = self.read_vec(&ranges).await?;

        let mut result = Vec::with_capacity(bss.len());
        for bs in bss {
            let filter = F::deserialize(&bs)?;
            result.push(filter);
        }

        Ok(result)
    }
}

/// Generic implementation of FilterReader for any RangeReader
pub struct FilterReaderImpl<R: RangeReader, F: ProbabilisticFilter> {
    reader: R,
    _phantom: PhantomData<F>,
}

impl<R: RangeReader, F: ProbabilisticFilter> FilterReaderImpl<R, F> {
    /// Creates a new FilterReaderImpl with the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            _phantom: PhantomData,
        }
    }

    /// Custom filter deserialization that allows passing additional parameters.
    /// This is useful for bloom filters that need element_count for proper deserialization.
    pub async fn filter_with_params<Params, DeserializeFn>(
        &self,
        loc: &FilterLoc,
        params: Params,
        deserialize_fn: DeserializeFn,
    ) -> CommonResult<F>
    where
        DeserializeFn: FnOnce(&[u8], Params) -> CommonResult<F>,
    {
        let bytes = self.range_read(loc.offset, loc.size as u32).await?;
        deserialize_fn(&bytes, params)
    }

    /// Custom filter vector deserialization with parameters.
    pub async fn filter_vec_with_params<Params, ParamsFn, DeserializeFn>(
        &self,
        locs: &[FilterLoc],
        params_fn: ParamsFn,
        deserialize_fn: DeserializeFn,
    ) -> CommonResult<Vec<F>>
    where
        ParamsFn: Fn(&FilterLoc) -> Params,
        DeserializeFn: Fn(&[u8], Params) -> CommonResult<F>,
    {
        let ranges = locs
            .iter()
            .map(|l| l.offset..l.offset + l.size)
            .collect::<Vec<_>>();
        let bytes_vec = self.read_vec(&ranges).await?;

        let mut result = Vec::with_capacity(bytes_vec.len());
        for (bytes, loc) in bytes_vec.into_iter().zip(locs.iter()) {
            let params = params_fn(loc);
            let filter = deserialize_fn(&bytes, params)?;
            result.push(filter);
        }

        Ok(result)
    }
}

#[async_trait]
impl<R: RangeReader, F: ProbabilisticFilter> FilterReader<F> for FilterReaderImpl<R, F> {
    async fn range_read(&self, offset: u64, size: u32) -> CommonResult<Bytes> {
        self.reader
            .read(offset..offset + size as u64)
            .await
            .context(IoSnafu)
    }

    async fn read_vec(&self, ranges: &[Range<u64>]) -> CommonResult<Vec<Bytes>> {
        self.reader.read_vec(ranges).await.context(IoSnafu)
    }

    async fn metadata(&self) -> CommonResult<FilterMeta> {
        let metadata = self.reader.metadata().await.context(IoSnafu)?;
        let file_size = metadata.content_length;

        let mut meta_reader = FilterMetaReader::new(&self.reader, file_size, Some(DEFAULT_PREFETCH_SIZE));
        meta_reader.metadata().await
    }
}

/// Generic metadata reader for probabilistic filters
struct FilterMetaReader<R: RangeReader> {
    reader: R,
    file_size: u64,
    prefetch_size: u64,
}

impl<R: RangeReader> FilterMetaReader<R> {
    pub fn new(reader: R, file_size: u64, prefetch_size: Option<u64>) -> Self {
        Self {
            reader,
            file_size,
            prefetch_size: prefetch_size
                .unwrap_or(FILTER_META_LEN_SIZE)
                .max(FILTER_META_LEN_SIZE),
        }
    }

    /// Reads the metadata of the filter.
    ///
    /// It will first prefetch some bytes from the end of the file,
    /// then parse the metadata from the prefetch bytes.
    pub async fn metadata(&mut self) -> CommonResult<FilterMeta> {
        ensure!(
            self.file_size >= FILTER_META_LEN_SIZE,
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

        if length > suffix_len as u64 - FILTER_META_LEN_SIZE {
            let metadata_start = self.file_size - length - FILTER_META_LEN_SIZE;
            let meta = self
                .reader
                .read(metadata_start..self.file_size - FILTER_META_LEN_SIZE)
                .await
                .context(IoSnafu)?;
            FilterMeta::decode(meta).context(DecodeProtoSnafu)
        } else {
            let metadata_start = self.file_size - length - FILTER_META_LEN_SIZE - meta_start;
            let meta = &suffix[metadata_start as usize..suffix_len - FILTER_META_LEN_SIZE as usize];
            FilterMeta::decode(meta).context(DecodeProtoSnafu)
        }
    }

    fn read_tailing_four_bytes(suffix: &[u8]) -> CommonResult<[u8; 4]> {
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

    fn validate_meta_size(&self, length: u64) -> CommonResult<()> {
        let max_meta_size = self.file_size - FILTER_META_LEN_SIZE;
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