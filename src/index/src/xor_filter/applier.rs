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

use greptime_proto::v1::index::XorFilterMeta;
use itertools::Itertools;

use crate::xor_filter::error::Result;
use crate::xor_filter::reader::XorFilterReader;

pub struct XorFilterApplier {
    reader: Box<dyn XorFilterReader + Send>,
    meta: XorFilterMeta,
}

impl XorFilterApplier {
    pub async fn new(reader: Box<dyn XorFilterReader + Send>) -> Result<Self> {
        let meta = reader.metadata().await?;

        Ok(Self { reader, meta })
    }

    /// Searches ranges of rows that match the given keys in the given search range.
    pub async fn search(
        &mut self,
        keys: &[u64],
        search_range: Range<usize>,
    ) -> Result<Vec<Range<usize>>> {
        let rows_per_segment = self.meta.rows_per_segment as usize;
        let start_seg = search_range.start / rows_per_segment;
        let end_seg = search_range.end.div_ceil(rows_per_segment);

        let locs = &self.meta.segment_loc_indices[start_seg..end_seg];

        // dedup locs
        let deduped_locs = locs
            .iter()
            .dedup()
            .map(|i| self.meta.xor_filter_locs[*i as usize])
            .collect::<Vec<_>>();
        let xfs = self.reader.xor_filter_vec(&deduped_locs).await?;

        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(xfs.len());
        for ((_, mut group), filter) in locs
            .iter()
            .zip(start_seg..end_seg)
            .group_by(|(x, _)| **x)
            .into_iter()
            .zip(xfs.iter())
        {
            let start = group.next().unwrap().1 * rows_per_segment; // SAFETY: group is not empty
            let end = group.last().map_or(start + rows_per_segment, |(_, end)| {
                (end + 1) * rows_per_segment
            });
            let actual_start = start.max(search_range.start);
            let actual_end = end.min(search_range.end);
            for &key in keys {
                if filter.contains(key) {
                    match ranges.last_mut() {
                        Some(last) if last.end == actual_start => {
                            last.end = actual_end;
                        }
                        _ => {
                            ranges.push(actual_start..actual_end);
                        }
                    }
                    break;
                }
            }
        }

        Ok(ranges)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use arrow_array::UInt64Array;
    use futures::io::Cursor;

    use super::*;
    use crate::external_provider::MockExternalTempFileProvider;
    use crate::value_hasher::IntegerValueHasher;
    use crate::xor_filter::creator::{XorFilterCreator, XorFilterSegmentBuilder};
    use crate::xor_filter::reader::XorFilterReaderImpl;

    #[tokio::test]
    #[allow(clippy::single_range_in_vec_init)]
    async fn test_applier() {
        let mut writer = Cursor::new(Vec::new());
        let mut creator = XorFilterCreator::new(
            4,
            Arc::new(MockExternalTempFileProvider::new()),
            Arc::new(AtomicUsize::new(0)),
            None,
        );

        let hasher = IntegerValueHasher::new();

        // Segment 0: values 1-4
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![1, 2, 3, 4]).unwrap();
        creator.add_segment(segment_builder).unwrap();

        // Segment 1: values 5-8
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![5, 6, 7, 8]).unwrap();
        creator.add_segment(segment_builder).unwrap();

        // Segment 2: values 9-12
        let mut segment_builder = XorFilterSegmentBuilder::new();
        segment_builder.add_keys(vec![9, 10, 11, 12]).unwrap();
        creator.add_segment(segment_builder).unwrap();

        // Duplicate segments with value 42
        for _ in 0..4 {
            let mut segment_builder = XorFilterSegmentBuilder::new();
            segment_builder.add_key(42).unwrap();
            creator.add_segment(segment_builder).unwrap();
        }

        creator.finish(&mut writer).await.unwrap();

        let bytes = writer.into_inner();
        let reader = XorFilterReaderImpl::new(bytes);

        let mut applier = XorFilterApplier::new(Box::new(reader)).await.unwrap();

        let cases = vec![
            (vec![1], 0..28, vec![0..4]),            // search one key in full range
            (vec![6], 4..8, vec![4..8]),             // search one key in partial range
            (vec![3], 4..8, vec![]), // search for a key that doesn't exist in the partial range
            (vec![2, 7], 0..28, vec![0..8]), // search multiple keys in multiple ranges
            (vec![2, 11], 0..28, vec![0..4, 8..12]), // search multiple keys in multiple ranges
            (vec![99], 0..28, vec![]), // search for a key that doesn't exist in the full range
            (vec![1], 12..12, vec![]), // search in an empty range
            (vec![5, 6], 0..12, vec![4..8]), // search multiple keys in same segment
            (vec![42], 0..12, vec![]), // search for a duplicate key not in the range
            (vec![42], 0..16, vec![12..16]), // search for a duplicate key in the range
            (vec![42], 0..28, vec![12..28]), // search for a duplicate key in the full range
        ];

        for (keys, search_range, expected) in cases {
            let ranges = applier.search(&keys, search_range).await.unwrap();
            assert_eq!(ranges, expected);
        }
    }
}
