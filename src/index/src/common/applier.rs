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

use itertools::Itertools;

use crate::common::{CommonResult, ProbabilisticFilter};

/// Common trait for filter appliers that search for matching row ranges.
pub trait FilterApplier {
    /// The type of queries this applier handles.
    type Query;

    /// Search for row ranges that match the given query within the search ranges.  
    async fn search(
        &mut self,
        query: &Self::Query,
        search_ranges: &[Range<usize>],
    ) -> CommonResult<Vec<Range<usize>>>;
}

/// Convert row ranges to unique segment indices.
pub fn row_ranges_to_segments(
    row_ranges: &[Range<usize>],
    rows_per_segment: usize,
    segment_count: usize,
) -> Vec<usize> {
    let mut segments = Vec::new();

    for range in row_ranges {
        let start_seg = range.start / rows_per_segment;
        let mut end_seg = range.end.div_ceil(rows_per_segment);

        // Handle legacy bug with missing last segment
        if end_seg == segment_count + 1 {
            end_seg -= 1;
        }

        segments.extend(start_seg..end_seg);
    }

    // Ensure segments are unique and sorted
    segments.sort_unstable();
    segments.dedup();
    segments
}

/// Convert matching segments to row ranges and merge adjacent ones.
pub fn segments_to_row_ranges(
    matching_segments: &[usize],
    rows_per_segment: usize,
) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = matching_segments
        .iter()
        .map(|&segment| {
            let start_row = segment * rows_per_segment;
            let end_row = (segment + 1) * rows_per_segment;
            start_row..end_row
        })
        .collect();

    // Sort ranges by start position
    ranges.sort_by_key(|r| r.start);

    // Merge adjacent ranges
    merge_adjacent_ranges(ranges)
}

/// Merge adjacent row ranges to reduce the number of ranges.
pub fn merge_adjacent_ranges(ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges
        .into_iter()
        .coalesce(|prev, next| {
            if prev.end == next.start {
                Ok(prev.start..next.end)
            } else {
                Err((prev, next))
            }
        })
        .collect()
}

/// Intersects two lists of ranges and returns the intersection.
/// The input lists are assumed to be sorted and non-overlapping.
pub fn intersect_ranges(lhs: &[Range<usize>], rhs: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut i = 0;
    let mut j = 0;
    let mut output = Vec::new();

    while i < lhs.len() && j < rhs.len() {
        let r1 = &lhs[i];
        let r2 = &rhs[j];

        // Find intersection if exists
        let start = r1.start.max(r2.start);
        let end = r1.end.min(r2.end);

        if start < end {
            output.push(start..end);
        }

        // Move forward the range that ends first
        if r1.end < r2.end {
            i += 1;
        } else {
            j += 1;
        }
    }

    output
}

/// A predicate that checks if any of the provided keys match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrPredicate {
    /// Keys to check (OR semantics - any key matching satisfies the predicate).
    pub keys: Vec<u64>,
}

impl OrPredicate {
    /// Create a new OR predicate.
    pub fn new(keys: Vec<u64>) -> Self {
        Self { keys }
    }

    /// Check if this predicate matches the given filter.
    pub fn matches<F: ProbabilisticFilter>(&self, filter: &F) -> bool {
        self.keys.iter().any(|&key| filter.contains(key))
    }
}

/// A predicate that checks if all of the provided sub-predicates match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AndPredicate {
    /// Sub-predicates that all must match (AND semantics).
    pub predicates: Vec<OrPredicate>,
}

impl AndPredicate {
    /// Create a new AND predicate.
    pub fn new(predicates: Vec<OrPredicate>) -> Self {
        Self { predicates }
    }

    /// Check if this predicate matches the given filter.
    pub fn matches<F: ProbabilisticFilter>(&self, filter: &F) -> bool {
        self.predicates.iter().all(|pred| pred.matches(filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_ranges() {
        // Empty inputs
        assert_eq!(intersect_ranges(&[], &[]), Vec::<Range<usize>>::new());
        assert_eq!(intersect_ranges(&[1..5], &[]), Vec::<Range<usize>>::new());
        assert_eq!(intersect_ranges(&[], &[1..5]), Vec::<Range<usize>>::new());

        // No overlap
        assert_eq!(
            intersect_ranges(&[1..3, 5..7], &[3..5, 7..9]),
            Vec::<Range<usize>>::new()
        );

        // Single overlap
        assert_eq!(intersect_ranges(&[1..5], &[3..7]), vec![3..5]);

        // Multiple overlaps
        assert_eq!(
            intersect_ranges(&[1..5, 7..10, 12..15], &[2..6, 8..13]),
            vec![2..5, 8..10, 12..13]
        );

        // Exact overlap
        assert_eq!(
            intersect_ranges(&[1..3, 5..7], &[1..3, 5..7]),
            vec![1..3, 5..7]
        );

        // Contained ranges
        assert_eq!(
            intersect_ranges(&[1..10], &[2..4, 5..7, 8..9]),
            vec![2..4, 5..7, 8..9]
        );

        // Partial overlaps
        assert_eq!(
            intersect_ranges(&[1..4, 6..9], &[2..7, 8..10]),
            vec![2..4, 6..7, 8..9]
        );

        // Single point overlap (no actual overlap)
        assert_eq!(
            intersect_ranges(&[1..3], &[3..5]),
            Vec::<Range<usize>>::new()
        );

        // Large ranges
        assert_eq!(intersect_ranges(&[0..100], &[50..150]), vec![50..100]);
    }
}
