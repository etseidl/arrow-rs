// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Cache providers for decoded Parquet Bloom filters.

use crate::bloom_filter::Sbbf;
use crate::errors::{ParquetError, Result};
use crate::file::metadata::column_chunk::{ColumnChunkEntries, ColumnChunkMap};
use crate::file::metadata::memory::HeapSize;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

/// State of one valid coordinate in a [`BloomFilterProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloomFilterStatus<'a> {
    /// A decoded filter is available.
    Present(&'a Sbbf),
    /// The coordinate was loaded and the file metadata says it has no filter.
    Absent,
    /// The provider does not cover this coordinate.
    NotLoaded,
}

/// Provides already-loaded Bloom filters.
///
/// [`BloomFilterStatus::Present`] is a cache hit, [`BloomFilterStatus::Absent`] is a negative-cache
/// hit, and only [`BloomFilterStatus::NotLoaded`] permits a caller to perform fallback I/O.
/// Out-of-bounds lookups return an error and must not trigger fallback I/O.
///
/// Callers that may perform fallback I/O must use [`Self::status`], because [`Self::bloom_filter`]
/// intentionally collapses the two valid non-present states, `Absent` and `NotLoaded`, to `None`
/// while still propagating lookup errors.
pub trait BloomFilterProvider: Send + Sync + Debug {
    /// Returns the cache state for a row-group and column coordinate.
    fn status(&self, row_group_idx: usize, column_idx: usize) -> Result<BloomFilterStatus<'_>>;

    /// Returns a cached filter, collapsing valid non-present states to `None`.
    ///
    /// Use [`Self::status`] instead when `NotLoaded` should trigger fallback I/O.
    fn bloom_filter(&self, row_group_idx: usize, column_idx: usize) -> Result<Option<&Sbbf>> {
        Ok(match self.status(row_group_idx, column_idx)? {
            BloomFilterStatus::Present(filter) => Some(filter),
            BloomFilterStatus::Absent | BloomFilterStatus::NotLoaded => None,
        })
    }

    /// Returns whether this provider contains at least one decoded filter.
    fn has_bloom_filters(&self) -> bool;

    /// Returns this provider as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// Overlays newly loaded entries on an existing provider without cloning either cache.
#[derive(Debug)]
struct CompositeBloomFilterProvider {
    loaded: Arc<dyn BloomFilterProvider>,
    existing: Arc<dyn BloomFilterProvider>,
}

impl BloomFilterProvider for CompositeBloomFilterProvider {
    fn status(&self, row_group_idx: usize, column_idx: usize) -> Result<BloomFilterStatus<'_>> {
        match self.loaded.status(row_group_idx, column_idx)? {
            BloomFilterStatus::NotLoaded => self.existing.status(row_group_idx, column_idx),
            status => Ok(status),
        }
    }

    fn has_bloom_filters(&self) -> bool {
        self.loaded.has_bloom_filters() || self.existing.has_bloom_filters()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn merge_bloom_filter_provider(
    existing: Option<Arc<dyn BloomFilterProvider>>,
    loaded: BloomFilterIndex,
) -> Arc<dyn BloomFilterProvider> {
    let loaded: Arc<dyn BloomFilterProvider> = Arc::new(loaded);
    match existing {
        Some(existing) => Arc::new(CompositeBloomFilterProvider { loaded, existing }),
        None => loaded,
    }
}

#[derive(Debug, PartialEq)]
enum BloomFilterEntry {
    Absent,
    Present(Sbbf),
}

impl HeapSize for BloomFilterEntry {
    fn heap_size(&self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Present(filter) => filter.heap_size(),
        }
    }
}

/// Immutable sparse cache of decoded Bloom filters and known absences.
#[derive(Debug, Clone, PartialEq)]
pub struct BloomFilterIndex {
    entries: ColumnChunkEntries<BloomFilterEntry>,
}

impl BloomFilterIndex {
    /// Number of covered coordinates, including known absences.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Number of covered coordinates containing a filter.
    pub fn num_filters(&self) -> usize {
        self.entries
            .iter()
            .filter(|(_, entry)| matches!(entry, BloomFilterEntry::Present(_)))
            .count()
    }

    /// Returns whether no coordinates are covered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns whether every coordinate in the declared dimensions is covered.
    pub fn is_complete(&self) -> Result<bool> {
        let expected = self
            .num_row_groups()
            .checked_mul(self.num_columns())
            .ok_or_else(|| {
                ParquetError::General(format!(
                    "Bloom filter index dimension product overflow: {} by {}",
                    self.num_row_groups(),
                    self.num_columns()
                ))
            })?;
        Ok(self.len() == expected)
    }

    /// Declared number of row groups.
    pub fn num_row_groups(&self) -> usize {
        self.entries.num_row_groups()
    }

    /// Declared number of columns.
    pub fn num_columns(&self) -> usize {
        self.entries.num_columns()
    }

    /// Iterates over decoded filters in deterministic row-major order.
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &Sbbf)> {
        self.entries.iter().filter_map(|(key, entry)| match entry {
            BloomFilterEntry::Present(filter) => Some((key.coordinates(), filter)),
            BloomFilterEntry::Absent => None,
        })
    }

    /// Iterates over all covered entries in deterministic row-major order.
    pub fn covered_entries(&self) -> impl Iterator<Item = ((usize, usize), BloomFilterStatus<'_>)> {
        self.entries.iter().map(|(key, entry)| {
            let status = match entry {
                BloomFilterEntry::Present(filter) => BloomFilterStatus::Present(filter),
                BloomFilterEntry::Absent => BloomFilterStatus::Absent,
            };
            (key.coordinates(), status)
        })
    }
}

impl BloomFilterProvider for BloomFilterIndex {
    fn status(&self, row_group_idx: usize, column_idx: usize) -> Result<BloomFilterStatus<'_>> {
        Ok(match self.entries.get(row_group_idx, column_idx)? {
            Some(BloomFilterEntry::Present(filter)) => BloomFilterStatus::Present(filter),
            Some(BloomFilterEntry::Absent) => BloomFilterStatus::Absent,
            None => BloomFilterStatus::NotLoaded,
        })
    }

    fn has_bloom_filters(&self) -> bool {
        self.num_filters() != 0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl HeapSize for BloomFilterIndex {
    fn heap_size(&self) -> usize {
        self.entries.heap_size()
    }
}

/// Mutable builder for a [`BloomFilterIndex`].
#[derive(Debug)]
pub struct BloomFilterIndexBuilder {
    entries: ColumnChunkMap<BloomFilterEntry>,
    absent: Arc<BloomFilterEntry>,
}

impl BloomFilterIndexBuilder {
    /// Creates a builder with validated logical dimensions.
    ///
    /// No storage is allocated for individual coordinates until insertion.
    pub fn try_new(num_row_groups: usize, num_columns: usize) -> Result<Self> {
        Ok(Self {
            entries: ColumnChunkMap::try_new(num_row_groups, num_columns)?,
            absent: Arc::new(BloomFilterEntry::Absent),
        })
    }

    /// Inserts a decoded filter, rejecting duplicates and out-of-bounds coordinates.
    pub fn put_filter(
        &mut self,
        filter: Sbbf,
        row_group_idx: usize,
        column_idx: usize,
    ) -> Result<&mut Self> {
        self.entries.try_insert_vacant(
            row_group_idx,
            column_idx,
            Arc::new(BloomFilterEntry::Present(filter)),
        )?;
        Ok(self)
    }

    /// Records a known absence, rejecting duplicates and out-of-bounds coordinates.
    pub fn put_absent(&mut self, row_group_idx: usize, column_idx: usize) -> Result<&mut Self> {
        self.entries
            .try_insert_vacant(row_group_idx, column_idx, Arc::clone(&self.absent))?;
        Ok(self)
    }

    /// Fallibly freezes this builder into immutable row-major storage.
    pub fn try_build(self) -> Result<BloomFilterIndex> {
        // Validate arithmetic here as well so finalization cannot produce an index whose complete
        // coverage count cannot be represented.
        self.entries
            .num_row_groups()
            .checked_mul(self.entries.num_columns())
            .ok_or_else(|| {
                ParquetError::General(format!(
                    "Bloom filter index dimension product overflow: {} by {}",
                    self.entries.num_row_groups(),
                    self.entries.num_columns()
                ))
            })?;
        Ok(BloomFilterIndex {
            entries: self.entries.try_freeze()?,
        })
    }
}

/// Row-group-scoped view of an optional Bloom filter provider.
#[derive(Debug, Clone)]
pub struct RowGroupBloomFilters {
    provider: Option<Arc<dyn BloomFilterProvider>>,
    row_group_idx: usize,
    num_columns: usize,
}

impl RowGroupBloomFilters {
    /// Constructs an accessor for an already-validated row-group index.
    pub(crate) fn new(
        row_group_idx: usize,
        num_columns: usize,
        provider: Option<Arc<dyn BloomFilterProvider>>,
    ) -> Self {
        Self {
            provider,
            row_group_idx,
            num_columns,
        }
    }

    fn validate_column(&self, column_idx: usize) -> Result<()> {
        if column_idx >= self.num_columns {
            return Err(ParquetError::General(format!(
                "Bloom filter column index out of bounds: {column_idx}; row group {} has {} columns",
                self.row_group_idx, self.num_columns
            )));
        }
        Ok(())
    }

    /// Returns the cache status for a column in this row group.
    pub fn status(&self, column_idx: usize) -> Result<BloomFilterStatus<'_>> {
        self.validate_column(column_idx)?;
        match &self.provider {
            Some(provider) => provider.status(self.row_group_idx, column_idx),
            None => Ok(BloomFilterStatus::NotLoaded),
        }
    }

    /// Returns a decoded filter, collapsing valid non-present states to `None`.
    ///
    /// Lookup errors are propagated. Use [`Self::status`] instead when `NotLoaded` should trigger
    /// fallback I/O.
    pub fn bloom_filter(&self, column_idx: usize) -> Result<Option<&Sbbf>> {
        Ok(match self.status(column_idx)? {
            BloomFilterStatus::Present(filter) => Some(filter),
            BloomFilterStatus::Absent | BloomFilterStatus::NotLoaded => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[derive(Debug)]
    struct CustomProvider(Sbbf);

    impl BloomFilterProvider for CustomProvider {
        fn status(&self, row_group_idx: usize, column_idx: usize) -> Result<BloomFilterStatus<'_>> {
            match (row_group_idx, column_idx) {
                (0, 0) => Ok(BloomFilterStatus::Present(&self.0)),
                (0, 1) => Ok(BloomFilterStatus::NotLoaded),
                _ => Err(ParquetError::General(
                    "test coordinate out of bounds".into(),
                )),
            }
        }

        fn has_bloom_filters(&self) -> bool {
            true
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn filter(seed: u8) -> Sbbf {
        let mut bytes = vec![0; 32];
        bytes[0] = seed;
        Sbbf::try_from_bitset(&bytes).unwrap()
    }

    #[test]
    fn present_absent_not_loaded_and_bounds() {
        let present = filter(1);
        let mut builder = BloomFilterIndexBuilder::try_new(2, 3).unwrap();
        builder.put_filter(present.clone(), 0, 2).unwrap();
        builder.put_absent(1, 0).unwrap();
        let index = builder.try_build().unwrap();

        assert_eq!(
            index.status(0, 2).unwrap(),
            BloomFilterStatus::Present(&present)
        );
        assert_eq!(index.status(1, 0).unwrap(), BloomFilterStatus::Absent);
        assert_eq!(index.status(0, 0).unwrap(), BloomFilterStatus::NotLoaded);
        assert_eq!(index.bloom_filter(1, 0).unwrap(), None);
        assert!(index.status(2, 0).is_err());
        assert!(index.status(0, 3).is_err());
        assert_eq!(index.len(), 2);
        assert_eq!(index.num_filters(), 1);
        assert!(!index.is_complete().unwrap());
    }

    #[test]
    fn complete_and_zero_dimensions() {
        let mut builder = BloomFilterIndexBuilder::try_new(1, 2).unwrap();
        builder.put_absent(0, 0).unwrap().put_absent(0, 1).unwrap();
        assert!(builder.try_build().unwrap().is_complete().unwrap());

        let empty = BloomFilterIndexBuilder::try_new(0, 0)
            .unwrap()
            .try_build()
            .unwrap();
        assert!(empty.is_empty());
        assert!(empty.is_complete().unwrap());
    }

    #[test]
    fn duplicates_dimensions_and_bounds_return_errors_without_panicking() {
        let result = catch_unwind(|| BloomFilterIndexBuilder::try_new(i32::MAX as usize + 1, 1));
        assert!(result.unwrap().is_err());

        let mut builder = BloomFilterIndexBuilder::try_new(1, 1).unwrap();
        builder.put_absent(0, 0).unwrap();
        let duplicate = catch_unwind(AssertUnwindSafe(|| builder.put_absent(0, 0).map(|_| ())));
        assert!(duplicate.unwrap().is_err());
        let bounds = catch_unwind(AssertUnwindSafe(|| builder.put_absent(1, 0).map(|_| ())));
        assert!(bounds.unwrap().is_err());
    }

    #[test]
    fn equality_deterministic_iteration_and_freeze_without_cloning_filter() {
        let make = |reverse: bool| {
            let mut builder = BloomFilterIndexBuilder::try_new(2, 2).unwrap();
            if reverse {
                builder.put_filter(filter(2), 1, 1).unwrap();
                builder.put_absent(0, 1).unwrap();
            } else {
                builder.put_absent(0, 1).unwrap();
                builder.put_filter(filter(2), 1, 1).unwrap();
            }
            builder.try_build().unwrap()
        };
        let left = make(false);
        let right = make(true);
        // BloomFilterEntry is intentionally not Clone: freezing moves the Sbbf-owned entry into
        // immutable storage rather than cloning the filter.
        assert_eq!(left, right);
        assert_eq!(
            left.covered_entries()
                .map(|(coordinate, _)| coordinate)
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)]
        );
        assert_eq!(
            left.iter()
                .map(|(coordinate, _)| coordinate)
                .collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }

    #[test]
    fn row_group_accessor_validates_without_provider() {
        let accessor = RowGroupBloomFilters::new(3, 2, None);
        assert_eq!(accessor.status(1).unwrap(), BloomFilterStatus::NotLoaded);
        assert_eq!(accessor.bloom_filter(0).unwrap(), None);
        assert!(accessor.status(2).is_err());
    }

    #[test]
    fn row_group_accessor_reads_provider() {
        let expected = filter(7);
        let mut builder = BloomFilterIndexBuilder::try_new(2, 2).unwrap();
        builder.put_filter(expected.clone(), 1, 0).unwrap();
        builder.put_absent(1, 1).unwrap();
        let provider: Arc<dyn BloomFilterProvider> = Arc::new(builder.try_build().unwrap());
        let accessor = RowGroupBloomFilters::new(1, 2, Some(provider));
        assert_eq!(accessor.bloom_filter(0).unwrap(), Some(&expected));
        assert_eq!(accessor.status(1).unwrap(), BloomFilterStatus::Absent);
    }

    #[test]
    fn heap_size_grows_with_filters() {
        let empty = BloomFilterIndexBuilder::try_new(1, 2)
            .unwrap()
            .try_build()
            .unwrap();
        let mut absent_builder = BloomFilterIndexBuilder::try_new(1, 2).unwrap();
        absent_builder.put_absent(0, 0).unwrap();
        let absent = absent_builder.try_build().unwrap();
        let mut present_builder = BloomFilterIndexBuilder::try_new(1, 2).unwrap();
        present_builder.put_filter(filter(1), 0, 0).unwrap();
        let present = present_builder.try_build().unwrap();

        assert!(absent.heap_size() > empty.heap_size());
        assert!(present.heap_size() > absent.heap_size());
    }

    #[test]
    fn loaded_entries_overlay_custom_provider() {
        let expected = filter(9);
        let existing: Arc<dyn BloomFilterProvider> = Arc::new(CustomProvider(expected.clone()));
        let mut loaded = BloomFilterIndexBuilder::try_new(1, 2).unwrap();
        loaded.put_absent(0, 1).unwrap();
        let merged = merge_bloom_filter_provider(Some(existing), loaded.try_build().unwrap());

        assert_eq!(
            merged.status(0, 0).unwrap(),
            BloomFilterStatus::Present(&expected)
        );
        assert_eq!(merged.status(0, 1).unwrap(), BloomFilterStatus::Absent);
    }
}
