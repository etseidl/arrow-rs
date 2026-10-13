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

use super::{cache_options, cached_bloom_filter_file, malformed_bloom_filter_header};
use crate::io::TestReader;
use bytes::Bytes;
use futures::{FutureExt, TryStreamExt, future::BoxFuture};
use parquet::{
    arrow::{
        ArrowWriter, ParquetRecordBatchStreamBuilder,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder},
        async_reader::AsyncFileReader,
    },
    errors::{ParquetError, Result},
    file::metadata::{
        BloomFilterPolicy, BloomFilterStatus, ColumnChunkMask, ParquetMetaData,
        ParquetMetaDataBuilder, ParquetMetaDataReader,
    },
    file::properties::WriterProperties,
};
use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::external_cache_file;

#[derive(Clone)]
struct CountingMultiRangeReader {
    data: Bytes,
    single_requests: Arc<AtomicUsize>,
    multi_requests: Arc<AtomicUsize>,
}

impl CountingMultiRangeReader {
    fn new(data: Bytes) -> Self {
        Self {
            data,
            single_requests: Arc::new(AtomicUsize::new(0)),
            multi_requests: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn single_request_count(&self) -> usize {
        self.single_requests.load(Ordering::SeqCst)
    }

    fn multi_request_count(&self) -> usize {
        self.multi_requests.load(Ordering::SeqCst)
    }

    fn read_range(data: &Bytes, range: Range<u64>) -> Result<Bytes> {
        let start = usize::try_from(range.start)?;
        let end = usize::try_from(range.end)?;
        data.get(start..end)
            .map(Bytes::copy_from_slice)
            .ok_or_else(|| ParquetError::General(format!("invalid test range {start}..{end}")))
    }
}

impl AsyncFileReader for CountingMultiRangeReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        self.single_requests.fetch_add(1, Ordering::SeqCst);
        let result = Self::read_range(&self.data, range);
        futures::future::ready(result).boxed()
    }

    fn get_byte_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, Result<Vec<Bytes>>> {
        self.multi_requests.fetch_add(1, Ordering::SeqCst);
        let result = ranges
            .into_iter()
            .map(|range| Self::read_range(&self.data, range))
            .collect();
        futures::future::ready(result).boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
        let result = ParquetMetaDataReader::new()
            .with_arrow_reader_options(options)
            .parse_and_finish(&self.data)
            .map(Arc::new);
        futures::future::ready(result).boxed()
    }
}

#[tokio::test]
async fn test_malformed_filter_error_matches_sync_reader() {
    let (data, metadata) = malformed_bloom_filter_header();

    let sync_builder =
        ParquetRecordBatchReaderBuilder::new_with_metadata(data.clone(), metadata.clone());
    let sync_error = sync_builder
        .get_row_group_column_bloom_filter(0, 0)
        .unwrap_err()
        .to_string();

    let mut async_builder =
        ParquetRecordBatchStreamBuilder::new_with_metadata(TestReader::new(data), metadata);
    let async_error = async_builder
        .get_row_group_column_bloom_filter(0, 0)
        .await
        .unwrap_err()
        .to_string();

    assert!(
        sync_error.contains("negative"),
        "unexpected error: {sync_error}"
    );
    assert_eq!(async_error, sync_error);
}

#[tokio::test]
async fn test_get_row_group_column_bloom_filter_without_length() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/data_index_bloom_encoding_stats.parquet");
    let data = Bytes::from(std::fs::read(path).unwrap());
    test_get_row_group_column_bloom_filter(data, false).await;
}

#[tokio::test]
async fn test_get_row_group_column_bloom_filter_with_length() {
    // convert to new parquet file with bloom_filter_length
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/data_index_bloom_encoding_stats.parquet");
    let data = Bytes::from(std::fs::read(path).unwrap());
    let async_reader = TestReader::new(data.clone());
    let builder = ParquetRecordBatchStreamBuilder::new(async_reader)
        .await
        .unwrap();
    let schema = builder.schema().clone();
    let stream = builder.build().unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    let mut parquet_data = Vec::new();
    let props = WriterProperties::builder()
        .set_bloom_filter_enabled(true)
        .build();
    let mut writer = ArrowWriter::try_new(&mut parquet_data, schema, Some(props)).unwrap();
    for batch in batches {
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();

    // test the new parquet file
    test_get_row_group_column_bloom_filter(parquet_data.into(), true).await;
}

async fn test_get_row_group_column_bloom_filter(data: Bytes, with_length: bool) {
    let async_reader = TestReader::new(data.clone());

    let mut builder = ParquetRecordBatchStreamBuilder::new(async_reader)
        .await
        .unwrap();

    let metadata = builder.metadata();
    assert_eq!(metadata.num_row_groups(), 1);
    let row_group = metadata.row_group(0);
    let column = row_group.column(0);
    assert_eq!(column.bloom_filter_length().is_some(), with_length);

    let sbbf = builder
        .get_row_group_column_bloom_filter(0, 0)
        .await
        .unwrap()
        .unwrap();
    assert!(sbbf.check(&"Hello"));
    assert!(!sbbf.check(&"Hello_Not_Exists"));
}

#[tokio::test]
async fn test_cached_bloom_filter_status_controls_async_io() {
    let data = cached_bloom_filter_file();
    let metadata = ArrowReaderMetadata::load(&data, cache_options(ColumnChunkMask::all())).unwrap();
    let input = TestReader::new(data.clone());
    let requests = input.requests();
    let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(input, metadata);

    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 0)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 1)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        builder
            .get_row_group_column_bloom_filter(1, 0)
            .await
            .is_err()
    );
    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 2)
            .await
            .is_err()
    );
    assert!(requests.lock().unwrap().is_empty());

    let metadata =
        ArrowReaderMetadata::load(&data, cache_options(ColumnChunkMask::columns([1]))).unwrap();
    let input = TestReader::new(data);
    let requests = input.requests();
    let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(input, metadata);
    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 0)
            .await
            .unwrap()
            .is_some()
    );
    assert!(!requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn test_external_bloom_filter_cache_workflow_async() {
    let data = external_cache_file();
    let footer_metadata = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Skip)
        .parse_and_finish(&data)
        .unwrap();
    assert_eq!(footer_metadata.num_row_groups(), 2);

    let mut preload_source = CountingMultiRangeReader::new(data.clone());
    let (_metadata, external_cache) = ParquetMetaDataReader::new_with_metadata(footer_metadata)
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::row_groups([
            0,
        ])))
        .load_bloom_filter_and_finish(&mut preload_source)
        .await
        .unwrap();
    let external_cache = external_cache.unwrap();
    assert_eq!(preload_source.single_request_count(), 0);
    assert_eq!(preload_source.multi_request_count(), 1);

    let fresh_metadata = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Skip)
        .parse_and_finish(&data)
        .unwrap();
    let metadata = ParquetMetaDataBuilder::new_from_metadata(fresh_metadata)
        .set_bloom_filters(Some(Arc::new(external_cache)))
        .build();
    let provider = metadata.bloom_filters().unwrap();
    assert!(matches!(
        provider.status(0, 0).unwrap(),
        BloomFilterStatus::Present(_)
    ));
    assert_eq!(provider.status(0, 1).unwrap(), BloomFilterStatus::Absent);
    assert_eq!(provider.status(1, 0).unwrap(), BloomFilterStatus::NotLoaded);

    let arrow_metadata =
        ArrowReaderMetadata::try_new(Arc::new(metadata), ArrowReaderOptions::new()).unwrap();
    let input = CountingMultiRangeReader::new(data);
    let mut reader =
        ParquetRecordBatchStreamBuilder::new_with_metadata(input.clone(), arrow_metadata);
    assert!(
        reader
            .get_row_group_column_bloom_filter(0, 0)
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        reader
            .get_row_group_column_bloom_filter(0, 1)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(input.single_request_count(), 0);
    assert_eq!(input.multi_request_count(), 0);

    assert!(
        reader
            .get_row_group_column_bloom_filter(1, 0)
            .await
            .unwrap()
            .is_some()
    );
    assert!(input.single_request_count() > 0);
}
