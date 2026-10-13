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

use std::{fs::File, sync::Arc};

use bytes::Bytes;
use parquet::{
    arrow::{
        ArrowWriter,
        arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder},
    },
    file::metadata::{
        BloomFilterPolicy, BloomFilterStatus, ColumnChunkMask, ParquetMetaDataBuilder,
        ParquetMetaDataReader,
    },
    file::properties::WriterProperties,
};

use super::{CountingChunkReader, cache_options, cached_bloom_filter_file, external_cache_file};

#[test]
#[cfg_attr(miri, ignore)] // Takes too long
fn test_get_row_group_column_bloom_filter_with_length() {
    // convert to new parquet file with bloom_filter_length
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/data_index_bloom_encoding_stats.parquet");
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = builder.schema().clone();
    let reader = builder.build().unwrap();

    let mut parquet_data = Vec::new();
    let props = WriterProperties::builder()
        .set_bloom_filter_enabled(true)
        .build();
    let mut writer = ArrowWriter::try_new(&mut parquet_data, schema, Some(props)).unwrap();
    for batch in reader {
        let batch = batch.unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();

    // test the new parquet file
    test_get_row_group_column_bloom_filter(parquet_data.into(), true);
}

#[test]
fn test_get_row_group_column_bloom_filter_without_length() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/data_index_bloom_encoding_stats.parquet");
    let data = Bytes::from(std::fs::read(path).unwrap());
    test_get_row_group_column_bloom_filter(data, false);
}

fn test_get_row_group_column_bloom_filter(data: Bytes, with_length: bool) {
    let builder = ParquetRecordBatchReaderBuilder::try_new(data.clone()).unwrap();

    let metadata = builder.metadata();
    assert_eq!(metadata.num_row_groups(), 1);
    let row_group = metadata.row_group(0);
    let column = row_group.column(0);
    assert_eq!(column.bloom_filter_length().is_some(), with_length);

    let sbbf = builder
        .get_row_group_column_bloom_filter(0, 0)
        .unwrap()
        .unwrap();
    assert!(sbbf.check(&"Hello"));
    assert!(!sbbf.check(&"Hello_Not_Exists"));
}

#[test]
fn test_cached_bloom_filter_status_controls_sync_io() {
    let data = cached_bloom_filter_file();
    let metadata = ArrowReaderMetadata::load(&data, cache_options(ColumnChunkMask::all())).unwrap();
    let input = CountingChunkReader::new(data.clone());
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input.clone(), metadata);

    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 0)
            .unwrap()
            .is_some()
    );
    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 1)
            .unwrap()
            .is_none()
    );
    assert!(builder.get_row_group_column_bloom_filter(1, 0).is_err());
    assert!(builder.get_row_group_column_bloom_filter(0, 2).is_err());
    assert_eq!(input.request_count(), 0);

    let metadata =
        ArrowReaderMetadata::load(&data, cache_options(ColumnChunkMask::columns([1]))).unwrap();
    let input = CountingChunkReader::new(data);
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(input.clone(), metadata);
    assert!(
        builder
            .get_row_group_column_bloom_filter(0, 0)
            .unwrap()
            .is_some()
    );
    assert!(input.request_count() > 0);
}

#[test]
fn test_external_bloom_filter_cache_workflow() {
    let data = external_cache_file();
    let footer_metadata = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Skip)
        .parse_and_finish(&data)
        .unwrap();
    assert_eq!(footer_metadata.num_row_groups(), 2);

    let (_metadata, external_cache) = ParquetMetaDataReader::new_with_metadata(footer_metadata)
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::row_groups([
            0,
        ])))
        .read_bloom_filter_and_finish(&data)
        .unwrap();
    let external_cache = external_cache.unwrap();

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
    let input = CountingChunkReader::new(data);
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(input.clone(), arrow_metadata);
    assert!(
        reader
            .get_row_group_column_bloom_filter(0, 0)
            .unwrap()
            .is_some()
    );
    assert!(
        reader
            .get_row_group_column_bloom_filter(0, 1)
            .unwrap()
            .is_none()
    );
    assert_eq!(input.request_count(), 0);

    assert!(
        reader
            .get_row_group_column_bloom_filter(1, 0)
            .unwrap()
            .is_some()
    );
    assert!(input.request_count() > 0);
}
