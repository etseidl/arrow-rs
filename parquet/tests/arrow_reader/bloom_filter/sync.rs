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

use std::fs::File;

use bytes::Bytes;
use parquet::{
    arrow::{
        ArrowWriter,
        arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder},
    },
    file::properties::WriterProperties,
};

use super::{CountingChunkReader, cache_options, cached_bloom_filter_file};
use parquet::file::metadata::ColumnChunkMask;

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
