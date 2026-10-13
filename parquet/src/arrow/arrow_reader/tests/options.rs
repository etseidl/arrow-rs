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

//! Reader options for encoding statistics, size statistics, and page indexes.

use super::*;

fn bloom_filter_test_data() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowDataType::Int32, false),
        Field::new("b", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![4, 5, 6])),
        ],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_bloom_filter_enabled(true)
        .build();
    let mut data = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut data, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    data.into()
}

#[test]
fn test_bloom_filter_options_and_sync_propagation() {
    let default_options = ArrowReaderOptions::default();
    assert_eq!(
        default_options.bloom_filter_policy(),
        &BloomFilterPolicy::Skip
    );

    let policy = BloomFilterPolicy::Optional(ColumnChunkMask::columns([1]));
    let options = ArrowReaderOptions::new().with_bloom_filter_policy(policy.clone());
    assert_eq!(options.bloom_filter_policy(), &policy);

    let data = bloom_filter_test_data();
    let default_builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        data.clone(),
        ArrowReaderOptions::default(),
    )
    .unwrap();
    assert!(default_builder.metadata().bloom_filters().is_none());

    let builder =
        ParquetRecordBatchReaderBuilder::try_new_with_options(data.clone(), options).unwrap();
    let provider = builder.metadata().bloom_filters().unwrap();
    assert_eq!(provider.status(0, 0).unwrap(), BloomFilterStatus::NotLoaded);
    assert!(matches!(
        provider.status(0, 1).unwrap(),
        BloomFilterStatus::Present(_)
    ));

    let metadata = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::columns([0])))
        .with_arrow_reader_options(Some(&ArrowReaderOptions::default()))
        .parse_and_finish(&data)
        .unwrap();
    let provider = metadata.bloom_filters().unwrap();
    assert!(matches!(
        provider.status(0, 0).unwrap(),
        BloomFilterStatus::Present(_)
    ));
    assert_eq!(provider.status(0, 1).unwrap(), BloomFilterStatus::NotLoaded);

    let override_options = ArrowReaderOptions::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::columns([1])));
    let metadata = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::columns([0])))
        .with_arrow_reader_options(Some(&override_options))
        .parse_and_finish(&data)
        .unwrap();
    let provider = metadata.bloom_filters().unwrap();
    assert_eq!(provider.status(0, 0).unwrap(), BloomFilterStatus::NotLoaded);
    assert!(matches!(
        provider.status(0, 1).unwrap(),
        BloomFilterStatus::Present(_)
    ));

    let partial = ParquetMetaDataReader::new()
        .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::columns([0])))
        .parse_and_finish(&data)
        .unwrap();
    let original = Arc::clone(partial.bloom_filters().unwrap());
    let metadata = ArrowReaderMetadata::try_new(
        Arc::new(partial),
        ArrowReaderOptions::new()
            .with_bloom_filter_policy(BloomFilterPolicy::Optional(ColumnChunkMask::all())),
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        metadata.metadata.bloom_filters().unwrap(),
        &original
    ));
}

#[test]
fn test_page_encoding_stats_mask() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/alltypes_tiny_pages.parquet");
    let file = File::open(path).unwrap();

    let arrow_options = ArrowReaderOptions::new().with_encoding_stats_as_mask(true);
    let builder =
        ParquetRecordBatchReaderBuilder::try_new_with_options(file, arrow_options).unwrap();

    let row_group_metadata = builder.metadata.row_group(0);

    // test page encoding stats
    let page_encoding_stats = row_group_metadata
        .column(0)
        .page_encoding_stats_mask()
        .unwrap();
    assert!(page_encoding_stats.is_only(Encoding::PLAIN));
    let page_encoding_stats = row_group_metadata
        .column(2)
        .page_encoding_stats_mask()
        .unwrap();
    assert!(page_encoding_stats.is_only(Encoding::PLAIN_DICTIONARY));
}

#[test]
fn test_stats_stats_skipped() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/alltypes_tiny_pages.parquet");
    let file = File::open(path).unwrap();

    // test skipping all
    let arrow_options = ArrowReaderOptions::new()
        .with_encoding_stats_policy(ParquetStatisticsPolicy::SkipAll)
        .with_column_stats_policy(ParquetStatisticsPolicy::SkipAll);
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        file.try_clone().unwrap(),
        arrow_options,
    )
    .unwrap();

    let row_group_metadata = builder.metadata.row_group(0);
    for column in row_group_metadata.columns() {
        assert!(column.page_encoding_stats().is_none());
        assert!(column.page_encoding_stats_mask().is_none());
        assert!(column.statistics().is_none());
    }

    // test skipping all but one column and converting to mask
    let arrow_options = ArrowReaderOptions::new()
        .with_encoding_stats_as_mask(true)
        .with_encoding_stats_policy(ParquetStatisticsPolicy::skip_except(&[0]))
        .with_column_stats_policy(ParquetStatisticsPolicy::skip_except(&[0]));
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        file.try_clone().unwrap(),
        arrow_options,
    )
    .unwrap();

    let row_group_metadata = builder.metadata.row_group(0);
    for (idx, column) in row_group_metadata.columns().iter().enumerate() {
        assert!(column.page_encoding_stats().is_none());
        assert_eq!(column.page_encoding_stats_mask().is_some(), idx == 0);
        assert_eq!(column.statistics().is_some(), idx == 0);
    }
}

#[test]
fn test_size_stats_stats_skipped() {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/repeated_primitive_no_list.parquet");
    let file = File::open(path).unwrap();

    // test skipping all
    let arrow_options =
        ArrowReaderOptions::new().with_size_stats_policy(ParquetStatisticsPolicy::SkipAll);
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        file.try_clone().unwrap(),
        arrow_options,
    )
    .unwrap();

    let row_group_metadata = builder.metadata.row_group(0);
    for column in row_group_metadata.columns() {
        assert!(column.repetition_level_histogram().is_none());
        assert!(column.definition_level_histogram().is_none());
        assert!(column.unencoded_byte_array_data_bytes().is_none());
    }

    // test skipping all but one column and converting to mask
    let arrow_options = ArrowReaderOptions::new()
        .with_encoding_stats_as_mask(true)
        .with_size_stats_policy(ParquetStatisticsPolicy::skip_except(&[1]));
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        file.try_clone().unwrap(),
        arrow_options,
    )
    .unwrap();

    let row_group_metadata = builder.metadata.row_group(0);
    for (idx, column) in row_group_metadata.columns().iter().enumerate() {
        assert_eq!(column.repetition_level_histogram().is_some(), idx == 1);
        assert_eq!(column.definition_level_histogram().is_some(), idx == 1);
        assert_eq!(column.unencoded_byte_array_data_bytes().is_some(), idx == 1);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Takes too long
fn test_read_with_page_index_enabled() {
    let testdata = arrow::util::test_util::parquet_test_data();

    {
        // `alltypes_tiny_pages.parquet` has page index
        let path = format!("{testdata}/alltypes_tiny_pages.parquet");
        let test_file = File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            test_file,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required),
        )
        .unwrap();
        let row_group_page_index = builder.metadata().page_index_for_row_group(0);
        assert!(row_group_page_index.offset_index(0).is_some());
        assert!(row_group_page_index.column_index(0).is_some());
        assert!(row_group_page_index.page_locations(0).is_some());
        assert_eq!(row_group_page_index.num_data_pages(0), Some(325));
        let reader = builder.build().unwrap();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 8);
    }

    {
        // `alltypes_plain.parquet` doesn't have page index
        let path = format!("{testdata}/alltypes_plain.parquet");
        let test_file = File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            test_file,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required),
        )
        .unwrap();
        // Although `Vec<Vec<PageLocation>>` of each row group is empty,
        // we should read the file successfully.
        assert!(builder.metadata().page_index().is_none());
        let reader = builder.build().unwrap();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
    }
}
