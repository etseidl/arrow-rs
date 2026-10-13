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

#[cfg(feature = "async")]
mod r#async;
mod sync;

use bytes::Bytes;
use std::io::Cursor;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
#[cfg(feature = "async")]
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::{ArrowWriter, arrow_reader::ArrowReaderOptions};
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::{BloomFilterPolicy, ColumnChunkMask};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{ChunkReader, Length};
use parquet::schema::types::ColumnPath;

fn cached_bloom_filter_file() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
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
        .set_column_bloom_filter_enabled(ColumnPath::from("a"), true)
        .build();
    let mut data = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut data, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    data.into()
}

fn external_cache_file() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
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
        .set_max_row_group_row_count(Some(batch.num_rows()))
        .set_column_bloom_filter_enabled(ColumnPath::from("a"), true)
        .build();
    let mut data = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut data, schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    data.into()
}

fn cache_options(mask: ColumnChunkMask) -> ArrowReaderOptions {
    ArrowReaderOptions::new().with_bloom_filter_policy(BloomFilterPolicy::Optional(mask))
}

#[derive(Clone)]
struct CountingChunkReader {
    data: Bytes,
    requests: Arc<AtomicUsize>,
}

impl CountingChunkReader {
    fn new(data: Bytes) -> Self {
        Self {
            data,
            requests: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl Length for CountingChunkReader {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
}

impl ChunkReader for CountingChunkReader {
    type T = Cursor<Bytes>;

    fn get_read(&self, start: u64) -> Result<Self::T> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let start = usize::try_from(start)?;
        let data = self
            .data
            .get(start..)
            .ok_or_else(|| ParquetError::General(format!("invalid test offset {start}")))?;
        Ok(Cursor::new(Bytes::copy_from_slice(data)))
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let start = usize::try_from(start)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| ParquetError::General("test range overflow".into()))?;
        self.data
            .get(start..end)
            .map(Bytes::copy_from_slice)
            .ok_or_else(|| ParquetError::General(format!("invalid test range {start}..{end}")))
    }
}

#[cfg(feature = "async")]
fn malformed_bloom_filter_header() -> (Bytes, ArrowReaderMetadata) {
    let testdata = arrow::util::test_util::parquet_test_data();
    let path = format!("{testdata}/data_index_bloom_encoding_stats.parquet");
    let data = Bytes::from(std::fs::read(path).unwrap());
    let metadata = ArrowReaderMetadata::load(&data, ArrowReaderOptions::new()).unwrap();
    let offset = usize::try_from(
        metadata
            .metadata()
            .row_group(0)
            .column(0)
            .bloom_filter_offset()
            .unwrap(),
    )
    .unwrap();

    let mut corrupted = data.to_vec();
    assert_eq!(
        corrupted[offset], 0x15,
        "unexpected compact-protocol field header"
    );

    let value_start = offset + 1;
    let value_end = corrupted[value_start..]
        .iter()
        .position(|byte| byte & 0x80 == 0)
        .map(|relative| value_start + relative)
        .unwrap();
    if value_end == value_start {
        corrupted[value_start] = 1;
    } else {
        corrupted[value_start..value_end].fill(0xff);
        corrupted[value_end] = 0;
    }

    (corrupted.into(), metadata)
}
