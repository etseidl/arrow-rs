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

#[cfg(feature = "async")]
use bytes::Bytes;
#[cfg(feature = "async")]
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};

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
