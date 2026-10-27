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

pub use crate::file::metadata::{MetadataFetch, MetadataSuffixFetch};

use crate::arrow::async_reader::AsyncFileReader;
use crate::errors::Result;
use crate::file::metadata::async_reader::validate_metadata_ranges;
use bytes::Bytes;
use futures::future::BoxFuture;
use std::ops::Range;

impl<T: AsyncFileReader> MetadataFetch for &mut T {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        self.get_bytes(range)
    }

    fn fetch_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        Box::pin(async move {
            let result = self.get_byte_ranges(ranges.clone()).await?;
            validate_metadata_ranges(&ranges, &result)?;
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::arrow_reader::ArrowReaderOptions;
    use crate::errors::ParquetError;
    use crate::file::metadata::ParquetMetaData;
    use futures::FutureExt;
    use std::sync::Arc;

    struct BatchReader {
        calls: Vec<Vec<Range<u64>>>,
        response: Result<Vec<Bytes>>,
    }

    impl AsyncFileReader for BatchReader {
        fn get_bytes(&mut self, _: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
            panic!("metadata batch must delegate to get_byte_ranges");
        }

        fn get_byte_ranges(
            &mut self,
            ranges: Vec<Range<u64>>,
        ) -> BoxFuture<'_, Result<Vec<Bytes>>> {
            self.calls.push(ranges);
            futures::future::ready(std::mem::replace(&mut self.response, Ok(vec![]))).boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, Result<Arc<ParquetMetaData>>> {
            panic!("batch must not load footer metadata");
        }
    }

    #[tokio::test]
    async fn test_metadata_fetch_ranges_bridge() {
        let ranges = vec![4..6, 0..1];
        let expected = vec![Bytes::from_static(b"ef"), Bytes::from_static(b"a")];
        let mut reader = BatchReader {
            calls: vec![],
            response: Ok(expected.clone()),
        };
        let mut borrowed = &mut reader;
        let fetch: &mut (dyn MetadataFetch + Send) = &mut borrowed;
        assert_eq!(fetch.fetch_ranges(ranges.clone()).await.unwrap(), expected);
        assert_eq!(reader.calls, vec![ranges]);
    }

    #[tokio::test]
    async fn test_metadata_fetch_ranges_bridge_invalid_and_error() {
        for (response, message) in [
            (Ok(vec![]), "response count mismatch"),
            (Ok(vec![Bytes::new()]), "response length mismatch"),
            (
                Ok(vec![Bytes::from_static(b"abc")]),
                "response length mismatch",
            ),
            (
                Err(ParquetError::General("batch failed".into())),
                "batch failed",
            ),
        ] {
            let mut reader = BatchReader {
                calls: vec![],
                response,
            };
            let mut borrowed = &mut reader;
            let error = MetadataFetch::fetch_ranges(&mut borrowed, vec![0..2])
                .await
                .unwrap_err();
            assert!(error.to_string().contains(message));
            assert_eq!(reader.calls, vec![vec![0..2]]);
        }
    }
}
