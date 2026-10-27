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

//! Async byte sources for reading Parquet metadata without Arrow.

use crate::errors::Result;
use bytes::Bytes;
use futures::future::BoxFuture;
use std::ops::Range;

/// A data source that can be used with [`ParquetMetaDataReader`] to load [`ParquetMetaData`]
///
/// With the `arrow` feature enabled, an implementation is provided for
/// mutable references to `AsyncFileReader`.
///
/// # Example `MetadataFetch` for a custom async data source
///
/// ```rust
/// # use parquet::errors::Result;
/// # use parquet::file::metadata::MetadataFetch;
/// # use bytes::Bytes;
/// # use std::ops::Range;
/// # use std::io::SeekFrom;
/// # use futures::future::BoxFuture;
/// # use futures::FutureExt;
/// # use tokio::io::{AsyncReadExt, AsyncSeekExt};
/// // Adapter that implements the API for reading bytes from an async source (in
/// // this case a tokio::fs::File)
/// struct TokioFileMetadata {
///     file: tokio::fs::File,
/// }
/// impl MetadataFetch for TokioFileMetadata {
///     fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
///         // return a future that fetches data in range
///         async move {
///             let len = (range.end - range.start).try_into().unwrap();
///             let mut buf = vec![0; len]; // target buffer
///             // seek to the start of the range and read the data
///             self.file.seek(SeekFrom::Start(range.start)).await?;
///             self.file.read_exact(&mut buf).await?;
///             Ok(Bytes::from(buf)) // convert to Bytes
///         }
///             .boxed() // turn into BoxedFuture, using FutureExt::boxed
///     }
/// }
///```
///
/// [`ParquetMetaDataReader`]: crate::file::metadata::ParquetMetaDataReader
/// [`ParquetMetaData`]: crate::file::metadata::ParquetMetaData
pub trait MetadataFetch {
    /// Return a future that fetches exactly the half-open byte range `[start, end)`
    /// asynchronously, returning `end - start` bytes
    ///
    /// Note the returned type is a boxed future, often created by
    /// [`FutureExt::boxed`]. See the trait documentation for an example
    ///
    /// [`FutureExt::boxed`]: futures::FutureExt::boxed
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>>;
}

impl MetadataFetch for Box<dyn MetadataFetch + '_> {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        self.as_mut().fetch(range)
    }
}

impl MetadataFetch for &mut dyn MetadataFetch {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
        (**self).fetch(range)
    }
}

/// A data source that can be used with [`ParquetMetaDataReader`] to load [`ParquetMetaData`] via suffix
/// requests, without knowing the file size
///
/// [`ParquetMetaDataReader`]: crate::file::metadata::ParquetMetaDataReader
/// [`ParquetMetaData`]: crate::file::metadata::ParquetMetaData
pub trait MetadataSuffixFetch: MetadataFetch {
    /// Return a future that fetches the last `n` bytes asynchronously
    ///
    /// Note the returned type is a boxed future, often created by
    /// [`FutureExt::boxed`]. See the trait documentation for an example
    ///
    /// [`FutureExt::boxed`]: futures::FutureExt::boxed
    fn fetch_suffix(&mut self, suffix: usize) -> BoxFuture<'_, Result<Bytes>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::metadata::{
        FileMetaData, ParquetMetaData, ParquetMetaDataReader, ParquetMetaDataWriter,
    };
    use crate::schema::{parser::parse_message_type, types::SchemaDescriptor};
    use futures::FutureExt;
    use std::sync::Arc;

    struct MemorySource(Bytes);

    impl MetadataFetch for MemorySource {
        fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
            futures::future::ready(Ok(self.0.slice(range.start as usize..range.end as usize)))
                .boxed()
        }
    }

    impl MetadataSuffixFetch for MemorySource {
        fn fetch_suffix(&mut self, suffix: usize) -> BoxFuture<'_, Result<Bytes>> {
            let start = self.0.len().saturating_sub(suffix);
            futures::future::ready(Ok(self.0.slice(start..))).boxed()
        }
    }

    #[tokio::test]
    async fn test_metadata_fetch_footer() {
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(
            parse_message_type("message schema { REQUIRED INT32 value; }").unwrap(),
        )));
        let metadata =
            ParquetMetaData::new(FileMetaData::new(1, 0, None, None, schema, None), vec![]);
        let mut buffer = Vec::new();
        ParquetMetaDataWriter::new(&mut buffer, &metadata)
            .finish()
            .unwrap();
        let bytes = Bytes::from(buffer);
        let loaded = ParquetMetaDataReader::new()
            .load_and_finish(MemorySource(bytes.clone()), bytes.len() as u64)
            .await
            .unwrap();
        let suffix_loaded = ParquetMetaDataReader::new()
            .load_via_suffix_and_finish(MemorySource(bytes.clone()))
            .await
            .unwrap();
        for loaded in [loaded, suffix_loaded] {
            assert_eq!(loaded.num_row_groups(), 0);
            assert_eq!(loaded.file_metadata().num_rows(), 0);
            assert_eq!(
                loaded.file_metadata().schema_descr(),
                metadata.file_metadata().schema_descr()
            );
        }

        // Both traits remain object safe, including at the canonical reexport.
        let mut source = MemorySource(bytes.clone());
        let fetch: &mut dyn crate::file::metadata::MetadataFetch = &mut source;
        assert_eq!(fetch.fetch(0..4).await.unwrap(), bytes.slice(0..4));
        let suffix: &mut dyn crate::file::metadata::MetadataSuffixFetch = &mut source;
        assert_eq!(
            suffix.fetch_suffix(4).await.unwrap(),
            bytes.slice(bytes.len() - 4..)
        );
    }

    #[cfg(feature = "arrow")]
    #[tokio::test]
    async fn test_arrow_metadata_fetch_reexports() {
        let mut source = MemorySource(Bytes::from_static(b"PAR1"));
        let old: &mut dyn crate::arrow::async_reader::MetadataFetch = &mut source;
        let canonical: &mut dyn MetadataFetch = old;
        assert_eq!(canonical.fetch(0..4).await.unwrap(), &b"PAR1"[..]);
        let old: &mut dyn crate::arrow::async_reader::MetadataSuffixFetch = &mut source;
        let canonical: &mut dyn MetadataSuffixFetch = old;
        assert_eq!(canonical.fetch_suffix(4).await.unwrap(), &b"PAR1"[..]);

        // The borrowed AsyncFileReader adapter still implements the canonical trait.
        let mut reader = std::io::Cursor::new(b"PAR1".to_vec());
        let mut borrowed = &mut reader;
        let canonical: &mut dyn MetadataFetch = &mut borrowed;
        assert_eq!(canonical.fetch(0..4).await.unwrap(), &b"PAR1"[..]);
    }
}
