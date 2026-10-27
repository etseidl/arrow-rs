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

use crate::errors::{ParquetError, Result};
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

    /// Fetch multiple half-open ranges, returning one buffer per request in input
    /// order. Each buffer must contain exactly `range.end - range.start` bytes.
    /// The default calls [`Self::fetch`] sequentially and propagates fetch errors.
    /// Implementations may override this to use batch I/O.
    ///
    /// Only this method requires `Send`, since its future retains the source
    /// between requests. Existing non-`Send` sources can still use [`Self::fetch`].
    /// Consumers of overridden implementations must validate response count and
    /// lengths; these checks cannot detect same-length reordered or altered bytes.
    ///
    /// A fetch-only implementation remains sufficient, including through a trait
    /// object. Inherent methods with the same name retain normal Rust precedence:
    ///
    /// ```rust
    /// use bytes::Bytes;
    /// use futures::future::{ready, BoxFuture};
    /// use parquet::errors::Result;
    /// use parquet::file::metadata::MetadataFetch;
    /// use std::ops::Range;
    ///
    /// struct Source;
    /// impl MetadataFetch for Source {
    ///     fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
    ///         Box::pin(ready(Ok(Bytes::from(vec![0; (range.end - range.start) as usize]))))
    ///     }
    /// }
    /// impl Source {
    ///     fn fetch_ranges(&self) -> usize { 42 }
    /// }
    /// # async fn example() -> Result<()> {
    /// let mut source = Source;
    /// assert_eq!(source.fetch_ranges(), 42);
    /// let fetch: &mut (dyn MetadataFetch + Send) = &mut source;
    /// assert_eq!(fetch.fetch_ranges(vec![2..4, 0..1]).await?.len(), 2);
    /// assert_eq!(MetadataFetch::fetch_ranges(&mut source, vec![]).await?.len(), 0);
    /// # Ok(())
    /// # }
    /// ```
    fn fetch_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        Box::pin(async move {
            let mut result = Vec::with_capacity(ranges.len());
            for range in &ranges {
                let bytes = self.fetch(range.clone()).await?;
                validate_metadata_ranges(
                    std::slice::from_ref(range),
                    std::slice::from_ref(&bytes),
                )?;
                result.push(bytes);
            }
            Ok(result)
        })
    }
}

/// Checks the structural batch contract, not byte identity or ordering authenticity.
pub(crate) fn validate_metadata_ranges(ranges: &[Range<u64>], responses: &[Bytes]) -> Result<()> {
    if ranges.len() != responses.len() {
        return Err(ParquetError::General(format!(
            "Metadata range response count mismatch: expected {}, got {}",
            ranges.len(),
            responses.len()
        )));
    }
    for (range, response) in ranges.iter().zip(responses) {
        let length = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| ParquetError::General(format!("Invalid metadata range: {range:?}")))?;
        if u64::try_from(response.len()).ok() != Some(length) {
            return Err(ParquetError::General(format!(
                "Metadata range {range:?} response length mismatch: expected {length}, got {}",
                response.len()
            )));
        }
    }
    Ok(())
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
    async fn test_metadata_fetch_ranges_default_and_dyn() {
        let mut source = MemorySource(Bytes::from_static(b"abcdefgh"));
        let fetch: &mut (dyn MetadataFetch + Send) = &mut source;
        assert_eq!(
            fetch
                .fetch_ranges(vec![4..6, 0..3, 4..6, 8..8])
                .await
                .unwrap(),
            vec![
                Bytes::from_static(b"ef"),
                Bytes::from_static(b"abc"),
                Bytes::from_static(b"ef"),
                Bytes::new()
            ]
        );
        assert!(fetch.fetch_ranges(vec![]).await.unwrap().is_empty());

        // A preexisting non-Send implementation still compiles and supports fetch.
        struct LocalSource(std::rc::Rc<Bytes>);
        impl MetadataFetch for LocalSource {
            fn fetch(&mut self, _: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
                futures::future::ready(Ok((*self.0).clone())).boxed()
            }
        }
        let mut local = LocalSource(std::rc::Rc::new(Bytes::from_static(b"a")));
        let fetch: &mut dyn MetadataFetch = &mut local;
        assert_eq!(fetch.fetch(0..1).await.unwrap(), &b"a"[..]);
    }

    #[tokio::test]
    async fn test_metadata_fetch_ranges_custom() {
        struct BatchSource(usize);
        impl MetadataFetch for BatchSource {
            fn fetch(&mut self, _: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
                panic!("batch override must not use single fetch");
            }
            fn fetch_ranges(
                &mut self,
                ranges: Vec<Range<u64>>,
            ) -> BoxFuture<'_, Result<Vec<Bytes>>> {
                self.0 += 1;
                let result = ranges
                    .iter()
                    .map(|r| Bytes::from(vec![r.start as u8; (r.end - r.start) as usize]))
                    .collect();
                futures::future::ready(Ok(result)).boxed()
            }
        }
        let mut source = BatchSource(0);
        let fetch: &mut (dyn MetadataFetch + Send) = &mut source;
        let ranges = vec![2..4, 1..2];
        let result = fetch.fetch_ranges(ranges.clone()).await.unwrap();
        validate_metadata_ranges(&ranges, &result).unwrap();
        assert_eq!(
            result,
            vec![Bytes::from_static(&[2, 2]), Bytes::from_static(&[1])]
        );
        assert_eq!(source.0, 1);
    }

    #[tokio::test]
    async fn test_metadata_fetch_ranges_default_invalid_and_error() {
        struct ResponseSource {
            calls: Vec<Range<u64>>,
            response: Result<Bytes>,
        }
        impl MetadataFetch for ResponseSource {
            fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes>> {
                self.calls.push(range);
                futures::future::ready(std::mem::replace(&mut self.response, Ok(Bytes::new())))
                    .boxed()
            }
        }
        for response in [
            Ok(Bytes::new()),
            Ok(Bytes::from_static(b"extra")),
            Err(ParquetError::General("fetch failed".into())),
        ] {
            let mut source = ResponseSource {
                calls: vec![],
                response,
            };
            let is_error = source.response.is_err();
            let error = source.fetch_ranges(vec![2..4, 0..1]).await.unwrap_err();
            assert_eq!(source.calls, vec![2..4]);
            if is_error {
                assert_eq!(error.to_string(), "Parquet error: fetch failed");
            } else {
                assert!(error.to_string().contains("response length mismatch"));
            }
        }
    }

    #[test]
    fn test_validate_metadata_ranges() {
        let ranges = [0..2, 5..6];
        let valid = [Bytes::from_static(b"ab"), Bytes::from_static(b"f")];
        validate_metadata_ranges(&ranges, &valid).unwrap();
        validate_metadata_ranges(&[], &[]).unwrap();
        for responses in [
            vec![],
            vec![valid[0].clone()],
            vec![valid[0].clone(), valid[1].clone(), Bytes::new()],
        ] {
            assert!(
                validate_metadata_ranges(&ranges, &responses)
                    .unwrap_err()
                    .to_string()
                    .contains("response count mismatch")
            );
        }
        for bytes in [Bytes::new(), Bytes::from_static(b"abc")] {
            assert!(
                validate_metadata_ranges(&ranges, &[bytes, valid[1].clone()])
                    .unwrap_err()
                    .to_string()
                    .contains("response length mismatch")
            );
        }
        let inverted = Range { start: 2, end: 1 };
        assert!(validate_metadata_ranges(&[inverted], &[Bytes::new()]).is_err());
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
