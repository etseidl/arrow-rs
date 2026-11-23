// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Contains codec interface and supported codec implementations.
//!
//! See [`Compression`](crate::basic::Compression) enum for all available compression
//! algorithms.
//!
// NOTE: this must be an inner attribute so that the example is attached to the module (and
// therefore actually run as a doc test) rather than to the `use` statement below.
#![cfg_attr(
    feature = "experimental",
    doc = r"
# Example

```no_run
use parquet::{basic::Compression, compression::{create_codec, CodecOptionsBuilder}};

let codec_options = CodecOptionsBuilder::default()
    .set_backward_compatible_lz4(false)
    .build();
let mut codec = match create_codec(Compression::SNAPPY, &codec_options) {
 Ok(Some(codec)) => codec,
 _ => panic!(),
};

let data = vec![b'p', b'a', b'r', b'q', b'u', b'e', b't'];
let mut compressed = vec![];
codec.compress(&data[..], &mut compressed).unwrap();

let mut output = vec![];
codec.decompress(&compressed[..], &mut output, None).unwrap();

assert_eq!(output, data);
```
"
)]
use crate::basic::Compression as CodecType;
use crate::errors::{ParquetError, Result};

/// Parquet compression codec interface.
pub trait Codec: Send {
    /// Compresses data stored in slice `input_buf` and appends the compressed result
    /// to `output_buf`.
    ///
    /// Note that you'll need to call `clear()` before reusing the same `output_buf`
    /// across different `compress` calls.
    fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()>;

    /// Decompresses data stored in slice `input_buf` and appends output to `output_buf`.
    ///
    /// The built-in codecs use `uncompress_size`, when provided, as a limit on
    /// newly appended bytes. Raw LZ4 and the Hadoop framing path require an exact
    /// size; backward-compatible LZ4 frame fallback uses an upper bound. Raw and
    /// Hadoop LZ4 do not support `None`. This is not a limit on total decoder memory.
    ///
    /// Returns the number of bytes appended. Built-in codecs restore the original
    /// output length on error.
    fn decompress(
        &mut self,
        input_buf: &[u8],
        output_buf: &mut Vec<u8>,
        uncompress_size: Option<usize>,
    ) -> Result<usize>;
}

#[cfg(any(feature = "snap", feature = "lz4", test))]
fn resize_output(output: &mut Vec<u8>, len: usize) -> Result<()> {
    if len > output.len() {
        output
            .try_reserve(len - output.len())
            .map_err(|e| ParquetError::External(Box::new(e)))?;
    }
    output.resize(len, 0);
    Ok(())
}

#[cfg(any(
    feature = "flate2",
    feature = "brotli",
    feature = "lz4",
    feature = "zstd",
    test
))]
fn append_decoded(
    mut decoder: impl std::io::Read,
    output: &mut Vec<u8>,
    limit: Option<usize>,
) -> Result<usize> {
    let offset = output.len();
    let result = (|| {
        let mut scratch = [0; 8192];
        loop {
            let appended = output.len() - offset;
            // Probe EOF at the exact limit without appending a byte or adding one
            // to a potentially maximum-sized limit.
            let remaining = limit.map_or(scratch.len(), |max| max - appended);
            let read_len = remaining.min(scratch.len()).max(1);
            let n = match decoder.read(&mut scratch[..read_len]) {
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if n == 0 {
                return Ok(appended);
            }
            if n > remaining {
                return Err(general_err!("Decompressed data exceeds uncompress_size"));
            }
            output
                .try_reserve(n)
                .map_err(|e| ParquetError::External(Box::new(e)))?;
            output.extend_from_slice(&scratch[..n]);
        }
    })();
    if result.is_err() {
        output.truncate(offset);
    }
    result
}

/// Struct to hold `Codec` creation options.
#[derive(Debug, PartialEq, Eq)]
pub struct CodecOptions {
    /// Whether or not to fallback to other LZ4 older implementations on error in LZ4_HADOOP.
    backward_compatible_lz4: bool,
}

impl Default for CodecOptions {
    fn default() -> Self {
        CodecOptionsBuilder::default().build()
    }
}

pub struct CodecOptionsBuilder {
    /// Whether or not to fallback to other LZ4 older implementations on error in LZ4_HADOOP.
    backward_compatible_lz4: bool,
}

impl Default for CodecOptionsBuilder {
    fn default() -> Self {
        Self {
            backward_compatible_lz4: true,
        }
    }
}

impl CodecOptionsBuilder {
    /// Enable/disable backward compatible LZ4.
    ///
    /// If backward compatible LZ4 is enable, on LZ4_HADOOP error it will fallback
    /// to the older versions LZ4 algorithms. That is LZ4_FRAME, for backward compatibility
    /// with files generated by older versions of this library, and LZ4_RAW, for backward
    /// compatibility with files generated by older versions of parquet-cpp.
    ///
    /// If backward compatible LZ4 is disabled, on LZ4_HADOOP error it will return the error.
    pub fn set_backward_compatible_lz4(mut self, value: bool) -> CodecOptionsBuilder {
        self.backward_compatible_lz4 = value;
        self
    }

    pub fn build(self) -> CodecOptions {
        CodecOptions {
            backward_compatible_lz4: self.backward_compatible_lz4,
        }
    }
}

/// Defines valid compression levels.
pub(crate) trait CompressionLevel<T: std::fmt::Display + std::cmp::PartialOrd> {
    const MINIMUM_LEVEL: T;
    const MAXIMUM_LEVEL: T;

    /// Tests if the provided compression level is valid.
    fn is_valid_level(level: T) -> Result<()> {
        let compression_range = Self::MINIMUM_LEVEL..=Self::MAXIMUM_LEVEL;
        if compression_range.contains(&level) {
            Ok(())
        } else {
            Err(ParquetError::General(format!(
                "valid compression range {}..={} exceeded.",
                compression_range.start(),
                compression_range.end()
            )))
        }
    }
}

/// Given the compression type `codec`, returns a codec used to compress and decompress
/// bytes for the compression type.
/// This returns `None` if the codec type is `UNCOMPRESSED`.
#[cfg_attr(
    any(feature = "lz4", test),
    expect(
        clippy::used_underscore_binding,
        reason = "`_options` is only read when the `lz4` feature is on"
    )
)]
pub fn create_codec(codec: CodecType, _options: &CodecOptions) -> Result<Option<Box<dyn Codec>>> {
    #[cfg_attr(
        any(
            test,
            feature = "brotli",
            feature = "flate2",
            feature = "lz4",
            feature = "snap",
            feature = "zstd"
        ),
        expect(unreachable_code)
    )]
    #[cfg_attr(
        all(
            not(test),
            not(all(feature = "brotli", feature = "flate2", feature = "zstd"))
        ),
        expect(unused_variables)
    )]
    match codec {
        CodecType::BROTLI(level) => {
            #[cfg(any(feature = "brotli", test))]
            return Ok(Some(Box::new(BrotliCodec::new(level))));
            Err(ParquetError::General(
                "Disabled feature at compile time: brotli".into(),
            ))
        }
        CodecType::GZIP(level) => {
            #[cfg(any(feature = "flate2", test))]
            return Ok(Some(Box::new(GZipCodec::new(level))));
            Err(ParquetError::General(
                "Disabled feature at compile time: flate2".into(),
            ))
        }
        CodecType::SNAPPY => {
            #[cfg(any(feature = "snap", test))]
            return Ok(Some(Box::new(SnappyCodec::new())));
            Err(ParquetError::General(
                "Disabled feature at compile time: snap".into(),
            ))
        }
        CodecType::LZ4 => {
            #[cfg(any(feature = "lz4", test))]
            return Ok(Some(Box::new(LZ4HadoopCodec::new(
                _options.backward_compatible_lz4,
            ))));
            Err(ParquetError::General(
                "Disabled feature at compile time: lz4".into(),
            ))
        }
        CodecType::ZSTD(level) => {
            #[cfg(any(feature = "zstd", test))]
            return Ok(Some(Box::new(ZSTDCodec::new(level))));
            Err(ParquetError::General(
                "Disabled feature at compile time: zstd".into(),
            ))
        }
        CodecType::LZ4_RAW => {
            #[cfg(any(feature = "lz4", test))]
            return Ok(Some(Box::new(LZ4RawCodec::new())));
            Err(ParquetError::General(
                "Disabled feature at compile time: lz4".into(),
            ))
        }
        CodecType::UNCOMPRESSED => Ok(None),
        CodecType::LZO => Err(nyi_err!("The codec type {} is not supported yet", codec)),
    }
}

#[cfg(any(feature = "snap", test))]
mod snappy_codec {
    use snap::raw::{Decoder, Encoder, decompress_len, max_compress_len};

    use crate::compression::{Codec, resize_output};
    use crate::errors::{ParquetError, Result};

    /// Codec for Snappy compression format.
    pub struct SnappyCodec {
        decoder: Decoder,
        encoder: Encoder,
    }

    impl SnappyCodec {
        /// Creates new Snappy compression codec.
        pub(crate) fn new() -> Self {
            Self {
                decoder: Decoder::new(),
                encoder: Encoder::new(),
            }
        }
    }

    impl Codec for SnappyCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            let len = decompress_len(input_buf)?;
            if uncompress_size.is_some_and(|max| len > max) {
                return Err(general_err!("Decompressed data exceeds uncompress_size"));
            }
            // Copy2 has the greatest expansion: 64 output bytes per three input
            // bytes. Including the header makes this a conservative validity check,
            // not validation of the actual tokens or a limit on decoder memory.
            let min_compressed_len = u64::try_from(len)?
                .checked_mul(3)
                .ok_or_else(|| general_err!("Snappy output size overflow"))?
                .div_ceil(64);
            let min_compressed_len = usize::try_from(min_compressed_len)?;
            if input_buf.len() < min_compressed_len {
                return Err(general_err!(
                    "Snappy header declares {} bytes, which need at least {} compressed bytes, but {} were provided",
                    len,
                    min_compressed_len,
                    input_buf.len()
                ));
            }
            let offset = output_buf.len();
            let end = offset
                .checked_add(len)
                .ok_or_else(|| general_err!("Snappy output size overflow"))?;
            resize_output(output_buf, end)?;
            match self
                .decoder
                .decompress(input_buf, &mut output_buf[offset..end])
            {
                Ok(n) => {
                    output_buf.truncate(offset + n);
                    Ok(n)
                }
                Err(e) => {
                    output_buf.truncate(offset);
                    Err(e.into())
                }
            }
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let output_buf_len = output_buf.len();
            let required_len = max_compress_len(input_buf.len());
            if required_len == 0 {
                return Err(general_err!("Snappy input exceeds supported size"));
            }
            let end = output_buf_len
                .checked_add(required_len)
                .ok_or_else(|| general_err!("Snappy output size overflow"))?;
            resize_output(output_buf, end)?;
            let n = self
                .encoder
                .compress(input_buf, &mut output_buf[output_buf_len..])?;
            output_buf.truncate(output_buf_len + n);
            Ok(())
        }
    }
}
#[cfg(any(feature = "snap", test))]
pub use snappy_codec::*;

#[cfg(any(feature = "flate2", test))]
mod gzip_codec {

    use std::io::Write;

    use flate2::{Compression, read, write};

    use crate::compression::{Codec, append_decoded};
    use crate::errors::Result;

    use super::GzipLevel;

    /// Codec for GZIP compression algorithm.
    pub struct GZipCodec {
        level: GzipLevel,
    }

    impl GZipCodec {
        /// Creates new GZIP compression codec.
        pub(crate) fn new(level: GzipLevel) -> Self {
            Self { level }
        }
    }

    impl Codec for GZipCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            append_decoded(
                read::MultiGzDecoder::new(input_buf),
                output_buf,
                uncompress_size,
            )
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let mut encoder = write::GzEncoder::new(output_buf, Compression::new(self.level.0));
            encoder.write_all(input_buf)?;
            encoder.try_finish().map_err(|e| e.into())
        }
    }
}
#[cfg(any(feature = "flate2", test))]
pub use gzip_codec::*;

/// Represents a valid gzip compression level.
///
/// Defaults to 6.
///
/// * 0: least compression
/// * 9: most compression (that other software can read)
/// * 10: most compression (incompatible with other software, see below)
/// #### WARNING:
/// Level 10 compression can offer smallest file size,
/// but Parquet files created with it will not be readable
/// by other "standard" paquet readers.
///
/// Do **NOT** use level 10 if you need other software to
/// be able to read the files. Read below for details.
///
/// ### IMPORTANT:
/// There's often confusion about the compression levels in `flate2` vs `arrow`
/// as highlighted in issue [#1011](https://github.com/apache/arrow-rs/issues/6282).
///
/// `flate2` supports two compression backends: `miniz_oxide` and `zlib`.
///
/// - `zlib` supports levels from 0 to 9.
/// - `miniz_oxide` supports levels from 0 to 10.
///
/// `arrow` uses `flate2` with `rust_backend` feature,
/// which provides `miniz_oxide` as the backend.
/// Therefore 0-10 levels are supported.
///
/// `flate2` documents this behavior properly with
/// [this commit](https://github.com/rust-lang/flate2-rs/pull/430).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct GzipLevel(u32);

impl Default for GzipLevel {
    fn default() -> Self {
        // The default as of miniz_oxide 0.5.1 is 6 for compression level
        // (miniz_oxide::deflate::CompressionLevel::DefaultLevel)
        Self(6)
    }
}

impl CompressionLevel<u32> for GzipLevel {
    const MINIMUM_LEVEL: u32 = 0;
    const MAXIMUM_LEVEL: u32 = 9;
}

impl GzipLevel {
    /// Attempts to create a gzip compression level.
    ///
    /// Compression levels must be valid (i.e. be acceptable for [`flate2::Compression`]).
    pub fn try_new(level: u32) -> Result<Self> {
        Self::is_valid_level(level).map(|()| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

#[cfg(any(feature = "brotli", test))]
mod brotli_codec {

    use std::io::Write;

    use crate::compression::{Codec, append_decoded};
    use crate::errors::Result;

    use super::BrotliLevel;

    const BROTLI_DEFAULT_BUFFER_SIZE: usize = 4096;
    const BROTLI_DEFAULT_LG_WINDOW_SIZE: u32 = 22; // recommended between 20-22

    /// Codec for Brotli compression algorithm.
    pub struct BrotliCodec {
        level: BrotliLevel,
    }

    impl BrotliCodec {
        /// Creates new Brotli compression codec.
        pub(crate) fn new(level: BrotliLevel) -> Self {
            Self { level }
        }
    }

    impl Codec for BrotliCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            let buffer_size = uncompress_size
                .unwrap_or(BROTLI_DEFAULT_BUFFER_SIZE)
                .clamp(1, BROTLI_DEFAULT_BUFFER_SIZE);
            append_decoded(
                brotli::Decompressor::new(input_buf, buffer_size),
                output_buf,
                uncompress_size,
            )
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let mut encoder = brotli::CompressorWriter::new(
                output_buf,
                BROTLI_DEFAULT_BUFFER_SIZE,
                self.level.0,
                BROTLI_DEFAULT_LG_WINDOW_SIZE,
            );
            encoder.write_all(input_buf)?;
            encoder.flush().map_err(|e| e.into())
        }
    }
}
#[cfg(any(feature = "brotli", test))]
pub use brotli_codec::*;

/// Represents a valid brotli compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct BrotliLevel(u32);

impl Default for BrotliLevel {
    fn default() -> Self {
        Self(1)
    }
}

impl CompressionLevel<u32> for BrotliLevel {
    const MINIMUM_LEVEL: u32 = 0;
    const MAXIMUM_LEVEL: u32 = 11;
}

impl BrotliLevel {
    /// Attempts to create a brotli compression level.
    ///
    /// Compression levels must be valid.
    pub fn try_new(level: u32) -> Result<Self> {
        Self::is_valid_level(level).map(|()| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

#[cfg(any(feature = "lz4", test))]
mod lz4_codec {
    use std::io::Write;

    use crate::compression::{Codec, append_decoded};
    use crate::errors::{ParquetError, Result};

    const LZ4_BUFFER_SIZE: usize = 4096;

    /// Codec for LZ4 compression algorithm.
    pub struct LZ4Codec {}

    impl LZ4Codec {
        /// Creates new LZ4 compression codec.
        pub(crate) fn new() -> Self {
            Self {}
        }
    }

    impl Codec for LZ4Codec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            append_decoded(
                lz4_flex::frame::FrameDecoder::new(input_buf),
                output_buf,
                uncompress_size,
            )
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(output_buf);
            for chunk in input_buf.chunks(LZ4_BUFFER_SIZE) {
                encoder.write_all(chunk)?;
            }
            match encoder.finish() {
                Ok(_) => Ok(()),
                Err(e) => Err(ParquetError::External(Box::new(e))),
            }
        }
    }
}

#[cfg(all(feature = "experimental", any(feature = "lz4", test)))]
pub use lz4_codec::*;

#[cfg(any(feature = "zstd", test))]
mod zstd_codec {
    use crate::compression::{Codec, ZstdLevel, append_decoded};
    use crate::errors::{ParquetError, Result};
    use std::io::Cursor;

    /// Codec for Zstandard compression algorithm.
    ///
    /// Reuses compression and decompression contexts across operations.
    pub struct ZSTDCodec {
        level: ZstdLevel,
        // Created on first use, because writers only compress and readers only
        // decompress.
        compressor: Option<zstd::bulk::Compressor<'static>>,
        decompressor: Option<zstd::zstd_safe::DCtx<'static>>,
    }

    impl ZSTDCodec {
        /// Creates new Zstandard compression codec.
        pub(crate) fn new(level: ZstdLevel) -> Self {
            Self {
                level,
                compressor: None,
                decompressor: None,
            }
        }
    }

    impl Codec for ZSTDCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            if self.decompressor.is_none() {
                self.decompressor = Some(zstd::zstd_safe::DCtx::try_create().ok_or_else(|| {
                    general_err!("Could not allocate Zstd decompression context")
                })?);
            }
            let decoder = zstd::stream::read::Decoder::with_context(
                input_buf,
                self.decompressor.as_mut().unwrap(),
            );
            append_decoded(decoder, output_buf, uncompress_size)
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let offset = output_buf.len();
            let len = zstd::zstd_safe::compress_bound(input_buf.len());
            output_buf
                .try_reserve(len)
                .map_err(|e| ParquetError::External(Box::new(e)))?;

            let mut cursor = Cursor::new(output_buf);
            cursor.set_position(offset as u64);
            let _written = self
                .compressor
                .get_or_insert_with(|| {
                    zstd::bulk::Compressor::new(self.level.compression_level())
                        .expect("valid zstd compression level")
                })
                .compress_to_buffer(input_buf, &mut cursor)?;
            Ok(())
        }
    }
}
#[cfg(any(feature = "zstd", test))]
pub use zstd_codec::*;

/// Represents a valid zstd compression level.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct ZstdLevel(i32);

impl CompressionLevel<i32> for ZstdLevel {
    // zstd binds to C, and hence zstd::compression_level_range() is not const as this calls the
    // underlying C library.
    const MINIMUM_LEVEL: i32 = -131072;
    const MAXIMUM_LEVEL: i32 = 22;
}

impl ZstdLevel {
    /// Attempts to create a zstd compression level from a given compression level.
    ///
    /// Compression levels must be valid (i.e. be acceptable for [`zstd::compression_level_range`]).
    pub fn try_new(level: i32) -> Result<Self> {
        Self::is_valid_level(level).map(|()| Self(level))
    }

    /// Returns the compression level.
    pub fn compression_level(&self) -> i32 {
        self.0
    }
}

impl Default for ZstdLevel {
    fn default() -> Self {
        Self(1)
    }
}

#[cfg(any(feature = "lz4", test))]
mod lz4_raw_codec {
    use crate::compression::{Codec, resize_output};
    use crate::errors::ParquetError;
    use crate::errors::Result;

    /// Codec for LZ4 Raw compression algorithm.
    pub struct LZ4RawCodec {}

    impl LZ4RawCodec {
        /// Creates new LZ4 Raw compression codec.
        pub(crate) fn new() -> Self {
            Self {}
        }
    }

    impl Codec for LZ4RawCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            let offset = output_buf.len();
            let Some(required_len) = uncompress_size else {
                return Err(ParquetError::General(
                    "LZ4RawCodec unsupported without uncompress_size".into(),
                ));
            };
            let result = (|| {
                let mut len = required_len.min(8192);
                loop {
                    let end = offset
                        .checked_add(len)
                        .ok_or_else(|| general_err!("LZ4 output size overflow"))?;
                    resize_output(output_buf, end)?;
                    match lz4_flex::block::decompress_into(input_buf, &mut output_buf[offset..end])
                    {
                        Ok(n) if n == required_len => return Ok(n),
                        Ok(_) => {
                            return Err(general_err!(
                                "LZ4RawCodec uncompress_size is not the expected one"
                            ));
                        }
                        Err(lz4_flex::block::DecompressError::OutputTooSmall { .. })
                            if len < required_len =>
                        {
                            // The reported required size is not validated output.
                            // Grow incrementally instead of allocating a forged declaration.
                            len = len.saturating_mul(2).min(required_len);
                        }
                        Err(e) => return Err(ParquetError::External(Box::new(e))),
                    }
                }
            })();
            if result.is_err() {
                output_buf.truncate(offset);
            }
            result
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            let offset = output_buf.len();
            // Guard the dependency's unchecked maximum-output-size arithmetic.
            let scaled = (input_buf.len() as u64)
                .checked_mul(110)
                .ok_or_else(|| general_err!("LZ4 input size overflow"))?
                / 100;
            usize::try_from(scaled)?
                .checked_add(20)
                .ok_or_else(|| general_err!("LZ4 output size overflow"))?;
            let required_len = lz4_flex::block::get_maximum_output_size(input_buf.len());
            let end = offset
                .checked_add(required_len)
                .ok_or_else(|| general_err!("LZ4 output size overflow"))?;
            resize_output(output_buf, end)?;
            match lz4_flex::block::compress_into(input_buf, &mut output_buf[offset..]) {
                Ok(n) => {
                    output_buf.truncate(offset + n);
                    Ok(())
                }
                Err(e) => Err(ParquetError::External(Box::new(e))),
            }
        }
    }
}
#[cfg(any(feature = "lz4", test))]
pub use lz4_raw_codec::*;

#[cfg(any(feature = "lz4", test))]
mod lz4_hadoop_codec {
    use crate::compression::lz4_codec::LZ4Codec;
    use crate::compression::lz4_raw_codec::LZ4RawCodec;
    use crate::compression::{Codec, resize_output};
    use crate::errors::{ParquetError, Result};
    use std::collections::TryReserveError;

    /// Size of u32 type.
    const SIZE_U32: usize = std::mem::size_of::<u32>();

    /// Length of the LZ4_HADOOP prefix.
    const PREFIX_LEN: usize = SIZE_U32 * 2;

    /// Codec for LZ4 Hadoop compression algorithm.
    pub struct LZ4HadoopCodec {
        /// Whether or not to fallback to other LZ4 implementations on error.
        /// Fallback is done to be backward compatible with older versions of this
        /// library and older versions parquet-cpp.
        backward_compatible_lz4: bool,
    }

    impl LZ4HadoopCodec {
        /// Creates new LZ4 Hadoop compression codec.
        pub(crate) fn new(backward_compatible_lz4: bool) -> Self {
            Self {
                backward_compatible_lz4,
            }
        }
    }

    /// Try to decompress the buffer as if it was compressed with the Hadoop Lz4Codec.
    /// Adapted from pola-rs [compression.rs:try_decompress_hadoop](https://pola-rs.github.io/polars/src/parquet2/compression.rs.html#225)
    /// Translated from the apache arrow c++ function [TryDecompressHadoop](https://github.com/apache/arrow/blob/bf18e6e4b5bb6180706b1ba0d597a65a4ce5ca48/cpp/src/arrow/util/compression_lz4.cc#L474).
    /// Returns error if decompression failed.
    fn try_decompress_hadoop(
        input_buf: &[u8],
        output_buf: &mut Vec<u8>,
        limit: usize,
    ) -> Result<usize> {
        // Parquet files written with the Hadoop Lz4Codec use their own framing.
        // The input buffer can contain an arbitrary number of "frames", each
        // with the following structure:
        // - bytes 0..3: big-endian uint32_t representing the frame decompressed size
        // - bytes 4..7: big-endian uint32_t representing the frame compressed size
        // - bytes 8...: frame compressed data
        //
        // The Hadoop Lz4Codec source code can be found here:
        // https://github.com/apache/hadoop/blob/trunk/hadoop-mapreduce-project/hadoop-mapreduce-client/hadoop-mapreduce-client-nativetask/src/main/native/src/codec/Lz4Codec.cc
        let offset = output_buf.len();
        let mut input = input_buf;
        while !input.is_empty() {
            let prefix = input
                .get(..PREFIX_LEN)
                .ok_or_else(|| general_err!("Not enough bytes for Hadoop frame"))?;
            let expected_decompressed_size =
                u32::from_be_bytes(prefix[..SIZE_U32].try_into().unwrap()) as usize;
            let expected_compressed_size =
                u32::from_be_bytes(prefix[SIZE_U32..].try_into().unwrap()) as usize;
            input = &input[PREFIX_LEN..];
            let compressed = input
                .get(..expected_compressed_size)
                .ok_or_else(|| general_err!("Not enough bytes for Hadoop frame"))?;
            let appended = output_buf.len() - offset;
            if expected_decompressed_size > limit - appended {
                return Err(general_err!("Decompressed data exceeds uncompress_size"));
            }
            LZ4RawCodec::new().decompress(
                compressed,
                output_buf,
                Some(expected_decompressed_size),
            )?;
            input = &input[expected_compressed_size..];
        }
        Ok(output_buf.len() - offset)
    }

    fn is_allocation_error(error: &ParquetError) -> bool {
        matches!(error, ParquetError::External(e) if e.is::<TryReserveError>())
    }

    impl Codec for LZ4HadoopCodec {
        fn decompress(
            &mut self,
            input_buf: &[u8],
            output_buf: &mut Vec<u8>,
            uncompress_size: Option<usize>,
        ) -> Result<usize> {
            let output_len = output_buf.len();
            let Some(required_len) = uncompress_size else {
                return Err(ParquetError::General(
                    "LZ4HadoopCodec unsupported without uncompress_size".into(),
                ));
            };
            let result = match try_decompress_hadoop(input_buf, output_buf, required_len) {
                Ok(n) if n == required_len => Ok(n),
                Ok(_) => Err(general_err!(
                    "LZ4HadoopCodec uncompress_size is not the expected one"
                )),
                Err(e) if !self.backward_compatible_lz4 || is_allocation_error(&e) => Err(e),
                // Preserve compatibility with old arrow-rs and parquet-cpp files.
                Err(_) => {
                    output_buf.truncate(output_len);
                    match LZ4Codec::new().decompress(input_buf, output_buf, uncompress_size) {
                        Ok(n) => Ok(n),
                        Err(e) if is_allocation_error(&e) => Err(e),
                        Err(_) => {
                            output_buf.truncate(output_len);
                            LZ4RawCodec::new().decompress(input_buf, output_buf, uncompress_size)
                        }
                    }
                }
            };
            if result.is_err() {
                output_buf.truncate(output_len);
            }
            result
        }

        fn compress(&mut self, input_buf: &[u8], output_buf: &mut Vec<u8>) -> Result<()> {
            // Allocate memory to store the LZ4_HADOOP prefix.
            let offset = output_buf.len();
            let uncompressed_size = u32::try_from(input_buf.len())?;
            let end = offset
                .checked_add(PREFIX_LEN)
                .ok_or_else(|| general_err!("LZ4 output size overflow"))?;
            resize_output(output_buf, end)?;
            let result = (|| {
                LZ4RawCodec::new().compress(input_buf, output_buf)?;
                let compressed_size = u32::try_from(output_buf.len() - end)?;
                let prefix = &mut output_buf[offset..end];
                prefix[..SIZE_U32].copy_from_slice(&uncompressed_size.to_be_bytes());
                prefix[SIZE_U32..].copy_from_slice(&compressed_size.to_be_bytes());
                Ok(())
            })();
            if result.is_err() {
                output_buf.truncate(offset);
            }
            result
        }
    }
}
#[cfg(any(feature = "lz4", test))]
pub use lz4_hadoop_codec::*;

#[cfg(test)]
mod tests {
    use super::lz4_codec::LZ4Codec;
    use super::*;

    use crate::util::test_common::rand_gen::random_bytes;

    fn test_roundtrip(c: CodecType, data: &[u8], uncompress_size: Option<usize>) {
        let codec_options = CodecOptionsBuilder::default()
            .set_backward_compatible_lz4(false)
            .build();
        let mut c1 = create_codec(c, &codec_options).unwrap().unwrap();
        let mut c2 = create_codec(c, &codec_options).unwrap().unwrap();

        // Compress with c1
        let mut compressed = Vec::new();
        let mut decompressed = Vec::new();
        c1.compress(data, &mut compressed)
            .expect("Error when compressing");

        // Decompress with c2
        let decompressed_size = c2
            .decompress(compressed.as_slice(), &mut decompressed, uncompress_size)
            .expect("Error when decompressing");
        assert_eq!(data.len(), decompressed_size);
        assert_eq!(data, decompressed.as_slice());

        decompressed.clear();
        compressed.clear();

        // Compress with c2
        c2.compress(data, &mut compressed)
            .expect("Error when compressing");

        // Decompress with c1
        let decompressed_size = c1
            .decompress(compressed.as_slice(), &mut decompressed, uncompress_size)
            .expect("Error when decompressing");
        assert_eq!(data.len(), decompressed_size);
        assert_eq!(data, decompressed.as_slice());

        decompressed.clear();
        compressed.clear();

        // Test does not trample existing data in output buffers
        let prefix = &[0xDE, 0xAD, 0xBE, 0xEF];
        decompressed.extend_from_slice(prefix);
        compressed.extend_from_slice(prefix);

        c2.compress(data, &mut compressed)
            .expect("Error when compressing");

        assert_eq!(&compressed[..4], prefix);

        let decompressed_size = c2
            .decompress(&compressed[4..], &mut decompressed, uncompress_size)
            .expect("Error when decompressing");

        assert_eq!(data.len(), decompressed_size);
        assert_eq!(data, &decompressed[4..]);
        assert_eq!(&decompressed[..4], prefix);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Zstd calls native C functions unsupported by Miri
    fn test_decompression_append_limit_and_rollback() {
        let options = CodecOptions::default();
        let kinds = [
            CodecType::SNAPPY,
            CodecType::GZIP(GzipLevel::default()),
            CodecType::BROTLI(BrotliLevel::default()),
            CodecType::LZ4,
            CodecType::LZ4_RAW,
            CodecType::ZSTD(ZstdLevel::default()),
        ];
        let payload = b"bounded page values";
        for kind in kinds {
            let mut codec = create_codec(kind, &options).unwrap().unwrap();
            let mut compressed = Vec::new();
            codec.compress(payload, &mut compressed).unwrap();
            let mut output = Vec::with_capacity(65536);
            output.extend_from_slice(b"levels");
            let prefix = output.clone();
            assert!(
                codec
                    .decompress(&compressed, &mut output, Some(payload.len() - 1))
                    .is_err(),
                "{kind:?}"
            );
            assert_eq!(output, prefix, "{kind:?}");
            let n = codec
                .decompress(&compressed, &mut output, Some(payload.len()))
                .unwrap();
            assert_eq!(n, payload.len(), "{kind:?}");
            assert_eq!(output, [b"levels".as_slice(), payload].concat(), "{kind:?}");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Zstd calls native C functions unsupported by Miri
    fn test_decompression_small_payload_large_declaration() {
        let options = CodecOptionsBuilder::default()
            .set_backward_compatible_lz4(false)
            .build();
        for kind in [
            CodecType::SNAPPY,
            CodecType::GZIP(GzipLevel::default()),
            CodecType::BROTLI(BrotliLevel::default()),
            CodecType::LZ4,
            CodecType::LZ4_RAW,
            CodecType::ZSTD(ZstdLevel::default()),
        ] {
            let mut codec = create_codec(kind, &options).unwrap().unwrap();
            let mut compressed = Vec::new();
            codec.compress(b"value", &mut compressed).unwrap();
            let mut output = b"prefix".to_vec();
            let result = codec.decompress(&compressed, &mut output, Some(usize::MAX));
            if matches!(kind, CodecType::LZ4 | CodecType::LZ4_RAW) {
                assert!(
                    result.unwrap_err().to_string().contains("expected one"),
                    "{kind:?}"
                );
                assert_eq!(output, b"prefix");
            } else {
                assert_eq!(result.unwrap(), 5, "{kind:?}");
                assert_eq!(output, b"prefixvalue");
            }
            // Check destination growth, not the decoder's internal allocations.
            assert!(
                output.capacity() <= 16384,
                "{kind:?}: {}",
                output.capacity()
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Zstd calls native C functions unsupported by Miri
    fn test_decompression_empty_payload() {
        for kind in [
            CodecType::SNAPPY,
            CodecType::GZIP(GzipLevel::default()),
            CodecType::BROTLI(BrotliLevel::default()),
            CodecType::LZ4,
            CodecType::LZ4_RAW,
            CodecType::ZSTD(ZstdLevel::default()),
        ] {
            test_roundtrip(kind, b"", Some(0));
        }
    }

    #[test]
    fn test_snappy_corruption_rollback_and_recovery() {
        let mut codec = SnappyCodec::new();
        let mut output = b"prefix".to_vec();
        // Declares two bytes but provides only a one-byte literal.
        assert!(
            codec
                .decompress(&[2, 0, b'a'], &mut output, Some(2))
                .is_err()
        );
        assert_eq!(output, b"prefix");
        assert_eq!(
            codec.decompress(&[1, 0, b'a'], &mut output, None).unwrap(),
            1
        );
        assert_eq!(output, b"prefixa");
    }

    #[test]
    fn test_snappy_format_bound_preserves_near_maximum_expansion() {
        // A literal seed followed by 64 offset-one Copy2 tokens of 64 bytes
        // approaches the format's maximum expansion.
        let mut compressed = vec![0x81, 0x20, 0x00, b'a']; // Declares 4097 bytes.
        for _ in 0..64 {
            compressed.extend_from_slice(&[0xfe, 0x01, 0x00]);
        }
        let mut codec = SnappyCodec::new();
        let mut output = b"prefix".to_vec();
        assert_eq!(
            codec
                .decompress(&compressed, &mut output, Some(4097))
                .unwrap(),
            4097
        );
        assert_eq!(&output[..6], b"prefix");
        assert_eq!(&output[6..], &[b'a'; 4097]);
    }

    #[test]
    fn test_snappy_impossible_output_size_rejected_before_allocation() {
        let mut codec = SnappyCodec::new();
        // A header-only block declares the largest Parquet page size.
        let header = &[255, 255, 255, 255, 7];
        for limit in [None, Some(i32::MAX as usize)] {
            let mut output = b"prefix".to_vec();
            let capacity = output.capacity();
            let error = codec.decompress(header, &mut output, limit).unwrap_err();
            assert!(matches!(error, ParquetError::General(_)));
            assert!(error.to_string().contains("need at least"));
            assert_eq!(output, b"prefix");
            assert_eq!(output.capacity(), capacity);
        }
    }

    #[test]
    fn test_gzip_members_validate_trailer_at_limit() {
        let mut codec = GZipCodec::new(GzipLevel::default());
        let mut compressed = Vec::new();
        codec.compress(b"first", &mut compressed).unwrap();
        let mut second = Vec::new();
        codec.compress(b"second", &mut second).unwrap();
        compressed.extend_from_slice(&second);
        let mut output = b"prefix".to_vec();
        assert_eq!(
            codec
                .decompress(&compressed, &mut output, Some(11))
                .unwrap(),
            11
        );
        assert_eq!(output, b"prefixfirstsecond");

        let mut empty = Vec::new();
        codec.compress(b"", &mut empty).unwrap();
        let crc = empty.len() - 8;
        empty[crc] ^= 1;
        compressed.extend_from_slice(&empty);
        let mut output = b"prefix".to_vec();
        let error = codec
            .decompress(&compressed, &mut output, Some(11))
            .unwrap_err();
        assert!(error.to_string().contains("corrupt"), "{error}");
        assert_eq!(output, b"prefix");
    }

    #[test]
    fn test_lz4_hadoop_multiple_frames_and_fallbacks() {
        let mut codec = LZ4HadoopCodec::new(false);
        // A smaller second frame exposed the old advancement condition.
        let first = vec![b'a'; 40000];
        let second = b"end";
        let mut compressed = Vec::new();
        codec.compress(&first, &mut compressed).unwrap();
        codec.compress(second, &mut compressed).unwrap();
        let expected = [&first, second.as_slice()].concat();
        let mut output = b"prefix".to_vec();
        assert_eq!(
            codec
                .decompress(&compressed, &mut output, Some(expected.len()))
                .unwrap(),
            expected.len()
        );
        assert_eq!(output, [b"prefix".as_slice(), &expected].concat());

        let mut malformed = compressed.clone();
        malformed.pop();
        for compatible in [false, true] {
            let mut codec = LZ4HadoopCodec::new(compatible);
            let mut output = b"prefix".to_vec();
            assert!(
                codec
                    .decompress(&malformed, &mut output, Some(expected.len()))
                    .is_err()
            );
            assert_eq!(output, b"prefix");
        }

        for mut legacy in [
            Box::new(LZ4Codec::new()) as Box<dyn Codec>,
            Box::new(LZ4RawCodec::new()),
        ] {
            let mut compressed = Vec::new();
            legacy.compress(&first, &mut compressed).unwrap();
            let mut output = b"prefix".to_vec();
            let mut compatible = LZ4HadoopCodec::new(true);
            assert_eq!(
                compatible
                    .decompress(&compressed, &mut output, Some(first.len()))
                    .unwrap(),
                first.len()
            );
            assert_eq!(output, [b"prefix".as_slice(), &first].concat());
            let mut output = b"prefix".to_vec();
            assert!(
                compatible
                    .decompress(&compressed, &mut output, Some(first.len() - 1))
                    .is_err()
            );
            assert_eq!(output, b"prefix");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Zstd calls native C functions unsupported by Miri
    fn test_zstd_frames_and_context_recovery() {
        use std::io::Write;
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_checksum(true).unwrap();
        encoder.write_all(b"first").unwrap();
        let mut compressed = encoder.finish().unwrap();
        let mut corrupt_checksum = compressed.clone();
        *corrupt_checksum.last_mut().unwrap() ^= 1;
        let mut codec = ZSTDCodec::new(ZstdLevel::default());
        let mut output = b"prefix".to_vec();
        let error = codec
            .decompress(&corrupt_checksum, &mut output, Some(5))
            .unwrap_err();
        assert!(error.to_string().contains("checksum"), "{error}");
        assert_eq!(output, b"prefix");
        // Skippable frame followed by a content-sized frame.
        compressed.extend_from_slice(&[0x50, 0x2a, 0x4d, 0x18, 2, 0, 0, 0, b'x', b'y']);
        codec.compress(b"second", &mut compressed).unwrap();
        let mut output = b"prefix".to_vec();
        assert_eq!(
            codec
                .decompress(&compressed, &mut output, Some(11))
                .unwrap(),
            11
        );
        assert_eq!(output, b"prefixfirstsecond");
        let mut output = b"prefix".to_vec();
        let mut truncated = compressed.clone();
        truncated.pop();
        assert!(codec.decompress(&truncated, &mut output, Some(11)).is_err());
        assert_eq!(output, b"prefix");
        assert_eq!(
            codec.decompress(&compressed, &mut output, None).unwrap(),
            11
        );
        assert_eq!(output, b"prefixfirstsecond");
    }

    fn test_codec_with_size(c: CodecType) {
        let sizes = vec![100, 10000, 100000];
        for size in sizes {
            let data = random_bytes(size);
            test_roundtrip(c, &data, Some(data.len()));
        }
    }

    fn test_codec_without_size(c: CodecType) {
        let sizes = vec![100, 10000, 100000];
        for size in sizes {
            let data = random_bytes(size);
            test_roundtrip(c, &data, None);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_codec_snappy() {
        test_codec_with_size(CodecType::SNAPPY);
        test_codec_without_size(CodecType::SNAPPY);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_codec_gzip() {
        for level in GzipLevel::MINIMUM_LEVEL..=GzipLevel::MAXIMUM_LEVEL {
            let level = GzipLevel::try_new(level).unwrap();
            test_codec_with_size(CodecType::GZIP(level));
            test_codec_without_size(CodecType::GZIP(level));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_codec_brotli() {
        for level in BrotliLevel::MINIMUM_LEVEL..=BrotliLevel::MAXIMUM_LEVEL {
            let level = BrotliLevel::try_new(level).unwrap();
            test_codec_with_size(CodecType::BROTLI(level));
            test_codec_without_size(CodecType::BROTLI(level));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_codec_lz4() {
        test_codec_with_size(CodecType::LZ4);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Zstd calls native C functions unsupported by Miri
    fn test_codec_zstd() {
        // since ZstdLevel::MINIMUM_LEVEL is a large negative number, we test a smaller range
        for level in
            std::iter::once(ZstdLevel::MINIMUM_LEVEL).chain(-100..=ZstdLevel::MAXIMUM_LEVEL)
        {
            let level = ZstdLevel::try_new(level).unwrap();
            test_codec_with_size(CodecType::ZSTD(level));
            test_codec_without_size(CodecType::ZSTD(level));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_codec_lz4_raw() {
        test_codec_with_size(CodecType::LZ4_RAW);
    }
}
