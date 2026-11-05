// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Bloom filter implementation specific to Parquet, as described
//! in the [spec][parquet-bf-spec].
//!
//! # Bloom Filter Size
//!
//! Parquet uses the [Split Block Bloom Filter][sbbf-paper] (SBBF) as its bloom filter
//! implementation. For each column upon which bloom filters are enabled, the offset and length of an SBBF
//! is stored in  the metadata for each row group in the parquet file. The size of each filter is
//! initialized using a calculation based on the desired number of distinct values (NDV) and false
//! positive probability (FPP). The FPP for a SBBF can be approximated as<sup>[1][bf-formulae]</sup>:
//!
//! ```text
//! f = (1 - e^(-k * n / m))^k
//! ```
//!
//! Where, `f` is the FPP, `k` the number of hash functions, `n` the NDV, and `m` the total number
//! of bits in the bloom filter. This can be re-arranged to determine the total number of bits
//! required to achieve a given FPP and NDV:
//!
//! ```text
//! m = -k * n / ln(1 - f^(1/k))
//! ```
//!
//! SBBFs use eight hash functions to cleanly fit in SIMD lanes<sup>[2][sbbf-paper]</sup>, therefore
//! `k` is set to 8. The SBBF will spread those `m` bits across a set of `b` blocks that
//! are each 256 bits, i.e., 32 bytes, in size. The number of blocks is chosen as:
//!
//! ```text
//! b = NP2(m/8) / 32
//! ```
//!
//! Where, `NP2` denotes *the next power of two*, and `m` is divided by 8 to be represented as bytes.
//!
//! Here is a table of calculated sizes for various FPP and NDV:
//!
//! | NDV       | FPP       | b       | Size (KB) |
//! |-----------|-----------|---------|-----------|
//! | 10,000    | 0.1       | 256     | 8         |
//! | 10,000    | 0.01      | 512     | 16        |
//! | 10,000    | 0.001     | 1,024   | 32        |
//! | 10,000    | 0.0001    | 1,024   | 32        |
//! | 100,000   | 0.1       | 4,096   | 128       |
//! | 100,000   | 0.01      | 4,096   | 128       |
//! | 100,000   | 0.001     | 8,192   | 256       |
//! | 100,000   | 0.0001    | 16,384  | 512       |
//! | 100,000   | 0.00001   | 16,384  | 512       |
//! | 1,000,000 | 0.1       | 32,768  | 1,024     |
//! | 1,000,000 | 0.01      | 65,536  | 2,048     |
//! | 1,000,000 | 0.001     | 65,536  | 2,048     |
//! | 1,000,000 | 0.0001    | 131,072 | 4,096     |
//! | 1,000,000 | 0.00001   | 131,072 | 4,096     |
//! | 1,000,000 | 0.000001  | 262,144 | 8,192     |
//!
//! # Structure: Filter → Blocks → Words → Bits
//!
//! An SBBF is an array of **blocks**. Each block is 256 bits (32 bytes),
//! divided into eight 32-bit **words**. A word is just a `u32` — an array of
//! 32 individual bits that can each be "set" (1) or "not set" (0).
//!
//! ```text
//!   Sbbf (the whole filter)
//!   ┌──────────┬──────────┬──────────┬─── ─── ──┬──────────┐
//!   │ Block 0  │ Block 1  │ Block 2  │   ...    │ Block N-1│
//!   └──────────┴──────────┴──────────┴─── ─── ──┴──────────┘
//!        │
//!        ▼
//!   One Block = 256 bits = 8 words
//!   ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
//!   │ word 0 │ word 1 │ word 2 │ word 3 │ word 4 │ word 5 │ word 6 │ word 7 │
//!   │ (u32)  │ (u32)  │ (u32)  │ (u32)  │ (u32)  │ (u32)  │ (u32)  │ (u32)  │
//!   └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
//!        │
//!        ▼
//!   One Word = 32 individual bits
//!   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
//!   │0│0│1│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│  ← bit 29 is set
//!   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
//! ```
//!
//! **Inserting** a value hashes it to a 64-bit number, then:
//!  1. The upper 32 bits pick which **block** (via `Sbbf::hash_to_block_index`).
//!  2. The lower 32 bits pick one bit position in each of the 8 **words** (via `Block::mask`).
//!     So each insert sets exactly **8 bits** (one per word) in a single block.
//!
//! **Checking** does the same two steps and returns `true` only if all 8 bits
//! are already set — meaning the value was *probably* inserted (or is a false
//! positive).
//!
//! # Bloom Filter Folding
//!
//! After inserting all values into a bloom filter it can be "folded" to minimize it's size.
//! See [`Sbbf::fold_to_target_fpp`] for details  on the algorithm and its mathematical basis.
//!
//! [parquet-bf-spec]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md
//! [sbbf-paper]: https://arxiv.org/pdf/2101.01719
//! [bf-formulae]: http://tfk.mit.edu/pdf/bloom.pdf

#[cfg(all(feature = "arrow", feature = "async"))]
use crate::arrow::async_reader::AsyncFileReader;
use crate::basic::{BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHash};
use crate::data_type::AsBytes;
use crate::errors::{ParquetError, Result};
use crate::file::metadata::ColumnChunkMetaData;
use crate::file::reader::ChunkReader;
use crate::parquet_thrift::{
    ElementType, FieldType, ReadThrift, ThriftCompactInputProtocol, ThriftCompactOutputProtocol,
    ThriftSliceInputProtocol, WriteThrift, WriteThriftField,
};
use crate::{DecodeResult, thrift_struct};
use bytes::Bytes;
use parquet_bloom_filter::Sbbf as CoreSbbf;
use std::io::Write;
use std::ops::Range;

pub use parquet_bloom_filter::{BITSET_MAX_LENGTH, BITSET_MIN_LENGTH};

mod push;
pub use push::SbbfPushDecoder;

thrift_struct!(
/// Bloom filter header is stored at beginning of Bloom filter data of each column
/// and followed by its bitset.
///
pub struct BloomFilterHeader {
  /// The size of bitset in bytes
  1: required i32 num_bytes;
  /// The algorithm for setting bits.
  2: required BloomFilterAlgorithm algorithm;
  /// The hash function used for Bloom filter
  3: required BloomFilterHash hash;
  /// The compression used in the Bloom filter
  4: required BloomFilterCompression compression;
}
);

/// A split block Bloom filter (SBBF).
///
/// An SBBF partitions its bit space into fixed-size 256-bit (32-byte) blocks, each fitting in a
/// single CPU cache line. Each block contains eight 32-bit words, aligned with SIMD lanes for
/// parallel bit manipulation. When checking membership, only one block is accessed per query,
/// eliminating the cache-miss penalty of standard Bloom filters.
///
/// ## Sizing and folding
///
/// Filters are initially sized for a maximum expected number of distinct values (NDV) via
/// [`Sbbf::new_with_ndv_fpp`]. After all values are inserted, the filter is compacted by
/// calling [`Sbbf::fold_to_target_fpp`], which folds the filter down to the smallest size
/// that still meets the target false positive probability.
///
/// The creation of this structure is based on the [`crate::file::properties::BloomFilterProperties`]
/// struct set via [`crate::file::properties::WriterProperties`] and is thus hidden by default.
#[derive(Debug, Clone)]
pub struct Sbbf(CoreSbbf);

impl Sbbf {
    /// Create a new [Sbbf] with given number of distinct values and false positive probability.
    /// Returns an error if `fpp` is outside `[0.0, 1.0)` or NaN.
    #[inline]
    pub fn new_with_ndv_fpp(ndv: u64, fpp: f64) -> Result<Self, ParquetError> {
        CoreSbbf::new_with_ndv_fpp(ndv, fpp)
            .map(Self)
            .map_err(|e| ParquetError::General(e.to_string()))
    }

    /// Create a new [Sbbf] with given number of bytes, the exact number of bytes will be adjusted
    /// to the next power of two bounded by [BITSET_MIN_LENGTH] and [BITSET_MAX_LENGTH].
    #[inline]
    pub fn new_with_num_of_bytes(num_bytes: usize) -> Self {
        Self(CoreSbbf::new_with_num_of_bytes(num_bytes))
    }

    /// Creates a new [Sbbf] from a raw byte slice.
    ///
    /// Trailing incomplete 32-byte blocks are ignored. Fewer than 32 bytes create
    /// a zero-block filter, on which insertion and membership checks panic.
    #[inline]
    pub fn new(bitset: &[u8]) -> Self {
        Self(CoreSbbf::new(bitset))
    }

    /// Write the bloom filter data (header and then bitset) to the output. This doesn't
    /// flush the writer in order to boost performance of bulk writing all blocks. Caller
    /// must remember to flush the writer.
    /// This method usually is used in conjunction with [`Self::from_bytes`] for serialization/deserialization.
    pub fn write<W: Write>(&self, mut writer: W) -> Result<(), ParquetError> {
        let mut protocol = ThriftCompactOutputProtocol::new(&mut writer);
        self.header().write_thrift(&mut protocol).map_err(|e| {
            ParquetError::General(format!("Could not write bloom filter header: {e}"))
        })?;
        self.write_bitset(&mut writer)?;
        Ok(())
    }

    /// Write the bitset in serialized form to the writer.
    #[inline]
    pub fn write_bitset<W: Write>(&self, writer: W) -> Result<(), ParquetError> {
        self.0.write_bitset(writer).map_err(|e| {
            ParquetError::General(format!("Could not write bloom filter bit set: {e}"))
        })
    }

    /// Create and populate [`BloomFilterHeader`] from this bitset for writing to serialized form
    fn header(&self) -> BloomFilterHeader {
        BloomFilterHeader {
            // 8 i32 per block, 4 bytes per i32
            num_bytes: self.num_blocks() as i32 * 4 * 8,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        }
    }

    /// Read a Bloom filter from existing column-chunk metadata.
    ///
    /// The metadata and reader must refer to the same file. A missing offset
    /// returns `None` without I/O. Invalid metadata, malformed filters, and read
    /// failures return an error.
    ///
    /// When the total filter length is present, one range is fetched. Otherwise,
    /// the header prefix and bitset are fetched separately.
    pub fn read_from_column_chunk<R: ChunkReader>(
        column_metadata: &ColumnChunkMetaData,
        reader: &R,
    ) -> Result<Option<Self>> {
        let Some(mut decoder) = SbbfPushDecoder::try_new(column_metadata)? else {
            return Ok(None);
        };

        loop {
            let result = decoder.try_decode()?;
            match result {
                DecodeResult::NeedsData(ranges) => {
                    for range in ranges {
                        let buffer = read_bloom_filter_bytes(reader, &range)?;
                        decoder.push_range(range, buffer)?;
                    }
                }
                DecodeResult::Data(sbbf) => return Ok(Some(sbbf)),
                DecodeResult::Finished => {
                    return Err(general_err!("invalid value returned from SbbfPushDecoder"));
                }
            }
        }
    }

    /// Read a Bloom filter asynchronously from existing column-chunk metadata.
    ///
    /// The metadata and reader must refer to the same file.
    ///
    /// A missing Bloom filter offset returns `None` without performing I/O. Invalid
    /// metadata, malformed filters, and read failures return an error.
    ///
    /// When the total filter length is present, one range is fetched. Otherwise,
    /// this may perform additional fetches.
    #[cfg(all(feature = "arrow", feature = "async"))]
    pub async fn read_from_column_chunk_async<R: AsyncFileReader + ?Sized>(
        column_metadata: &ColumnChunkMetaData,
        reader: &mut R,
    ) -> Result<Option<Self>> {
        let Some(mut decoder) = SbbfPushDecoder::try_new(column_metadata)? else {
            return Ok(None);
        };

        loop {
            let result = decoder.try_decode()?;
            match result {
                DecodeResult::NeedsData(ranges) => {
                    for range in ranges {
                        let buffer = read_bloom_filter_bytes_async(reader, &range).await?;
                        decoder.push_range(range, buffer)?;
                    }
                }
                DecodeResult::Data(sbbf) => return Ok(Some(sbbf)),
                DecodeResult::Finished => {
                    return Err(general_err!("invalid value returned from SbbfPushDecoder"));
                }
            }
        }
    }

    /// Insert an [AsBytes] value into the filter
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn insert<T: AsBytes + ?Sized>(&mut self, value: &T) {
        self.0.insert_bytes(value.as_bytes())
    }

    /// Check if an [AsBytes] value is probably present or definitely absent in the filter
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn check<T: AsBytes + ?Sized>(&self, value: &T) -> bool {
        self.0.check_bytes(value.as_bytes())
    }

    /// Return the total in memory size of this bloom filter in bytes
    #[inline]
    pub(crate) fn estimated_memory_size(&self) -> usize {
        self.0.estimated_memory_size()
    }

    /// Returns the number of blocks in this bloom filter.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.0.num_blocks()
    }

    /// Estimate the false positive probability (FPP) of this filter at its current size.
    ///
    /// This is the same estimate [`Self::fold_to_target_fpp`] uses to choose how far to fold.
    ///
    /// This lets a caller inspect a filter before folding or writing it, for example to
    /// discard a filter that already exceeds its target FPP. Returns `1.0` for a filter
    /// with no blocks.
    #[inline]
    pub fn estimated_fpp(&self) -> f64 {
        self.0.estimated_fpp()
    }

    /// Fold the bloom filter down to the smallest size that still meets the target FPP
    /// (False Positive Percentage).
    ///
    /// Folds the filter by merging groups of adjacent blocks via bitwise OR, where each
    /// fold level halves the number of blocks. The fold count is chosen as the maximum
    /// number of folds whose estimated FPP stays within `target_fpp`. The filter stops
    /// at a minimum size of 1 block (32 bytes).
    ///
    /// ## How it works
    ///
    /// SBBFs use multiplicative hashing for block selection:
    ///
    /// ```text
    /// block_index = ((hash >> 32) * num_blocks) >> 32
    /// ```
    ///
    /// A single fold halves the block count: when `num_blocks` is halved, the new index
    /// becomes `floor(original_index / 2)`, so blocks `2i` and `2i+1` map to the same
    /// position. More generally, `k` folds reduce the block count by `2^k`, merging
    /// groups of `2^k` adjacent blocks in a single pass:
    ///
    /// ```text
    /// folded[i] = blocks[i*2^k] | blocks[i*2^k + 1] | ... | blocks[i*2^k + 2^k - 1]
    /// ```
    ///
    /// This differs from standard Bloom filter folding, which merges the two halves
    /// (`B[i] | B[i + m/2]`) because standard filters use modular hashing where
    /// `h(x) mod (m/2)` maps indices `i` and `i + m/2` to the same position.
    ///
    /// ## Correctness
    ///
    /// Folding **never introduces false negatives**. Every bit that was set in the original
    /// filter remains set in the folded filter (via bitwise OR). The only effect is a controlled
    /// increase in FPP as set bits from different blocks are merged together.
    /// This is was originally proven in [Sailhan & Stehr 2012] for standard bloom filters and is empirically
    /// demonstrated for SBBFs in Lemma 1 and Lemma 2 of the tests.
    ///
    /// ## References
    ///
    /// [Sailhan & Stehr 2012]: https://doi.org/10.1109/GreenCom.2012.16
    ///
    /// # Panics
    ///
    /// Panics for populated non-power-of-two block counts with at least two blocks.
    /// The target FPP is not validated.
    #[inline]
    pub fn fold_to_target_fpp(&mut self, target_fpp: f64) {
        self.0.fold_to_target_fpp(target_fpp)
    }

    /// Reads an Sbbf from Thrift encoded bytes.
    ///
    /// Returns an error if the header is malformed, the bitset is not a positive
    /// multiple of 32 bytes, or its size does not match the header.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use parquet::errors::Result;
    /// # use parquet::bloom_filter::Sbbf;
    /// # fn main() -> Result<()> {
    /// // In a real application, you would read serialized bloom filter bytes from a cache.
    /// // This example demonstrates the deserialization process.
    /// // Assuming you have bloom filter bytes from a Parquet file:
    /// # let serialized_bytes: Vec<u8> = vec![];
    /// let bloom_filter = Sbbf::from_bytes(&serialized_bytes)?;
    /// // Now you can use the bloom filter to check for values
    /// if bloom_filter.check(&"some_value") {
    ///     println!("Value might be present (or false positive)");
    /// } else {
    ///     println!("Value is definitely not present");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ParquetError> {
        let range = bloom_filter_bitset_range(bytes, Some(bytes.len()))?;
        Ok(Self::new(&bytes[range]))
    }
}

/// Move a standalone filter into its Parquet facade without copying the bitset.
impl From<CoreSbbf> for Sbbf {
    #[inline]
    fn from(filter: CoreSbbf) -> Self {
        Self(filter)
    }
}

/// Move the bitset out of its Parquet facade without copying it.
impl From<Sbbf> for CoreSbbf {
    #[inline]
    fn from(filter: Sbbf) -> Self {
        filter.0
    }
}

const SBBF_HEADER_SIZE_ESTIMATE: usize = 20;

fn read_bloom_filter_bytes<R: ChunkReader>(reader: &R, range: &Range<u64>) -> Result<Bytes> {
    let length = usize::try_from(range.end - range.start)?;
    validate_bloom_filter_read(reader.get_bytes(range.start, length)?, length)
}

#[cfg(all(feature = "arrow", feature = "async"))]
async fn read_bloom_filter_bytes_async<R: AsyncFileReader + ?Sized>(
    reader: &mut R,
    range: &Range<u64>,
) -> Result<Bytes> {
    let length = usize::try_from(range.end - range.start)?;
    validate_bloom_filter_read(reader.get_bytes(range.clone()).await?, length)
}

/// Parses the header and returns the bitset's range relative to the filter start.
/// When the complete filter length is known, the bitset must consume it exactly.
fn bloom_filter_bitset_range(buffer: &[u8], length: Option<usize>) -> Result<Range<usize>> {
    let (header, header_length) = read_bloom_filter_header_and_length(buffer)?;

    // Force an explicit decision if supported variants change.
    match header.algorithm {
        BloomFilterAlgorithm::BLOCK => {}
    }
    match header.hash {
        BloomFilterHash::XXHASH => {}
    }
    match header.compression {
        BloomFilterCompression::UNCOMPRESSED => {}
    }

    let bitset_length = usize::try_from(header.num_bytes)
        .map_err(|_| general_err!("Bloom filter bitset length is invalid"))?;
    if bitset_length < BITSET_MIN_LENGTH || !bitset_length.is_multiple_of(BITSET_MIN_LENGTH) {
        return Err(general_err!(
            "Bloom filter bitset length must be a positive multiple of 32"
        ));
    }
    let end = header_length
        .checked_add(bitset_length)
        .ok_or_else(|| general_err!("Bloom filter length overflow"))?;
    if length.is_some_and(|length| length != end) {
        return Err(general_err!(
            "Bloom filter length does not match its header"
        ));
    }
    Ok(header_length..end)
}

/// Converts a filter-relative range to a file range without overflowing.
fn bloom_filter_read_range(offset: u64, range: Range<usize>) -> Result<Range<u64>> {
    let start = offset
        .checked_add(range.start as u64)
        .ok_or_else(|| general_err!("Bloom filter read range overflow"))?;
    let end = offset
        .checked_add(range.end as u64)
        .ok_or_else(|| general_err!("Bloom filter read range overflow"))?;
    Ok(start..end)
}

fn validate_bloom_filter_read(bytes: Bytes, length: usize) -> Result<Bytes> {
    if bytes.len() != length {
        return Err(general_err!(
            "Bloom filter read length mismatch: expected {length} bytes, got {}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// Parses a Bloom filter header and returns the number of bytes consumed.
fn read_bloom_filter_header_and_length(buffer: &[u8]) -> Result<(BloomFilterHeader, usize)> {
    let mut prot = ThriftSliceInputProtocol::new(buffer);
    let header = BloomFilterHeader::read_thrift(&mut prot)
        .map_err(|e| ParquetError::General(format!("Could not read bloom filter header: {e}")))?;
    Ok((header, buffer.len() - prot.as_slice().len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_conversions_preserve_bitset_and_memory_size() {
        let mut core = CoreSbbf::new_with_num_of_bytes(4096);
        for value in 0i32..100 {
            core.insert_bytes(&value.to_ne_bytes());
        }
        let memory_size = core.estimated_memory_size();
        let mut expected = Vec::new();
        core.write_bitset(&mut expected).unwrap();
        let mut filter = Sbbf::from(core);
        assert_eq!(filter.estimated_memory_size(), memory_size);
        assert_eq!(filter.num_blocks(), 128);
        for value in 0i32..100 {
            assert!(filter.check(&value));
        }
        let mut output = Vec::new();
        filter.write_bitset(&mut output).unwrap();
        assert_eq!(output, expected);
        filter.insert("facade_value");
        let core = CoreSbbf::from(filter);
        assert_eq!(core.estimated_memory_size(), memory_size);
        assert!(core.check_bytes(b"facade_value"));
    }

    #[test]
    fn test_core_invalid_fpp_error_mapping() {
        for fpp in [-1.0, 1.0, f64::INFINITY, f64::NAN] {
            let error = Sbbf::new_with_ndv_fpp(100, fpp).unwrap_err();
            assert!(matches!(
                error,
                ParquetError::General(message) if message == format!(
                    "False positive probability must be between 0.0 and 1.0, got {fpp}"
                )
            ));
        }
    }

    #[test]
    fn test_core_write_bitset_error_mapping() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("write failed"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let filter = Sbbf::new_with_num_of_bytes(32);
        assert!(matches!(
            filter.write_bitset(FailingWriter).unwrap_err(),
            ParquetError::General(message)
                if message == "Could not write bloom filter bit set: write failed"
        ));
    }

    #[test]
    fn test_bloom_filter_read_range_overflow() {
        assert!(bloom_filter_read_range(u64::MAX, 0..1).is_err());
        assert!(bloom_filter_read_range(u64::MAX, 1..2).is_err());
    }

    #[test]
    fn test_with_fixture() {
        // bloom filter produced by parquet-mr/spark for a column of i64 f"a{i}" for i in 0..10
        let bitset: &[u8] = &[
            200, 1, 80, 20, 64, 68, 8, 109, 6, 37, 4, 67, 144, 80, 96, 32, 8, 132, 43, 33, 0, 5,
            99, 65, 2, 0, 224, 44, 64, 78, 96, 4,
        ];
        let sbbf = Sbbf::new(bitset);
        for a in 0..10i64 {
            let value = format!("a{a}");
            assert!(sbbf.check(&value.as_str()));
        }
    }

    /// test the assumption that bloom filter header size should not exceed SBBF_HEADER_SIZE_ESTIMATE
    /// essentially we are testing that the struct is packed with 4 i32 fields, each can be 1-5 bytes
    /// so altogether it'll be 20 bytes at most.
    #[test]
    fn test_bloom_filter_header_size_assumption() {
        let buffer: &[u8; 16] = &[21, 64, 28, 28, 0, 0, 28, 28, 0, 0, 28, 28, 0, 0, 0, 99];
        let (
            BloomFilterHeader {
                algorithm,
                compression,
                hash,
                num_bytes,
            },
            read_length,
        ) = read_bloom_filter_header_and_length(buffer).unwrap();
        assert_eq!(read_length, 15);
        assert_eq!(algorithm, BloomFilterAlgorithm::BLOCK);
        assert_eq!(compression, BloomFilterCompression::UNCOMPRESSED);
        assert_eq!(hash, BloomFilterHash::XXHASH);
        assert_eq!(num_bytes, 32_i32);
        assert_eq!(20, SBBF_HEADER_SIZE_ESTIMATE);
    }

    #[test]
    fn test_sbbf_write_round_trip() {
        // Create a bloom filter with a 32-byte bitset (minimum size)
        let bitset_bytes = vec![0u8; 32];
        let mut original = Sbbf::new(&bitset_bytes);

        // Insert some test values
        let test_values = ["hello", "world", "rust", "parquet", "bloom", "filter"];
        for value in &test_values {
            original.insert(value);
        }

        // Serialize to bytes
        let mut output = Vec::new();
        original.write(&mut output).unwrap();

        // Validate header was written correctly
        let mut protocol = ThriftSliceInputProtocol::new(&output);
        let header = BloomFilterHeader::read_thrift(&mut protocol).unwrap();
        assert_eq!(header.num_bytes, bitset_bytes.len() as i32);
        assert_eq!(header.algorithm, BloomFilterAlgorithm::BLOCK);
        assert_eq!(header.hash, BloomFilterHash::XXHASH);
        assert_eq!(header.compression, BloomFilterCompression::UNCOMPRESSED);

        // Deserialize using from_bytes
        let reconstructed = Sbbf::from_bytes(&output).unwrap();

        // Most importantly: verify the bloom filter WORKS correctly after round-trip
        // Note: bloom filters can have false positives, but should never have false negatives
        // So we can't assert !check(), but we should verify inserted values are found
        for value in &test_values {
            assert!(
                reconstructed.check(value),
                "Value '{value}' should be present after round-trip"
            );
        }
    }
}
