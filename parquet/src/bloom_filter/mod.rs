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

use crate::basic::{BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHash};
use crate::data_type::AsBytes;
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{ColumnChunkMetaData, HeapSize};
use crate::file::reader::ChunkReader;
use crate::parquet_thrift::{
    ElementType, FieldType, ReadThrift, ThriftCompactInputProtocol, ThriftCompactOutputProtocol,
    ThriftSliceInputProtocol, WriteThrift, WriteThriftField,
};
use crate::thrift_struct;
use bytes::Bytes;
use std::io::Write;
use std::ops::Range;
use twox_hash::XxHash64;

/// Salt as defined in the [spec](https://github.com/apache/parquet-format/blob/master/BloomFilter.md#technical-approach).
const SALT: [u32; 8] = [
    0x47b6137b_u32,
    0x44974d91_u32,
    0x8824ad5b_u32,
    0xa2b7289d_u32,
    0x705495c7_u32,
    0x2df1424b_u32,
    0x9efc4947_u32,
    0x5c6bfb31_u32,
];

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

/// A single 256-bit block, the basic unit of the Split Block Bloom Filter.
///
/// A block is eight contiguous 32-bit **words** (`[u32; 8]`).
/// Each word is an independent bit-array of 32 positions:
///
/// ```text
///   Block (256 bits total)
///   ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
///   │ word 0 │ word 1 │ word 2 │ word 3 │ word 4 │ word 5 │ word 6 │ word 7 │
///   │ 32 bits│ 32 bits│ 32 bits│ 32 bits│ 32 bits│ 32 bits│ 32 bits│ 32 bits│
///   └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
/// ```
///
/// When a value is inserted, [`Block::mask`] picks one bit in each word
/// (8 bits total), and those bits are OR'd in. When checking, we verify
/// all 8 bits are set.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
struct Block([u32; 8]);
impl Block {
    const ZERO: Block = Block([0; 8]);

    /// Produce a block where each of the 8 words has exactly one bit set.
    ///
    /// For each word `i` the bit position is derived from `x`:
    ///
    /// ```text
    ///   y = (x wrapping* SALT[i]) >> 27   // top 5 bits → value in 0..31
    ///   word[i] = 1 << y                  // exactly one bit set per word
    /// ```
    ///
    /// Because only the top 5 bits survive the shift, each word picks one of
    /// 32 possible bit positions. The eight SALT constants spread the choices
    /// so different words usually light up different positions.
    ///
    /// Key property: the mask depends *only* on `x` (a u32) and the fixed
    /// SALT constants — it is independent of the filter size. This is why
    /// folding preserves bit patterns (see Lemma 2 in tests).
    fn mask(x: u32) -> Self {
        let mut result = [0_u32; 8];
        for i in 0..8 {
            let y = x.wrapping_mul(SALT[i]); // spread bits via multiply
            let y = y >> 27; // keep top 5 bits → 0..31
            result[i] = 1 << y; // set exactly that one bit
        }
        Self(result)
    }

    #[inline]
    #[cfg(not(target_endian = "little"))]
    fn to_ne_bytes(self) -> [u8; 32] {
        // SAFETY: [u32; 8] and [u8; 32] have the same size and neither has invalid bit patterns.
        unsafe { std::mem::transmute(self.0) }
    }

    #[inline]
    #[cfg(not(target_endian = "little"))]
    fn to_le_bytes(self) -> [u8; 32] {
        self.swap_bytes().to_ne_bytes()
    }

    #[inline]
    #[cfg(not(target_endian = "little"))]
    fn swap_bytes(mut self) -> Self {
        self.0.iter_mut().for_each(|x| *x = x.swap_bytes());
        self
    }

    /// OR the mask bits into this block (`block[i] |= mask[i]`).
    ///
    /// After insertion the 8 bits chosen by `mask(hash)` are guaranteed set;
    /// bits previously set by other hashes are preserved.
    fn insert(&mut self, hash: u32) {
        let mask = Self::mask(hash);
        for i in 0..8 {
            self[i] |= mask[i];
        }
    }

    /// Check membership: returns `true` when *every* bit from `mask(hash)` is
    /// already set in this block (`block[i] & mask[i] != 0` for all 8 words).
    ///
    /// A `true` result means "probably present" (other inserts may have set
    /// the same bits). A `false` is definitive — the value was never inserted.
    fn check(&self, hash: u32) -> bool {
        let mask = Self::mask(hash);
        for i in 0..8 {
            if self[i] & mask[i] == 0 {
                return false;
            }
        }
        true
    }
}

impl std::ops::Index<usize> for Block {
    type Output = u32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl std::ops::IndexMut<usize> for Block {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl std::ops::BitOr for Block {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        let mut result = [0u32; 8];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.0[i] | rhs.0[i];
        }
        Self(result)
    }
}

impl std::ops::BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] |= rhs.0[i];
        }
    }
}

impl Block {
    /// Count the total number of set bits across all 8 words.
    ///
    /// Computes popcount on each word separately and sums. Keeping the popcount
    /// separate from the OR allows the compiler to batch SIMD popcount instructions
    /// (e.g., `cnt.16b` on ARM NEON) instead of interleaving them with OR operations.
    #[inline]
    fn count_ones(self) -> u32 {
        // Written as a fold over the array so the compiler sees 8 independent
        // popcount operations it can vectorize into cnt.16b + horizontal sum.
        self.0.iter().map(|w| w.count_ones()).sum()
    }
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sbbf(Vec<Block>);

impl HeapSize for Sbbf {
    fn heap_size(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<Block>()
    }
}

pub(crate) const SBBF_HEADER_SIZE_ESTIMATE: usize = 20;

/// Information needed to read a Bloom filter from a file.
///
/// This struct is returned by [`decode_bloom_filter_header`] and contains
/// all the validated offsets and ranges needed to read the filter's bitset.
/// Temporary expect for Phase 1: These fields will be used in Phase 4 (batch loading)
/// and Phase 6 (cache integration). Remove this expect in Phase 4.
#[expect(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct BloomFilterReadInfo {
    /// The file offset where the Bloom filter starts (including header)
    pub(crate) filter_offset: u64,
    /// Length of the parsed header in bytes
    pub(crate) header_len: u64,
    /// Byte range in the file where the bitset data is located
    pub(crate) bitset_range: Range<u64>,
    /// Total length of the filter (header + bitset) in bytes
    pub(crate) total_len: u64,
}

/// Decode and validate a Bloom filter header.
///
/// This function parses the Thrift-encoded header, validates its contents,
/// and returns information needed to read the bitset.
///
/// # Arguments
/// * `filter_offset` - File offset where the Bloom filter starts
/// * `header_bytes` - Buffer containing at least the header (may contain more)
/// * `advertised_total_len` - Optional total length from column metadata's `bloom_filter_length`
///
/// # Validation
/// This function validates:
/// - The header can be successfully decoded
/// - Algorithm is `BLOCK`, compression is `UNCOMPRESSED`, hash is `XXHASH`
/// - `num_bytes` is non-negative and fits in `usize`
/// - Bitset length meets requirements (size, alignment, power-of-two)
/// - If `advertised_total_len` is provided, it matches `header_len + num_bytes`
/// - All offset arithmetic is checked for overflow
///
/// # Errors
/// Returns an error if any validation fails or arithmetic overflows.
pub(crate) fn decode_bloom_filter_header(
    filter_offset: u64,
    header_bytes: Bytes,
    advertised_total_len: Option<usize>,
) -> Result<BloomFilterReadInfo, ParquetError> {
    decode_bloom_filter_header_from_slice(
        filter_offset,
        header_bytes.as_ref(),
        advertised_total_len,
    )
}

fn decode_bloom_filter_header_from_slice(
    filter_offset: u64,
    header_bytes: &[u8],
    advertised_total_len: Option<usize>,
) -> Result<BloomFilterReadInfo, ParquetError> {
    let (header, header_len) = read_bloom_filter_header_and_length_from_bytes(header_bytes)
        .map_err(|err| {
            ParquetError::General(format!(
                "Could not read bloom filter header at offset {filter_offset}: {err}"
            ))
        })?;

    // Validate algorithm, compression, and hash
    match header.algorithm {
        BloomFilterAlgorithm::BLOCK => {
            // Currently the only supported algorithm
        }
    }
    match header.compression {
        BloomFilterCompression::UNCOMPRESSED => {
            // Currently the only supported compression
        }
    }
    match header.hash {
        BloomFilterHash::XXHASH => {
            // Currently the only supported hash
        }
    }

    // Validate and convert num_bytes
    let num_bytes: usize = header.num_bytes.try_into().map_err(|_| {
        ParquetError::General(format!(
            "Bloom filter num_bytes is negative or too large: {}",
            header.num_bytes
        ))
    })?;

    validate_bitset_len(num_bytes)?;

    let num_bytes_u64 = u64::try_from(num_bytes).map_err(|_| {
        ParquetError::General(format!("Bloom filter bitset length too large: {num_bytes}"))
    })?;

    let total_len = header_len.checked_add(num_bytes_u64).ok_or_else(|| {
        ParquetError::General(format!(
            "Bloom filter total length overflow: header_len {header_len} + num_bytes {num_bytes}"
        ))
    })?;

    if let Some(advertised) = advertised_total_len {
        let advertised_u64 = u64::try_from(advertised).map_err(|_| {
            ParquetError::General(format!(
                "Advertised bloom filter length too large: {advertised}"
            ))
        })?;

        if advertised_u64 != total_len {
            return Err(ParquetError::General(format!(
                "Bloom filter length mismatch at offset {filter_offset}: advertised {advertised} bytes, but header indicates {total_len} bytes"
            )));
        }
    }

    let bitset_start = filter_offset.checked_add(header_len).ok_or_else(|| {
        ParquetError::General(format!(
            "Bloom filter bitset offset overflow: filter_offset {filter_offset} + header_len {header_len}"
        ))
    })?;

    let bitset_end = bitset_start.checked_add(num_bytes_u64).ok_or_else(|| {
        ParquetError::General(format!(
            "Bloom filter bitset end overflow: bitset_start {bitset_start} + num_bytes {num_bytes}"
        ))
    })?;

    Ok(BloomFilterReadInfo {
        filter_offset,
        header_len,
        bitset_range: bitset_start..bitset_end,
        total_len,
    })
}

fn validate_bitset_len(len: usize) -> Result<(), ParquetError> {
    if len < BITSET_MIN_LENGTH {
        return Err(ParquetError::General(format!(
            "Bloom filter bitset too small: {len} bytes (minimum: {BITSET_MIN_LENGTH})"
        )));
    }
    if len > BITSET_MAX_LENGTH {
        return Err(ParquetError::General(format!(
            "Bloom filter bitset too large: {len} bytes (maximum: {BITSET_MAX_LENGTH})"
        )));
    }
    if !len.is_multiple_of(size_of::<Block>()) {
        return Err(ParquetError::General(format!(
            "Bloom filter bitset length {} is not a multiple of block size ({})",
            len,
            size_of::<Block>()
        )));
    }
    if !len.is_power_of_two() {
        return Err(ParquetError::General(format!(
            "Bloom filter bitset length {len} is not a power of two"
        )));
    }
    Ok(())
}

/// Decode a Bloom filter bitset from validated bytes.
///
/// This function constructs an [`Sbbf`] from the bitset bytes after the header
/// has been validated by [`decode_bloom_filter_header`].
///
/// # Arguments
/// * `info` - Validated header information
/// * `bitset` - The exact bitset bytes (must match expected length)
///
/// # Errors
/// Returns an error if:
/// - The bitset length doesn't match the expected length from `info`
/// - The bitset fails validation in [`Sbbf::try_from_bitset`]
pub(crate) fn decode_bloom_filter_bitset(
    info: &BloomFilterReadInfo,
    bitset: &[u8],
) -> Result<Sbbf, ParquetError> {
    let expected_len = info
        .bitset_range
        .end
        .checked_sub(info.bitset_range.start)
        .and_then(|len| usize::try_from(len).ok())
        .ok_or_else(|| {
            ParquetError::General(format!(
                "Invalid bloom filter bitset range: {:?}",
                info.bitset_range
            ))
        })?;

    if bitset.len() != expected_len {
        return Err(ParquetError::General(format!(
            "Bloom filter bitset length mismatch: expected {expected_len} bytes, got {}",
            bitset.len()
        )));
    }

    Sbbf::try_from_bitset(bitset)
}

/// Extract and validate Bloom filter location from column chunk metadata.
///
/// Returns `Ok(None)` if the column chunk has no Bloom filter (no offset in metadata).
/// Returns `Ok(Some((offset, total_length)))` with validated offset and optional total length.
///
/// # Validation
/// - Rejects negative offsets
/// - Rejects negative `bloom_filter_length`
/// - Validates `offset + length` doesn't overflow when length is present
///
/// # Errors
/// Returns an error if any validation fails.
pub(crate) fn bloom_filter_location(
    column: &ColumnChunkMetaData,
) -> Result<Option<(u64, Option<usize>)>, ParquetError> {
    // Check if bloom filter exists
    let Some(offset_i64) = column.bloom_filter_offset() else {
        return Ok(None);
    };

    // Validate and convert offset
    let offset = u64::try_from(offset_i64).map_err(|_| {
        ParquetError::General(format!(
            "Bloom filter offset is negative or invalid: {offset_i64}"
        ))
    })?;

    // Check if total length is provided
    let total_length = match column.bloom_filter_length() {
        Some(length_i32) => {
            // Validate and convert length
            let length = usize::try_from(length_i32).map_err(|_| {
                ParquetError::General(format!(
                    "Bloom filter length is negative or invalid: {length_i32}"
                ))
            })?;

            // Validate offset + length doesn't overflow
            let length_u64 = u64::try_from(length).map_err(|_| {
                ParquetError::General(format!("Bloom filter length too large: {length}"))
            })?;

            offset.checked_add(length_u64).ok_or_else(|| {
                ParquetError::General(format!(
                    "Bloom filter range overflow: offset {offset} + length {length}"
                ))
            })?;

            Some(length)
        }
        None => None,
    };

    Ok(Some((offset, total_length)))
}

/// Given a byte slice, try to read out a bloom filter header and return both the header and
/// length of the header.
#[inline]
fn read_bloom_filter_header_and_length_from_bytes(
    buffer: &[u8],
) -> Result<(BloomFilterHeader, u64), ParquetError> {
    let total_length = buffer.len();
    let mut prot = ThriftSliceInputProtocol::new(buffer);
    let header = BloomFilterHeader::read_thrift(&mut prot)
        .map_err(|e| ParquetError::General(format!("Could not read bloom filter header: {e}")))?;
    Ok((header, (total_length - prot.as_slice().len()) as u64))
}

/// The minimum number of bytes for a bloom filter bitset.
pub const BITSET_MIN_LENGTH: usize = 32;
/// The maximum number of bytes for a bloom filter bitset.
pub const BITSET_MAX_LENGTH: usize = 128 * 1024 * 1024;

#[inline]
fn optimal_num_of_bytes(num_bytes: usize) -> usize {
    let num_bytes = num_bytes.min(BITSET_MAX_LENGTH);
    let num_bytes = num_bytes.max(BITSET_MIN_LENGTH);
    num_bytes.next_power_of_two()
}

// see http://algo2.iti.kit.edu/documents/cacheefficientbloomfilters-jea.pdf
// given fpp = (1 - e^(-k * n / m)) ^ k
// we have m = - k * n / ln(1 - fpp ^ (1 / k))
// where k = number of hash functions, m = number of bits, n = number of distinct values
#[inline]
fn num_of_bits_from_ndv_fpp(ndv: u64, fpp: f64) -> usize {
    let num_bits = -8.0 * ndv as f64 / (1.0 - fpp.powf(1.0 / 8.0)).ln();
    num_bits as usize
}

impl Sbbf {
    /// Create a new [Sbbf] with given number of distinct values and false positive probability.
    /// Will return an error if `fpp` is greater than or equal to 1.0 or less than 0.0.
    pub fn new_with_ndv_fpp(ndv: u64, fpp: f64) -> Result<Self, ParquetError> {
        if !(0.0..1.0).contains(&fpp) {
            return Err(ParquetError::General(format!(
                "False positive probability must be between 0.0 and 1.0, got {fpp}"
            )));
        }
        let num_bits = num_of_bits_from_ndv_fpp(ndv, fpp);
        Ok(Self::new_with_num_of_bytes(num_bits / 8))
    }

    /// Create a new [Sbbf] with given number of bytes, the exact number of bytes will be adjusted
    /// to the next power of two bounded by [BITSET_MIN_LENGTH] and [BITSET_MAX_LENGTH].
    pub fn new_with_num_of_bytes(num_bytes: usize) -> Self {
        let num_bytes = optimal_num_of_bytes(num_bytes);
        assert_eq!(num_bytes % size_of::<Block>(), 0);
        let num_blocks = num_bytes / size_of::<Block>();
        let bitset = vec![Block::ZERO; num_blocks];
        Self(bitset)
    }

    /// Creates a new [Sbbf] from a raw byte slice.
    pub fn new(bitset: &[u8]) -> Self {
        let data = bitset
            .as_chunks::<32>()
            .0
            .iter()
            .map(|chunk| {
                let mut block = Block::ZERO;
                let (words, _remainder) = chunk.as_chunks::<4>();
                for (i, word) in words.iter().enumerate() {
                    block[i] = u32::from_le_bytes(*word);
                }
                block
            })
            .collect::<Vec<Block>>();
        Self(data)
    }

    /// Creates a new [Sbbf] from a bitset with validation suitable for file data.
    ///
    /// This method validates that the bitset meets requirements from the
    /// [Parquet Bloom Filter specification]:
    ///
    /// - Length is at least [`BITSET_MIN_LENGTH`] (32 bytes)
    /// - Length is at most [`BITSET_MAX_LENGTH`] (128 MiB)
    /// - Length is a multiple of 32 bytes (one [`Block`])
    /// - Length is a power of two (per the [specification §4.1])
    ///
    /// Unlike [`Self::new`], this method does not silently discard trailing bytes
    /// that don't form a complete block. All bytes must be consumed.
    ///
    /// # Errors
    ///
    /// Returns an error if the bitset does not meet the validation requirements.
    ///
    /// # Validation Examples
    ///
    /// Valid bitset sizes: 32, 64, 128, 256, 512, 1024, ... (powers of two)
    ///
    /// Invalid sizes:
    /// - 16 bytes (below minimum)
    /// - 33 bytes (not a multiple of 32)
    /// - 96 bytes (multiple of 32 but not a power of two)
    ///
    /// See unit tests `test_try_from_bitset_*` for working code examples.
    ///
    /// [Parquet Bloom Filter specification]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md
    /// [specification §4.1]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md#bloom-filter-data-page-format
    pub(crate) fn try_from_bitset(bitset: &[u8]) -> Result<Self, ParquetError> {
        validate_bitset_len(bitset.len())?;
        let (chunks, remainder) = bitset.as_chunks::<32>();
        debug_assert!(remainder.is_empty());

        let mut data = Vec::new();
        data.try_reserve_exact(chunks.len()).map_err(|err| {
            ParquetError::General(format!(
                "Could not allocate {} bloom filter blocks: {err}",
                chunks.len()
            ))
        })?;
        for chunk in chunks {
            let mut block = Block::ZERO;
            let (words, remainder) = chunk.as_chunks::<4>();
            debug_assert!(remainder.is_empty());
            for (i, word) in words.iter().enumerate() {
                block[i] = u32::from_le_bytes(*word);
            }
            data.push(block);
        }

        Ok(Self(data))
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
    #[cfg(not(target_endian = "little"))]
    pub fn write_bitset<W: Write>(&self, mut writer: W) -> Result<(), ParquetError> {
        for block in &self.0 {
            writer
                .write_all(block.to_le_bytes().as_slice())
                .map_err(|e| {
                    ParquetError::General(format!("Could not write bloom filter bit set: {e}"))
                })?;
        }
        Ok(())
    }

    /// Write the bitset in serialized form to the writer.
    #[cfg(target_endian = "little")]
    pub fn write_bitset<W: Write>(&self, mut writer: W) -> Result<(), ParquetError> {
        // Safety: Block is repr(transparent) and [u32; 8] can be reinterpreted as [u8; 32].
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr().cast::<u8>(),
                self.0.len() * size_of::<Block>(),
            )
        };
        writer.write_all(slice).map_err(|e| {
            ParquetError::General(format!("Could not write bloom filter bit set: {e}"))
        })?;
        Ok(())
    }

    /// Create and populate [`BloomFilterHeader`] from this bitset for writing to serialized form
    fn header(&self) -> BloomFilterHeader {
        BloomFilterHeader {
            // 8 i32 per block, 4 bytes per i32
            num_bytes: self.0.len() as i32 * 4 * 8,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        }
    }

    /// Read a new bloom filter from the given offset in the given reader.
    ///
    /// Returns `Ok(None)` if the column chunk has no Bloom filter.
    /// Returns `Err` if the Bloom filter exists but is malformed or truncated.
    ///
    /// This method uses validated decoding through [`decode_bloom_filter_header`]
    /// and [`decode_bloom_filter_bitset`] to ensure all Parquet specification
    /// requirements are met.
    pub fn read_from_column_chunk<R: ChunkReader>(
        column_metadata: &ColumnChunkMetaData,
        reader: &R,
    ) -> Result<Option<Self>, ParquetError> {
        // Extract and validate location from metadata
        let Some((offset, total_length)) = bloom_filter_location(column_metadata)? else {
            return Ok(None);
        };

        // Determine how much to read initially
        let initial_read_len = total_length.unwrap_or(SBBF_HEADER_SIZE_ESTIMATE);

        // Read header (and possibly full filter if length is known)
        let buffer = reader.get_bytes(offset, initial_read_len)?;

        // Decode and validate header
        let info = decode_bloom_filter_header(offset, buffer.clone(), total_length)?;

        // Get the bitset bytes
        let bitset = if total_length.is_some() {
            // We already have the full filter
            let relative_start = info.bitset_range.start.checked_sub(offset).ok_or_else(|| {
                ParquetError::General(format!(
                    "Bloom filter bitset starts before filter: {} < {offset}",
                    info.bitset_range.start
                ))
            })?;
            let bitset_start = usize::try_from(relative_start).map_err(|_| {
                ParquetError::General(format!("Bitset offset too large: {relative_start}"))
            })?;
            buffer.slice(bitset_start..)
        } else {
            // Need to read bitset separately
            let bitset_len = usize::try_from(info.bitset_range.end - info.bitset_range.start)
                .map_err(|_| {
                    ParquetError::General(format!(
                        "Bitset length too large: {}",
                        info.bitset_range.end - info.bitset_range.start
                    ))
                })?;
            reader.get_bytes(info.bitset_range.start, bitset_len)?
        };

        // Decode and validate bitset
        decode_bloom_filter_bitset(&info, &bitset).map(Some)
    }

    /// Map a 64-bit hash to a block index in `[0, num_blocks)`.
    ///
    /// Uses the "multiply-and-shift" trick (a fast alternative to modulo):
    ///
    /// ```text
    ///   upper32 = hash >> 32           // take the top 32 bits of the hash
    ///   index   = (upper32 * N) >> 32  // ∈ [0, N)  where N = num_blocks
    /// ```
    ///
    /// Why this matters for folding (Lemma 1): when N is a power of two and
    /// you halve it to N/2, the index also halves:
    ///
    /// ```text
    ///   index_N   = (upper32 * N)   >> 32
    ///   index_N/2 = (upper32 * N/2) >> 32 = index_N / 2  (integer division)
    /// ```
    ///
    /// So the block that held hash `h` in the big filter is at `index / 2` in
    /// the half-sized filter — exactly where `fold` ORs it.
    #[inline]
    fn hash_to_block_index(&self, hash: u64) -> usize {
        (((hash >> 32).saturating_mul(self.0.len() as u64)) >> 32) as usize
    }

    /// Insert an [AsBytes] value into the filter
    pub fn insert<T: AsBytes + ?Sized>(&mut self, value: &T) {
        self.insert_hash(hash_as_bytes(value));
    }

    /// Insert a hash into the filter
    fn insert_hash(&mut self, hash: u64) {
        let block_index = self.hash_to_block_index(hash);
        self.0[block_index].insert(hash as u32)
    }

    /// Check if an [AsBytes] value is probably present or definitely absent in the filter
    pub fn check<T: AsBytes + ?Sized>(&self, value: &T) -> bool {
        self.check_hash(hash_as_bytes(value))
    }

    /// Check if a hash is in the filter. May return
    /// true for values that was never inserted ("false positive")
    /// but will always return false if a hash has not been inserted.
    fn check_hash(&self, hash: u64) -> bool {
        let block_index = self.hash_to_block_index(hash);
        self.0[block_index].check(hash as u32)
    }

    /// Return the total in memory size of this bloom filter in bytes
    pub(crate) fn estimated_memory_size(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<Block>()
    }

    /// Returns the number of blocks in this bloom filter.
    pub fn num_blocks(&self) -> usize {
        self.0.len()
    }

    /// Estimate the false positive probability (FPP) of this filter at its current size.
    ///
    /// This is the same estimate [`Self::fold_to_target_fpp`] uses to choose how far to fold.
    ///
    /// This lets a caller inspect a filter before folding or writing it, for example to
    /// discard a filter that already exceeds its target FPP. Returns `1.0` for a filter
    /// with no blocks.
    pub fn estimated_fpp(&self) -> f64 {
        if self.0.is_empty() {
            return 1.0;
        }
        self.average_fill().powi(8)
    }

    /// Average fraction of bits set per block. The filter must have at least one block.
    fn average_fill(&self) -> f64 {
        let total_set_bits: u64 = self.0.iter().map(|b| u64::from(b.count_ones())).sum();
        total_set_bits as f64 / (self.0.len() as f64 * 256.0)
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
    pub fn fold_to_target_fpp(&mut self, target_fpp: f64) {
        let num_folds = self.num_folds_for_target_fpp(target_fpp);
        if num_folds > 0 {
            self.fold_n(num_folds);
        }
    }

    /// Determine how many folds can be applied without exceeding `target_fpp`.
    ///
    /// Computes the average per-block fill rate in a single pass (no allocation),
    /// then analytically estimates the FPP at each fold level.
    ///
    /// When two blocks with independent fill rate `f` are OR'd, the expected fill
    /// of the merged block is `1 - (1-f)^2`. After `k` folds (merging `2^k` blocks):
    ///
    /// ```text
    /// f_k = 1 - (1 - f)^(2^k)
    /// ```
    ///
    /// SBBF membership checks perform `k=8` bit checks within one 256-bit block,
    /// so the estimated FPP at fold level k is `f_k^8`.
    fn num_folds_for_target_fpp(&self, target_fpp: f64) -> u32 {
        let len = self.0.len();
        if len < 2 {
            return 0;
        }

        // Single pass: compute average per-block fill rate.
        let avg_fill = self.average_fill();

        // Empty filter: can fold all the way down.
        if avg_fill == 0.0 {
            return len.trailing_zeros();
        }

        // Find max folds where estimated FPP stays within target.
        // f_k = 1 - (1 - avg_fill)^(2^k), FPP_k = f_k^8
        assert!(
            len.is_power_of_two(),
            "Number of blocks must be a power of 2 for folding"
        );
        let max_folds = len.trailing_zeros(); // log2(len) since len is power of 2
        let one_minus_f = 1.0 - avg_fill;
        let mut num_folds = 0u32;
        let mut one_minus_fk = one_minus_f; // (1-f)^1 initially

        for _ in 0..max_folds {
            // After one more fold: (1-f)^(2^(k+1)) = ((1-f)^(2^k))^2
            one_minus_fk = one_minus_fk * one_minus_fk;
            let fk = 1.0 - one_minus_fk;
            let estimated_fpp = fk.powi(8);
            if estimated_fpp > target_fpp {
                break;
            }
            num_folds += 1;
        }

        num_folds
    }

    /// Fold the filter `num_folds` times in a single pass.
    ///
    /// Merges groups of `2^num_folds` adjacent blocks via bitwise OR, producing
    /// `len / 2^num_folds` output blocks. The original allocation is reused.
    ///
    /// # Panics
    ///
    /// Panics if `num_folds` is 0 or would reduce the filter below 1 block.
    fn fold_n(&mut self, num_folds: u32) {
        assert!(num_folds > 0, "num_folds must be at least 1");
        let len = self.0.len();
        let group_size = 1usize << num_folds;
        assert!(
            group_size <= len,
            "Cannot fold {num_folds} times: need at least {group_size} blocks, have {len}"
        );
        let new_len = len / group_size;
        for i in 0..new_len {
            let start = i * group_size;
            let mut merged = self.0[start];
            for j in 1..group_size {
                merged |= self.0[start + j];
            }
            self.0[i] = merged;
        }
        self.0.truncate(new_len);
    }

    /// Reads a Sbff from Thrift encoded bytes
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
        let info = decode_bloom_filter_header_from_slice(0, bytes, Some(bytes.len()))?;
        let start = usize::try_from(info.bitset_range.start).map_err(|_| {
            ParquetError::General(format!(
                "Bloom filter bitset offset is too large: {}",
                info.bitset_range.start
            ))
        })?;
        let end = usize::try_from(info.bitset_range.end).map_err(|_| {
            ParquetError::General(format!(
                "Bloom filter bitset end is too large: {}",
                info.bitset_range.end
            ))
        })?;
        let bitset = bytes
            .get(start..end)
            .ok_or_else(|| ParquetError::General("Bloom filter bitset is invalid".to_string()))?;

        decode_bloom_filter_bitset(&info, bitset)
    }
}

// per spec we use xxHash with seed=0
const SEED: u64 = 0;

#[inline]
fn hash_as_bytes<A: AsBytes + ?Sized>(value: &A) -> u64 {
    XxHash64::oneshot(SEED, value.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn encode_bloom_filter(num_bytes: i32, bitset_len: usize) -> Vec<u8> {
        let header = BloomFilterHeader {
            num_bytes,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut bytes = Vec::new();
        header
            .write_thrift(&mut ThriftCompactOutputProtocol::new(&mut bytes))
            .unwrap();
        bytes.resize(bytes.len() + bitset_len, 0);
        bytes
    }

    fn bloom_filter_column(offset: Option<i64>, length: Option<i32>) -> ColumnChunkMetaData {
        use crate::schema::types::{ColumnDescriptor, ColumnPath, Type};

        let column_descriptor = Arc::new(ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("test", crate::basic::Type::INT32)
                    .build()
                    .unwrap(),
            ),
            0,
            0,
            ColumnPath::new(vec!["test".to_string()]),
        ));
        ColumnChunkMetaData::builder(column_descriptor)
            .set_bloom_filter_offset(offset)
            .set_bloom_filter_length(length)
            .build()
            .unwrap()
    }

    #[test]
    fn test_hash_bytes() {
        assert_eq!(hash_as_bytes(""), 17241709254077376921);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_mask_set_quick_check() {
        for i in 0..1_000_000 {
            let result = Block::mask(i);
            assert!(result.0.iter().all(|&x| x.is_power_of_two()));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_block_insert_and_check() {
        for i in 0..1_000_000 {
            let mut block = Block::ZERO;
            block.insert(i);
            assert!(block.check(i));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_sbbf_insert_and_check() {
        let mut sbbf = Sbbf(vec![Block::ZERO; 1_000]);
        for i in 0..1_000_000 {
            sbbf.insert(&i);
            assert!(sbbf.check(&i));
        }
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
        ) = read_bloom_filter_header_and_length_from_bytes(buffer).unwrap();
        assert_eq!(read_length, 15);
        assert_eq!(algorithm, BloomFilterAlgorithm::BLOCK);
        assert_eq!(compression, BloomFilterCompression::UNCOMPRESSED);
        assert_eq!(hash, BloomFilterHash::XXHASH);
        assert_eq!(num_bytes, 32_i32);
        assert_eq!(20, SBBF_HEADER_SIZE_ESTIMATE);
    }

    #[test]
    fn test_optimal_num_of_bytes() {
        for (input, expected) in &[
            (0, 32),
            (9, 32),
            (31, 32),
            (32, 32),
            (33, 64),
            (99, 128),
            (1024, 1024),
            (999_000_000, 128 * 1024 * 1024),
        ] {
            assert_eq!(*expected, optimal_num_of_bytes(*input));
        }
    }

    #[test]
    fn test_num_of_bits_from_ndv_fpp() {
        for (fpp, ndv, num_bits) in &[
            (0.1, 10, 57),
            (0.01, 10, 96),
            (0.001, 10, 146),
            (0.1, 100, 577),
            (0.01, 100, 968),
            (0.001, 100, 1460),
            (0.1, 1000, 5772),
            (0.01, 1000, 9681),
            (0.001, 1000, 14607),
            (0.1, 10000, 57725),
            (0.01, 10000, 96815),
            (0.001, 10000, 146076),
            (0.1, 100000, 577254),
            (0.01, 100000, 968152),
            (0.001, 100000, 1460769),
            (0.1, 1000000, 5772541),
            (0.01, 1000000, 9681526),
            (0.001, 1000000, 14607697),
            (1e-50, 1_000_000_000_000, 14226231280773240832),
        ] {
            assert_eq!(*num_bits, num_of_bits_from_ndv_fpp(*ndv, *fpp) as u64);
        }
    }

    #[test]
    fn test_fold_n_halves_block_count() {
        let mut sbbf = Sbbf::new_with_num_of_bytes(1024); // 32 blocks
        assert_eq!(sbbf.num_blocks(), 32);
        sbbf.fold_n(1);
        assert_eq!(sbbf.num_blocks(), 16);
        sbbf.fold_n(1);
        assert_eq!(sbbf.num_blocks(), 8);
    }

    #[test]
    fn test_fold_preserves_inserted_values() {
        // Create a large filter, insert values, fold, verify no false negatives
        let mut sbbf = Sbbf::new_with_num_of_bytes(32 * 1024); // 32KB = 1024 blocks
        let values: Vec<String> = (0..1000).map(|i| format!("value_{i}")).collect();
        for v in &values {
            sbbf.insert(v.as_str());
        }

        // Fold several times
        let original_blocks = sbbf.num_blocks();
        sbbf.fold_to_target_fpp(0.05);
        assert!(
            sbbf.num_blocks() < original_blocks,
            "should have folded at least once"
        );

        // All inserted values must still be found (no false negatives)
        for v in &values {
            assert!(
                sbbf.check(v.as_str()),
                "Value '{v}' missing after folding (false negative!)"
            );
        }
    }

    #[test]
    fn test_fold_to_target_fpp_stops_before_exceeding_target() {
        let mut sbbf = Sbbf::new_with_num_of_bytes(64 * 1024); // 64KB
        // Insert enough values to set some bits
        for i in 0..5000 {
            sbbf.insert(&i);
        }

        let target_fpp = 0.01;
        sbbf.fold_to_target_fpp(target_fpp);

        // After folding, the estimated FPP should be at or below target
        // (the current state should not exceed target — we stopped before that would happen)
        let total_bits = (sbbf.num_blocks() * 256) as f64;
        let set_bits: u64 = sbbf
            .0
            .iter()
            .flat_map(|b| b.0.iter())
            .map(|w| w.count_ones() as u64)
            .sum();
        let fill = set_bits as f64 / total_bits;
        let current_fpp = fill.powi(8);
        assert!(
            current_fpp <= target_fpp,
            "FPP {current_fpp} exceeds target {target_fpp}"
        );
    }

    #[test]
    fn test_fold_empty_filter_folds_to_minimum() {
        // An empty filter has fill=0, so estimated FPP is always 0 — should fold all the way down
        let mut sbbf = Sbbf::new_with_num_of_bytes(1024); // 32 blocks
        sbbf.fold_to_target_fpp(0.01);
        assert_eq!(sbbf.num_blocks(), 1);
    }

    #[test]
    fn test_estimated_fpp_matches_serialized_bitset() {
        for num_bytes in [BITSET_MIN_LENGTH, 1024, 64 * 1024] {
            for ndv in [0u64, 1, 10, 100, 1_000, 10_000, 100_000] {
                let mut sbbf = Sbbf::new_with_num_of_bytes(num_bytes);
                for i in 0..ndv {
                    sbbf.insert(&i);
                }

                let mut bitset = Vec::new();
                sbbf.write_bitset(&mut bitset).unwrap();
                let set_bits: u64 = bitset.iter().map(|b| u64::from(b.count_ones())).sum();
                let expected = (set_bits as f64 / (bitset.len() as f64 * 8.0)).powi(8);

                assert_eq!(
                    sbbf.estimated_fpp().to_bits(),
                    expected.to_bits(),
                    "{num_bytes} bytes, {ndv} values"
                );
            }
        }
    }

    #[test]
    fn test_estimated_fpp_bounds() {
        assert_eq!(Sbbf::new_with_num_of_bytes(1024).estimated_fpp(), 0.0);
        assert_eq!(Sbbf::new(&[0xFF; 1024]).estimated_fpp(), 1.0);
        assert_eq!(Sbbf::new(&[]).estimated_fpp(), 1.0);
    }

    #[test]
    fn test_estimated_fpp_increases_when_folded() {
        let mut sbbf = Sbbf::new_with_num_of_bytes(64 * 1024);
        for i in 0..1_000 {
            sbbf.insert(&i);
        }
        let before = sbbf.estimated_fpp();
        sbbf.fold_n(3);
        assert!(
            sbbf.estimated_fpp() > before,
            "folding must not lower the estimate: {before} -> {}",
            sbbf.estimated_fpp()
        );
    }

    #[test]
    #[should_panic(expected = "Cannot fold 1 times: need at least 2 blocks, have 1")]
    fn test_fold_n_panics_at_minimum_size() {
        let mut sbbf = Sbbf::new_with_num_of_bytes(32); // 1 block (minimum)
        sbbf.fold_n(1);
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

    /// Prove that folding an SBBF by one level produces the exact same bits
    /// as building a fresh filter at the smaller size from scratch.
    ///
    /// # What is folding?
    ///
    /// ```text
    ///   Original (N = 8 blocks):
    ///   ┌───┬───┬───┬───┬───┬───┬───┬───┐
    ///   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
    ///   └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
    ///     │   │   │   │   │   │   │   │
    ///     └─OR┘   └─OR┘   └─OR┘   └─OR┘    pair-wise OR
    ///       │       │       │       │
    ///   ┌───┴──┬────┴──┬────┴──┬────┴──┐
    ///   │ 0|1  │ 2|3   │ 4|5   │ 6|7   │   Folded (N/2 = 4 blocks)
    ///   └──────┴───────┴───────┴───────┘
    /// ```
    ///
    /// # Why folded == fresh (the two lemmas)
    ///
    /// An SBBF insertion does two things with a 64-bit hash `h`:
    ///
    ///   1. **Pick a block** — uses the upper 32 bits via `hash_to_block_index`
    ///   2. **Set 8 bits in that block** — uses the lower 32 bits via `Block::mask`
    ///
    /// **Lemma 1 (block index halves):** `hash_to_block_index` uses
    /// `(upper32 * N) >> 32`. When N halves, the index halves too:
    /// `index_in(N/2) == index_in(N) / 2`. So the hash lands in the same
    /// destination block whether you fold or build fresh.
    ///
    /// **Lemma 2 (mask is size-independent):** `Block::mask(h as u32)` depends
    /// only on the lower 32 bits and the fixed SALT constants — the filter
    /// size N is not involved. So the same 8 bits get set regardless.
    ///
    /// Combined: every hash sets the *same bits* in the *same destination
    /// block* whether you fold or build fresh → filters are bit-identical.
    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_sbbf_folded_equals_fresh() {
        let values = (0..5000).map(|i| format!("elem_{i}")).collect::<Vec<_>>();
        let hashes = values
            .iter()
            .map(|v| hash_as_bytes(v.as_str()))
            .collect::<Vec<_>>();

        for num_blocks in [64, 256, 1024] {
            let half = num_blocks / 2;

            // Build a filter with N blocks and insert all values.
            let mut original = Sbbf::new_with_num_of_bytes(num_blocks * 32);
            assert_eq!(original.num_blocks(), num_blocks);
            for &h in &hashes {
                original.insert_hash(h);
            }

            // --- Per-hash verification of the two lemmas ---
            for &h in &hashes {
                // mask(h as u32) gives the 8-bit pattern that this hash sets
                // inside whichever block it lands in. It uses only the lower
                // 32 bits of h, so it's the same regardless of filter size.
                let mask = Block::mask(h as u32);

                // Lemma 1 check: the block index in the original N-block
                // filter, divided by 2, should equal the block index in a
                // fresh N/2-block filter.
                let orig_idx = original.hash_to_block_index(h);
                assert!(orig_idx < num_blocks);

                let fresh_idx = {
                    let tmp = Sbbf(vec![Block::ZERO; half]);
                    tmp.hash_to_block_index(h)
                };
                let folded_idx = orig_idx / 2;
                assert_eq!(
                    fresh_idx, folded_idx,
                    "Lemma 1 failed: fresh index {fresh_idx} != folded index {folded_idx}"
                );

                // Lemma 2 check: every bit that mask wants to set is actually
                // present in the original block.
                //
                // mask.0[w] has exactly ONE bit set (see Block::mask: `1 << y`).
                // The block at orig_idx has many bits set from many inserts, so
                // we can't test equality — we test that the specific mask bit is
                // *present*:
                //
                //   block_word & mask_word != 0
                //     ⟺  "the one bit in the mask is set in the block"
                //
                // (Since mask_word has exactly 1 bit, `& mask != 0` is the same
                //  as `& mask == mask` — but `!= 0` reads more naturally.)
                for w in 0..8 {
                    assert_ne!(
                        original.0[orig_idx].0[w] & mask.0[w],
                        0,
                        "Lemma 2 failed: mask bit not set in word {w} of block {orig_idx}"
                    );
                }
            }

            // --- Final bit-identical comparison ---
            // Fold the original N-block filter down to N/2 blocks.
            let mut folded = original.clone();
            folded.fold_n(1);
            assert_eq!(folded.num_blocks(), half);

            // Build a fresh N/2-block filter with the same values.
            let mut fresh = Sbbf::new_with_num_of_bytes(half * 32);
            for &h in &hashes {
                fresh.insert_hash(h);
            }

            // By lemmas 1 + 2, every block should be bit-identical.
            for j in 0..half {
                assert_eq!(
                    folded.0[j].0, fresh.0[j].0,
                    "Block {j} differs after fold (N={num_blocks} → {half})"
                );
            }
        }
    }

    /// Inductive multi-step folding: folding k times from N blocks produces
    /// a filter bit-identical to a fresh N/2^k-block filter.
    ///
    /// `test_sbbf_folded_equals_fresh` proves the base case (one fold).
    /// This test applies folds *repeatedly*, checking after each step:
    ///
    /// ```text
    ///   512 ─fold→ 256 ─fold→ 128 ─…→ 1  (9 folds total)
    /// ```
    ///
    /// At each intermediate size we build a fresh filter and assert
    /// bit-equality, confirming the lemma composes across folds.
    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_multi_step_fold() {
        let values = (0..3000).map(|i| format!("x_{i}")).collect::<Vec<_>>();

        // Start with a 512-block filter.
        let mut filter = Sbbf::new_with_num_of_bytes(512 * 32);
        for v in &values {
            filter.insert(v.as_str());
        }

        // Fold one level at a time, comparing against a fresh filter each step.
        for expected_blocks in [256, 128, 64, 32, 16, 8, 4, 2, 1] {
            filter.fold_n(1);
            assert_eq!(filter.num_blocks(), expected_blocks);

            let mut fresh = Sbbf::new_with_num_of_bytes(expected_blocks * 32);
            for v in &values {
                fresh.insert(v.as_str());
            }
            for (fb, rb) in filter.0.iter().zip(fresh.0.iter()) {
                assert_eq!(fb.0, rb.0);
            }
        }
    }

    /// test that the fpp estimator's overestimation doesn't cause fold_to_target_fpp
    /// to produce significantly oversized filters
    ///
    /// compare the final size after folding against the theoretical optimal size
    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_fold_size_vs_optimal_fixed_size() {
        for (ndv, target_fpp) in [
            (1000, 0.05),
            (1000, 0.01),
            (5000, 0.05),
            (5000, 0.01),
            (10000, 0.05),
        ] {
            let values = (0..ndv).map(|i| format!("d_{i}")).collect::<Vec<_>>();

            let mut folded = Sbbf::new_with_num_of_bytes(128 * 1024); // 128KB
            for v in &values {
                folded.insert(v.as_str());
            }
            folded.fold_to_target_fpp(target_fpp);

            let folded_bytes = folded.num_blocks() * 32;

            let optimal = Sbbf::new_with_ndv_fpp(ndv as u64, target_fpp).unwrap();
            let optimal_bytes = optimal.num_blocks() * 32;

            let ratio = folded_bytes as f64 / optimal_bytes as f64;

            assert_eq!(ratio, 1.0);
        }
    }

    /// verify that a folded sbbf has the same empirical fpp as a fresh filter of the same size
    /// this bridges the bit-identity proof above with the FPP guarantee from the folding paper
    ///     since the bits are identical, the false-positive rate must be too
    ///
    /// we measure fpp empirically by probing with values that were never inserted
    /// and counting how many are incorrectly marked as present
    #[test]
    #[cfg_attr(miri, ignore)] // Takes too long
    fn test_folded_fpp_matches_fresh_fpp() {
        let ndv = 2000;
        let num_probes = 50_000;
        let inserted = (0..ndv)
            .map(|i| format!("ins_{i}"))
            .collect::<Vec<String>>();

        // probe values that were NOT inserted (different prefix guarantees no overlap)
        let probes = (0..num_probes)
            .map(|i| format!("probe_{i}"))
            .collect::<Vec<String>>();

        // build a large filter and fold it down several times
        let mut folded = Sbbf::new_with_num_of_bytes(512 * 32); // 512 blocks
        for v in &inserted {
            folded.insert(v.as_str());
        }

        // check FPP at each fold level
        for expected_blocks in [256, 128, 64, 32, 16, 8, 4, 2, 1] {
            folded.fold_n(1);
            assert_eq!(folded.num_blocks(), expected_blocks);

            // build a fresh filter of the same size with the same values
            let mut fresh = Sbbf::new_with_num_of_bytes(expected_blocks * 32);
            for v in &inserted {
                fresh.insert(v.as_str());
            }

            // measure empirical FPP on both
            let mut folded_fp = 0u64;
            let mut fresh_fp = 0u64;
            for p in &probes {
                if folded.check(p.as_str()) {
                    folded_fp += 1;
                }
                if fresh.check(p.as_str()) {
                    fresh_fp += 1;
                }
            }

            // bit-identity means these must be exactly equal
            assert_eq!(folded_fp, fresh_fp);
        }
    }

    // ========================================================================
    // Phase 1 Validation Tests
    // ========================================================================

    #[test]
    fn test_try_from_bitset_valid_minimum() {
        // Valid minimum size (32 bytes)
        let bitset = vec![0u8; 32];
        let result = Sbbf::try_from_bitset(&bitset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_from_bitset_valid_power_of_two() {
        // Valid power of two sizes
        for size in [32, 64, 128, 256, 512, 1024] {
            let bitset = vec![0u8; size];
            let result = Sbbf::try_from_bitset(&bitset);
            assert!(result.is_ok(), "Size {size} should be valid");
        }
    }

    #[test]
    fn test_try_from_bitset_too_small() {
        // Below minimum (16 bytes < 32 minimum)
        let bitset = vec![0u8; 16];
        let result = Sbbf::try_from_bitset(&bitset);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_validate_bitset_too_large() {
        let error = validate_bitset_len(BITSET_MAX_LENGTH + 1).unwrap_err();
        assert!(error.to_string().contains("too large"));
    }

    #[test]
    fn test_try_from_bitset_not_multiple_of_32() {
        // Not a multiple of block size
        let bitset = vec![0u8; 33];
        let result = Sbbf::try_from_bitset(&bitset);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple"));
    }

    #[test]
    fn test_try_from_bitset_not_power_of_two() {
        // Multiple of 32 but not power of two (96 = 32 * 3)
        let bitset = vec![0u8; 96];
        let result = Sbbf::try_from_bitset(&bitset);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not a power of two")
        );
    }

    #[test]
    fn test_decode_bloom_filter_header_valid_known_length() {
        // Create a valid header
        let mut header_bytes = Vec::new();
        let header = BloomFilterHeader {
            num_bytes: 32,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut protocol = ThriftCompactOutputProtocol::new(&mut header_bytes);
        header.write_thrift(&mut protocol).unwrap();

        let result = decode_bloom_filter_header(
            100,
            Bytes::from(header_bytes.clone()),
            Some(header_bytes.len() + 32),
        );
        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.filter_offset, 100);
        assert_eq!(info.bitset_range.end - info.bitset_range.start, 32);
    }

    #[test]
    fn test_decode_bloom_filter_header_length_mismatch() {
        // Create a valid header claiming 32 bytes
        let mut header_bytes = Vec::new();
        let header = BloomFilterHeader {
            num_bytes: 32,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut protocol = ThriftCompactOutputProtocol::new(&mut header_bytes);
        header.write_thrift(&mut protocol).unwrap();

        // But advertise a different total length
        let result = decode_bloom_filter_header(
            100,
            Bytes::from(header_bytes.clone()),
            Some(header_bytes.len() + 64), // Wrong: should be +32
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length mismatch"));
    }

    #[test]
    fn test_decode_bloom_filter_header_negative_num_bytes() {
        // Create a header with negative num_bytes
        let mut header_bytes = Vec::new();
        let header = BloomFilterHeader {
            num_bytes: -1,
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut protocol = ThriftCompactOutputProtocol::new(&mut header_bytes);
        header.write_thrift(&mut protocol).unwrap();

        let result = decode_bloom_filter_header(100, Bytes::from(header_bytes), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("negative"));
    }

    #[test]
    fn test_decode_bloom_filter_header_bitset_too_small() {
        // Create a header with bitset smaller than minimum
        let mut header_bytes = Vec::new();
        let header = BloomFilterHeader {
            num_bytes: 16, // Below BITSET_MIN_LENGTH
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut protocol = ThriftCompactOutputProtocol::new(&mut header_bytes);
        header.write_thrift(&mut protocol).unwrap();

        let result = decode_bloom_filter_header(100, Bytes::from(header_bytes), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_decode_bloom_filter_header_bitset_not_power_of_two() {
        // Create a header with bitset that's not power of two
        let mut header_bytes = Vec::new();
        let header = BloomFilterHeader {
            num_bytes: 96, // Multiple of 32 but not power of two
            algorithm: BloomFilterAlgorithm::BLOCK,
            hash: BloomFilterHash::XXHASH,
            compression: BloomFilterCompression::UNCOMPRESSED,
        };
        let mut protocol = ThriftCompactOutputProtocol::new(&mut header_bytes);
        header.write_thrift(&mut protocol).unwrap();

        let result = decode_bloom_filter_header(100, Bytes::from(header_bytes), None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not a power of two")
        );
    }

    #[test]
    fn test_decode_bloom_filter_bitset_length_mismatch() {
        // Create valid info expecting 32 bytes
        let info = BloomFilterReadInfo {
            filter_offset: 100,
            header_len: 10,
            bitset_range: 110..142, // 32 bytes
            total_len: 42,
        };

        // But provide wrong number of bytes
        let bitset = vec![0u8; 64]; // Wrong: should be 32
        let result = decode_bloom_filter_bitset(&info, &bitset);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length mismatch"));
    }

    #[test]
    fn test_bloom_filter_location_missing_offset() {
        use crate::file::metadata::ColumnChunkMetaData;
        use crate::schema::types::Type;

        // Create column metadata without bloom filter
        let col_desc = Arc::new(crate::schema::types::ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("test", crate::basic::Type::INT32)
                    .build()
                    .unwrap(),
            ),
            0,
            0,
            crate::schema::types::ColumnPath::new(vec!["test".to_string()]),
        ));
        let column = ColumnChunkMetaData::builder(col_desc).build().unwrap();

        let result = bloom_filter_location(&column);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_bloom_filter_location_negative_offset() {
        use crate::file::metadata::ColumnChunkMetaData;
        use crate::schema::types::Type;

        let col_desc = Arc::new(crate::schema::types::ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("test", crate::basic::Type::INT32)
                    .build()
                    .unwrap(),
            ),
            0,
            0,
            crate::schema::types::ColumnPath::new(vec!["test".to_string()]),
        ));
        let column = ColumnChunkMetaData::builder(col_desc)
            .set_bloom_filter_offset(Some(-1))
            .build()
            .unwrap();

        let result = bloom_filter_location(&column);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("negative"));
    }

    #[test]
    fn test_bloom_filter_location_negative_length() {
        use crate::file::metadata::ColumnChunkMetaData;
        use crate::schema::types::Type;

        let col_desc = Arc::new(crate::schema::types::ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("test", crate::basic::Type::INT32)
                    .build()
                    .unwrap(),
            ),
            0,
            0,
            crate::schema::types::ColumnPath::new(vec!["test".to_string()]),
        ));
        let column = ColumnChunkMetaData::builder(col_desc)
            .set_bloom_filter_offset(Some(100))
            .set_bloom_filter_length(Some(-1))
            .build()
            .unwrap();

        let result = bloom_filter_location(&column);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("negative"));
    }

    #[test]
    fn test_bloom_filter_location_valid_with_large_offset() {
        use crate::file::metadata::ColumnChunkMetaData;
        use crate::schema::types::Type;

        let col_desc = Arc::new(crate::schema::types::ColumnDescriptor::new(
            Arc::new(
                Type::primitive_type_builder("test", crate::basic::Type::INT32)
                    .build()
                    .unwrap(),
            ),
            0,
            0,
            crate::schema::types::ColumnPath::new(vec!["test".to_string()]),
        ));
        // Note: In practice, overflow is very unlikely because:
        // - Offset comes from i64::MAX at most
        // - Length comes from i32::MAX at most
        // - u64::MAX is much larger than i64::MAX + i32::MAX
        // This test verifies the check exists and accepts valid large values
        let column = ColumnChunkMetaData::builder(col_desc)
            .set_bloom_filter_offset(Some(i64::MAX - 1000))
            .set_bloom_filter_length(Some(1000))
            .build()
            .unwrap();

        let result = bloom_filter_location(&column);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_decode_bloom_filter_header_offset_overflow() {
        let bytes = encode_bloom_filter(32, 0);
        let error = decode_bloom_filter_header(u64::MAX, Bytes::from(bytes), None).unwrap_err();
        assert!(error.to_string().contains("offset overflow"));
    }

    #[test]
    fn test_decode_bloom_filter_header_truncated() {
        let bytes = encode_bloom_filter(32, 0);
        let truncated = Bytes::copy_from_slice(&bytes[..bytes.len() - 1]);
        let error = decode_bloom_filter_header(123, truncated, None).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("offset 123"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn test_decode_bloom_filter_bitset_rejects_short_and_long_input() {
        let bytes = encode_bloom_filter(32, 0);
        let info = decode_bloom_filter_header(100, Bytes::from(bytes), None).unwrap();

        for bitset_len in [31, 33] {
            let error = decode_bloom_filter_bitset(&info, &vec![0; bitset_len]).unwrap_err();
            assert!(error.to_string().contains("length mismatch"));
        }
    }

    #[test]
    fn test_decode_bloom_filter_bitset_rejects_invalid_range() {
        let valid_range = 110..142;
        let info = BloomFilterReadInfo {
            filter_offset: 100,
            header_len: 10,
            bitset_range: valid_range.end..valid_range.start,
            total_len: 42,
        };
        let error = decode_bloom_filter_bitset(&info, &[]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Invalid bloom filter bitset range")
        );
    }

    #[test]
    fn test_from_bytes_uses_validated_decoder() {
        for num_bytes in [16, 33, 96] {
            let bytes = encode_bloom_filter(num_bytes, num_bytes as usize);
            assert!(
                Sbbf::from_bytes(&bytes).is_err(),
                "accepted invalid bitset length {num_bytes}"
            );
        }

        for actual_len in [31, 33] {
            let bytes = encode_bloom_filter(32, actual_len);
            assert!(
                Sbbf::from_bytes(&bytes).is_err(),
                "accepted {actual_len} bytes for a 32-byte bitset"
            );
        }
    }

    #[test]
    fn test_read_from_column_chunk_known_and_unknown_length() {
        let bytes = Bytes::from(encode_bloom_filter(32, 32));
        let total_len = i32::try_from(bytes.len()).unwrap();

        for length in [Some(total_len), None] {
            let column = bloom_filter_column(Some(0), length);
            let filter = Sbbf::read_from_column_chunk(&column, &bytes)
                .unwrap()
                .unwrap();
            assert_eq!(filter, Sbbf::try_from_bitset(&[0; 32]).unwrap());
        }
    }

    #[test]
    fn test_read_from_column_chunk_rejects_truncated_bitset() {
        let bytes = Bytes::from(encode_bloom_filter(32, 31));

        let known_length = bloom_filter_column(Some(0), Some(i32::try_from(bytes.len()).unwrap()));
        assert!(Sbbf::read_from_column_chunk(&known_length, &bytes).is_err());

        let unknown_length = bloom_filter_column(Some(0), None);
        assert!(Sbbf::read_from_column_chunk(&unknown_length, &bytes).is_err());
    }

    #[test]
    fn test_read_from_column_chunk_without_offset() {
        let column = bloom_filter_column(None, None);
        assert!(
            Sbbf::read_from_column_chunk(&column, &Bytes::new())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_block_and_sbbf_equality() {
        // Test Block equality
        let block1 = Block([1, 2, 3, 4, 5, 6, 7, 8]);
        let block2 = Block([1, 2, 3, 4, 5, 6, 7, 8]);
        let block3 = Block([1, 2, 3, 4, 5, 6, 7, 9]);
        assert_eq!(block1, block2);
        assert_ne!(block1, block3);

        // Test Sbbf equality
        let sbbf1 = Sbbf(vec![block1, block2]);
        let sbbf2 = Sbbf(vec![block1, block2]);
        let sbbf3 = Sbbf(vec![block1, block3]);
        assert_eq!(sbbf1, sbbf2);
        assert_ne!(sbbf1, sbbf3);
    }
}
