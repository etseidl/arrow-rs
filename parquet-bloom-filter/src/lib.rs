// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Standalone split block Bloom filters compatible with the Parquet specification.
//!
//! This crate provides byte and prehashed membership operations, raw little-endian
//! bitset serialization, sizing, FPP estimation, and folding without Parquet or Arrow
//! dependencies. Parquet headers and file I/O are provided by the `parquet` crate.
//! [`Sbbf::new`] and [`Sbbf::write_bitset`] work with the raw bitset, not a complete
//! serialized Parquet Bloom filter.
//!
//! For [Parquet compatibility][spec], [`hash_bytes`] must receive the plain-encoded
//! physical value bytes, without length prefixes for byte arrays. Logical and
//! Arrow value conversions belong to the caller. Precomputed hashes may use any
//! well-mixed 64-bit hash function, provided insertion and lookup use the same
//! encoding and hash function; only specification-compliant hashes yield
//! Parquet-compatible filters.
//!
//! [spec]: https://github.com/apache/parquet-format/blob/master/BloomFilter.md
//!
//! ```
//! use parquet_bloom_filter::{Sbbf, hash_bytes};
//! let mut filter = Sbbf::new_with_ndv_fpp(1_000, 0.01)?;
//! filter.insert_bytes(b"present");
//! assert!(filter.check_hash(hash_bytes(b"present")));
//! let mut bytes = Vec::new();
//! filter.write_bitset(&mut bytes)?;
//! assert!(Sbbf::new(&bytes).check_bytes(b"present"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::io::{self, Write};
use twox_hash::XxHash64;

mod error;
pub use error::SbbfError;

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
#[derive(Debug, Copy, Clone)]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
#[derive(Debug, Clone)]
pub struct Sbbf(Vec<Block>);

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
    /// Returns an error if `fpp` is outside `[0.0, 1.0)` or NaN.
    pub fn new_with_ndv_fpp(ndv: u64, fpp: f64) -> Result<Self, SbbfError> {
        if !(0.0..1.0).contains(&fpp) {
            return Err(SbbfError::InvalidFpp(fpp));
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
    ///
    /// Only complete 32-byte blocks are imported; trailing bytes are ignored.
    /// Fewer than 32 bytes produce a zero-block filter, on which insertion and
    /// membership checks panic. Use a power-of-two block count if folding is needed.
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

    /// Write the bitset in serialized form to the writer.
    #[cfg(not(target_endian = "little"))]
    pub fn write_bitset<W: Write>(&self, mut writer: W) -> io::Result<()> {
        for block in &self.0 {
            writer.write_all(block.to_le_bytes().as_slice())?;
        }
        Ok(())
    }

    /// Write the bitset in serialized form to the writer.
    #[cfg(target_endian = "little")]
    pub fn write_bitset<W: Write>(&self, mut writer: W) -> io::Result<()> {
        // Safety: Block is repr(transparent) and [u32; 8] can be reinterpreted as [u8; 32].
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr().cast::<u8>(),
                self.0.len() * size_of::<Block>(),
            )
        };
        writer.write_all(slice)?;
        Ok(())
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

    /// Insert a byte slice into the filter
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn insert_bytes(&mut self, value: &[u8]) {
        self.insert_hash(hash_bytes(value));
    }

    /// Insert a hash into the filter
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        let block_index = self.hash_to_block_index(hash);
        self.0[block_index].insert(hash as u32)
    }

    /// Check if a byte slice is probably present or definitely absent in the filter
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn check_bytes(&self, value: &[u8]) -> bool {
        self.check_hash(hash_bytes(value))
    }

    /// Check whether a hash is probably present or definitely absent.
    ///
    /// Returns `true` for inserted hashes and possibly for hashes never inserted
    /// (false positives). Returns `false` only for hashes that were not inserted.
    ///
    /// # Panics
    ///
    /// Panics if the filter has no blocks.
    #[inline]
    pub fn check_hash(&self, hash: u64) -> bool {
        let block_index = self.hash_to_block_index(hash);
        self.0[block_index].check(hash as u32)
    }

    /// Return the heap allocation size (excluding the filter struct) of this bloom filter in bytes
    pub fn estimated_memory_size(&self) -> usize {
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
    ///
    /// # Preconditions
    ///
    /// Use a power-of-two block count for folding. The target FPP is not validated.
    /// Zero-block filters are left unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a populated filter with at least two blocks has a non-power-of-two
    /// block count.
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
        // `truncate` keeps the capacity; release it so filters retained until the file is closed
        // only hold their folded size.
        self.0.shrink_to_fit();
    }
}

/// Hash a byte slice using xxHash64 with seed 0, as required by Parquet.
#[inline]
pub fn hash_bytes(value: &[u8]) -> u64 {
    XxHash64::oneshot(0, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_with_ndv_fpp_validation() {
        for fpp in [-1.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let error = Sbbf::new_with_ndv_fpp(100, fpp).unwrap_err();
            assert!(matches!(error, SbbfError::InvalidFpp(_)));
            assert_eq!(
                error.to_string(),
                format!("False positive probability must be between 0.0 and 1.0, got {fpp}")
            );
        }
        assert_eq!(Sbbf::new_with_ndv_fpp(100, 0.0).unwrap().num_blocks(), 1);
        assert!(Sbbf::new_with_ndv_fpp(100, 0.01).is_ok());
    }

    #[test]
    fn test_raw_bitset_little_endian_round_trip() {
        let bytes: Vec<u8> = (0..64).collect();
        let filter = Sbbf::new(&bytes);
        assert_eq!(filter.0[0].0[0], u32::from_le_bytes([0, 1, 2, 3]));
        assert_eq!(filter.0[1].0[7], u32::from_le_bytes([60, 61, 62, 63]));
        let mut output = Vec::new();
        filter.write_bitset(&mut output).unwrap();
        assert_eq!(output, bytes);
    }

    #[test]
    fn test_new_ignores_incomplete_blocks() {
        for length in [0, 1, 31, 32, 33, 63, 64, 65] {
            let filter = Sbbf::new(&vec![0xAB; length]);
            assert_eq!(filter.num_blocks(), length / 32);
            let mut output = Vec::new();
            filter.write_bitset(&mut output).unwrap();
            assert_eq!(output, vec![0xAB; length / 32 * 32]);
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_insert_hash_panics_without_blocks() {
        Sbbf::new(&[]).insert_hash(0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_check_hash_panics_without_blocks() {
        Sbbf::new(&[]).check_hash(0);
    }

    #[test]
    fn test_byte_and_prehashed_operations_match() {
        let mut bytes_filter = Sbbf::new_with_num_of_bytes(1024);
        let mut hash_filter = bytes_filter.clone();
        for value in [b"".as_slice(), b"present", b"other"] {
            bytes_filter.insert_bytes(value);
            hash_filter.insert_hash(hash_bytes(value));
            assert!(bytes_filter.check_hash(hash_bytes(value)));
            assert!(hash_filter.check_bytes(value));
        }
        for (bytes, hashes) in bytes_filter.0.iter().zip(&hash_filter.0) {
            assert_eq!(bytes.0, hashes.0);
        }
    }

    #[test]
    fn test_hash_bytes() {
        assert_eq!(hash_bytes(b""), 17241709254077376921);
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
        for i in 0i32..1_000_000 {
            sbbf.insert_bytes(&i.to_ne_bytes());
            assert!(sbbf.check_bytes(&i.to_ne_bytes()));
        }
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
        assert_eq!(sbbf.0.capacity(), 16);
        sbbf.fold_n(1);
        assert_eq!(sbbf.num_blocks(), 8);
        assert_eq!(sbbf.0.capacity(), 8);
    }

    #[test]
    fn test_fold_preserves_inserted_values() {
        // Create a large filter, insert values, fold, verify no false negatives
        let mut sbbf = Sbbf::new_with_num_of_bytes(32 * 1024); // 32KB = 1024 blocks
        let values: Vec<String> = (0..1000).map(|i| format!("value_{i}")).collect();
        for v in &values {
            sbbf.insert_bytes(v.as_bytes());
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
                sbbf.check_bytes(v.as_bytes()),
                "Value '{v}' missing after folding (false negative!)"
            );
        }
    }

    #[test]
    fn test_fold_to_target_fpp_stops_before_exceeding_target() {
        let mut sbbf = Sbbf::new_with_num_of_bytes(64 * 1024); // 64KB
        // Insert enough values to set some bits
        for i in 0i32..5000 {
            sbbf.insert_bytes(&i.to_ne_bytes());
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
                    sbbf.insert_bytes(&i.to_ne_bytes());
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
        for i in 0i32..1_000 {
            sbbf.insert_bytes(&i.to_ne_bytes());
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
            .map(|v| hash_bytes(v.as_bytes()))
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
            filter.insert_bytes(v.as_bytes());
        }

        // Fold one level at a time, comparing against a fresh filter each step.
        for expected_blocks in [256, 128, 64, 32, 16, 8, 4, 2, 1] {
            filter.fold_n(1);
            assert_eq!(filter.num_blocks(), expected_blocks);

            let mut fresh = Sbbf::new_with_num_of_bytes(expected_blocks * 32);
            for v in &values {
                fresh.insert_bytes(v.as_bytes());
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
                folded.insert_bytes(v.as_bytes());
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
            folded.insert_bytes(v.as_bytes());
        }

        // check FPP at each fold level
        for expected_blocks in [256, 128, 64, 32, 16, 8, 4, 2, 1] {
            folded.fold_n(1);
            assert_eq!(folded.num_blocks(), expected_blocks);

            // build a fresh filter of the same size with the same values
            let mut fresh = Sbbf::new_with_num_of_bytes(expected_blocks * 32);
            for v in &inserted {
                fresh.insert_bytes(v.as_bytes());
            }

            // measure empirical FPP on both
            let mut folded_fp = 0u64;
            let mut fresh_fp = 0u64;
            for p in &probes {
                if folded.check_bytes(p.as_bytes()) {
                    folded_fp += 1;
                }
                if fresh.check_bytes(p.as_bytes()) {
                    fresh_fp += 1;
                }
            }

            // bit-identity means these must be exactly equal
            assert_eq!(folded_fp, fresh_fp);
        }
    }
}
