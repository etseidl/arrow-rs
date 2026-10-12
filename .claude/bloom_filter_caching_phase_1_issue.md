# Parquet: Centralize and harden Bloom filter decoding

## Parent

Part of the EPIC to add cacheable Bloom filter access to Parquet metadata.

## Summary

Create one validated Bloom filter decoder shared by all current synchronous and asynchronous read
paths.

This issue deliberately does not add caching, providers, policies, masks, or eager loading. It is
the behavior-preserving foundation for those later changes.

## Motivation

Bloom filter header parsing, validation, range calculation, and bitset construction are duplicated
between the core `Sbbf` reader and the synchronous and asynchronous Arrow readers. This creates a
risk that corruption handling and future format changes behave differently depending on the API
used.

The current file-read paths also contain unchecked or insufficiently validated operations:

- Signed footer lengths are cast to unsigned sizes.
- Offset additions can overflow.
- A malformed bitset length can be silently truncated by `Sbbf::new` because it consumes complete
  32-byte blocks.
- Known total lengths are not consistently checked against the decoded header length and bitset
  length.

Centralizing this logic first reduces the scope and risk of the later caching work.

## Goals

- Add fallible construction for bitsets read from files.
- Use one implementation for header validation and range calculation.
- Reject invalid offsets and lengths using checked conversions and arithmetic.
- Make all existing single-filter readers delegate to the shared implementation.
- Preserve existing public method signatures and I/O patterns.
- Produce consistent synchronous and asynchronous errors for malformed filters.

## Non-goals

- Adding `BloomFilterProvider` or attaching filters to `ParquetMetaData`.
- Adding `BloomFilterPolicy` or `ColumnChunkMask` support.
- Eagerly loading more than one filter.
- Adding multi-range I/O.
- Changing the owned return type of existing Arrow APIs.
- Changing public `Sbbf::new` behavior unless separately approved as a breaking change.

## Proposed implementation

### 1. Add validated bitset construction

File: `parquet/src/bloom_filter/mod.rs`

Add an internal fallible constructor:

```rust
impl Sbbf {
    pub(crate) fn try_from_bitset(bitset: &[u8]) -> Result<Self>;
}
```

It must validate the requirements in the Parquet Bloom filter specification:

- The bitset is at least `BITSET_MIN_LENGTH` bytes.
- The bitset is at most `BITSET_MAX_LENGTH` bytes.
- The length is divisible by the 32-byte split-block size.
- Validate the specification's power-of-two size requirement if confirmed by the spec version
  referenced by this crate. Include a spec link beside the validation.
- Every input byte is consumed; no trailing remainder is silently discarded.

Keep `Sbbf::new` source-compatible. All file decoding introduced or updated by this issue must use
`try_from_bitset`.

Add `PartialEq` and `Eq` to the internal `Block` and `Sbbf` types if required by the planned cache
work and if their representations support these traits directly.

### 2. Separate header parsing from bitset decoding

Add an internal descriptor such as:

```rust
#[derive(Debug, Clone)]
pub(crate) struct BloomFilterReadInfo {
    pub(crate) filter_offset: u64,
    pub(crate) header_len: usize,
    pub(crate) bitset_range: Range<u64>,
    pub(crate) total_len: usize,
}
```

Add shared helpers similar to:

```rust
pub(crate) fn decode_bloom_filter_header(
    filter_offset: u64,
    header_bytes: Bytes,
    advertised_total_len: Option<usize>,
) -> Result<BloomFilterReadInfo>;

pub(crate) fn decode_bloom_filter_bitset(
    info: &BloomFilterReadInfo,
    bitset: &[u8],
) -> Result<Sbbf>;
```

The exact names may change, but synchronous and asynchronous callers must use the same parsing and
validation implementation.

Header decoding must:

1. Decode the Thrift `BloomFilterHeader`.
2. Validate algorithm `BLOCK`.
3. Validate compression `UNCOMPRESSED`.
4. Validate hash `XXHASH`.
5. Fallibly convert `header.num_bytes` and reject negative values.
6. Validate the decoded bitset size.
7. Use checked arithmetic for the bitset start, end, and total serialized length.
8. If `bloom_filter_length` is present, require it to match `header_len + num_bytes`.

Do not relax the total-length check without an interoperability fixture demonstrating that a
different representation is valid.

Bitset decoding must require exactly the number of bytes described by `BloomFilterReadInfo`.

### 3. Centralize footer location conversion

Add an internal helper such as:

```rust
pub(crate) fn bloom_filter_location(
    column: &ColumnChunkMetaData,
) -> Result<Option<(u64, Option<usize>)>>;
```

Required behavior:

- Missing `bloom_filter_offset`: return `Ok(None)`.
- Negative offset: return an error.
- Negative `bloom_filter_length`: return an error.
- Never use `as usize` or `as u64` for signed metadata fields.
- Validate `offset + length` with checked arithmetic.
- Include the invalid value in the error message.

### 4. Refactor the synchronous core reader

Keep this public API unchanged:

```rust
pub fn Sbbf::read_from_column_chunk<R: ChunkReader>(
    column_metadata: &ColumnChunkMetaData,
    reader: &R,
) -> Result<Option<Sbbf>>;
```

Replace its implementation with the shared location, header, and bitset helpers.

Behavior must remain:

- No advertised offset: `Ok(None)`.
- Valid filter: `Ok(Some(filter))`.
- Advertised but malformed or truncated filter: `Err`.

For a known `bloom_filter_length`, retain one read of the complete serialized filter. For an
unknown length, retain the existing two-step pattern: read the estimated header, then read the
exact bitset range.

### 5. Refactor existing call sites

#### Synchronous Arrow reader

File: `parquet/src/arrow/arrow_reader/mod.rs`

Keep `get_row_group_column_bloom_filter` public with its existing signature. Replace duplicated
decoding with `Sbbf::read_from_column_chunk`.

#### Asynchronous Arrow reader

File: `parquet/src/arrow/async_reader/mod.rs`

Keep the existing async method and I/O pattern, but use the shared location, header, and bitset
helpers. Do not introduce eager or batch loading in this issue.

#### Serialized reader

Confirm that `SerializedRowGroupReader` continues to use `Sbbf::read_from_column_chunk`. Its lazy
row-group behavior must not change.

#### Other utilities

Audit with:

```bash
rg -n "chunk_read_bloom_filter|BloomFilterHeader|SBBF_HEADER_SIZE_ESTIMATE|Sbbf::new" parquet/src
```

Update `parquet-concat` and other file-read call sites to delegate while preserving their current
error policy.

Writer-side construction with `Sbbf::new` is outside scope unless it is directly affected by the
new internal helper.

## Error semantics

- Absence is represented only by a missing footer offset and returns `Ok(None)`.
- An advertised malformed filter returns `Err`; it is not treated as absent.
- Error messages should identify whether the offset, advertised total length, header bitset length,
  or returned byte count is invalid.
- Sync and async readers should report equivalent errors for equivalent malformed input.

## Tests

Add or extend tests covering:

### Valid input

- Filter with advertised total length.
- Legacy filter without advertised total length.
- Missing filter offset.
- Existing sync Arrow behavior.
- Existing async Arrow behavior.

### Invalid footer metadata

- Negative Bloom filter offset.
- Negative advertised total length.
- Checked-add overflow where representable by the involved types.
- Advertised total length smaller than the decoded header.
- Advertised total length different from `header_len + num_bytes`.

### Invalid header and bitset

- Truncated Thrift header.
- Negative `header.num_bytes`.
- Bitset below the minimum size.
- Bitset above the maximum size.
- Bitset length not divisible by 32.
- Truncated returned bitset.
- Returned bitset larger than expected.
- Unsupported algorithm, compression, or hash when constructible by the generated types.

### Parity and regression

- Sync and async helpers accept the same valid bytes.
- Sync and async helpers reject the same malformed bytes.
- Known-length input retains one data-source read.
- Unknown-length input retains the existing header-plus-bitset reads.
- No reader loads filters that were not requested.

Use in-memory/counting readers where needed to assert request counts.

## Compatibility requirements

- Do not remove, rename, or change public method signatures.
- Do not add eager loading.
- Do not add provider or metadata fields in this issue.
- Do not change `ReaderProperties::read_bloom_filter()` behavior.
- Keep support for files without `bloom_filter_length`.
- Preserve feature gating for async and Arrow code.

## Suggested validation

Run at minimum:

```bash
cargo fmt --all -- --check
cargo test -p parquet --lib bloom_filter
cargo test -p parquet --test arrow_reader bloom_filter
cargo clippy -p parquet --lib --all-features -- -D warnings
cargo check -p parquet --no-default-features
```

Also run any existing `parquet-concat` tests or checks affected by import and delegation changes.

## Acceptance criteria

- One shared implementation validates Bloom filter locations, headers, and bitsets.
- Sync and async readers delegate parsing and validation to that implementation.
- Existing public APIs and lazy-I/O behavior remain unchanged.
- Signed metadata values use checked conversions.
- Offset arithmetic is checked.
- File decoding cannot silently truncate a malformed bitset.
- Advertised total lengths are verified against decoded content.
- Corruption tests cover all validation branches.
- Sync and async error behavior is consistent.
- Formatting, focused tests, Clippy, and relevant feature checks pass.

## Follow-up work

The EPIC's later phases will add:

- `BloomFilterStatus` and `BloomFilterProvider`
- Attachment to `ParquetMetaData`
- Partial-cache coverage
- Masked sync/async preload
- Multi-range fetching
- Arrow reader option propagation
- Provider-aware fallback in existing readers

