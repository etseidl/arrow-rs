# Bloom Filter Caching Implementation Plan

## Purpose

Add cacheable Bloom filter access to the Rust Parquet implementation without changing the existing lazy-I/O behavior of the Arrow and serialized readers.

The implementation must:

1. Have one validated Bloom filter decoding implementation shared by synchronous and asynchronous readers.
2. Allow decoded filters to be attached to `ParquetMetaData` or stored in an external cache.
3. Distinguish a cached absence from a cache miss.
4. Preserve existing public APIs and their on-demand behavior.
5. Support masked eager loading as an explicit opt-in.
6. Avoid one serial object-store request per filter when batch range reads are available.
7. Integrate with all metadata-loading entry points, including the push decoder and Arrow reader options.

Complete each phase and its tests before moving to the next phase.

---

## Non-Goals

The first implementation will not:

- Mutate immutable metadata when an on-demand read misses the cache. On-demand APIs may read and return a filter; explicit preload APIs populate caches.
- Implement eviction or LRU policy. Applications can implement these in an external provider.
- Claim that `BloomFilterProvider` performs I/O. It exposes already available cache state.
- Remove or rename existing public Bloom filter methods.
- Eagerly load Bloom filters by default.

---

## Existing Behavior That Must Be Preserved

Before changing code, add or identify tests that lock down these behaviors:

1. `ArrowReaderBuilder::get_row_group_column_bloom_filter` reads only the requested filter and returns an owned `Option<Sbbf>`.
2. `ParquetRecordBatchStreamBuilder::get_row_group_column_bloom_filter` does the asynchronous equivalent.
3. Both Arrow methods can be called after other pruning without loading every filter during metadata construction.
4. `SerializedFileReader` reads filters only when `ReaderProperties::read_bloom_filter()` is enabled and a row group reader is opened.
5. `ReaderProperties::read_bloom_filter()` defaults to `false`. Correct the existing doc comment claiming it defaults to enabled.
6. Files with and without `bloom_filter_length` remain readable.
7. Public constructors and method signatures keep compiling.

Relevant code:

- `parquet/src/bloom_filter/mod.rs`: `Sbbf::read_from_column_chunk`
- `parquet/src/file/serialized_reader.rs`: `SerializedRowGroupReader::new`
- `parquet/src/arrow/arrow_reader/mod.rs`: `get_row_group_column_bloom_filter`
- `parquet/src/arrow/async_reader/mod.rs`: `get_row_group_column_bloom_filter`
- `parquet/tests/arrow_reader/bloom_filter/`

---

## Core Design Decisions

### Cache entry state

Every coordinate has one of three states:

```rust
#[derive(Debug, Clone, Copy)]
pub enum BloomFilterStatus<'a> {
    /// A decoded filter is available in this provider.
    Present(&'a Sbbf),
    /// This coordinate was loaded and the file metadata says it has no filter.
    Absent,
    /// This provider does not cover the coordinate.
    NotLoaded,
}
```

Fallback behavior is:

- `Present`: use the cached filter.
- `Absent`: return `None` without I/O.
- `NotLoaded`: an on-demand reader may read from the file.

An out-of-bounds coordinate returns `NotLoaded`. Existing public APIs may still panic for invalid row-group or column indices before consulting the provider if that is their current behavior.

### Loading policy

Introduce a Bloom-filter-specific policy with exact semantics:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BloomFilterPolicy {
    /// Do not load Bloom filters. Preserve an already attached provider.
    #[default]
    Skip,
    /// Load selected coordinates. Missing filters are recorded as `Absent`.
    /// A malformed filter advertised by the footer remains an error.
    Optional,
}
```

Unlike `PageIndexPolicy`, this policy intentionally has no `Required` variant.
`PageIndexPolicy::Required` is a historical compatibility mechanism from when offset-index storage
could not represent a missing index. Bloom-filter caching has an explicit `Absent` state and does
not need the same behavior. A selected column chunk without a Bloom filter is therefore always a
valid result under `Optional`.

`Optional` describes whether absence is allowed; it does not suppress corruption. If the footer
advertises a Bloom filter but its offset, length, header, or bitset is invalid, loading returns an
error.

`Skip` must not clear an already attached provider. The metadata builder can explicitly remove one.

### Default behavior

- `ParquetMetaDataReader`, `ArrowReaderOptions`, and Arrow metadata loading default to `Skip`.
- Existing Arrow single-filter methods remain on demand.
- `SerializedFileReader` remains governed by `ReaderProperties::read_bloom_filter()` and loads at row-group-open time unless filters are cached.
- Eager loading occurs only when `Optional` is explicitly configured.

### Provider scope

`BloomFilterProvider` represents already available state. It supports in-memory and external caches but does not promise lazy or asynchronous I/O. This keeps borrowed access safe and makes the trait object usable from `ParquetMetaData`.

---

## Phase 1: Centralize and Harden Decoding

### Goal

Create one implementation for parsing and validating Bloom filters. All old and new readers must delegate to it.

### 1.1 Add validated bitset construction

File: `parquet/src/bloom_filter/mod.rs`

Add a fallible internal constructor instead of using `Sbbf::new` for file data:

```rust
impl Sbbf {
    pub(crate) fn try_from_bitset(bitset: &[u8]) -> Result<Self> {
        // Validate, then construct without discarding bytes.
    }
}
```

Validate requirements from the Parquet Bloom filter specification:

- Length is at least `BITSET_MIN_LENGTH`.
- Length is at most `BITSET_MAX_LENGTH`.
- Length is a multiple of 32 bytes.
- Validate the specification's power-of-two requirement if confirmed by the spec version referenced by this crate. Cite the spec beside the check.
- No bytes are silently discarded.

Keep public `Sbbf::new` behavior for source compatibility unless maintainers approve a breaking change. File decoding must use `try_from_bitset`.

Add `PartialEq` and `Eq` to `Block` and `Sbbf`; deterministic metadata equality needs them.

### 1.2 Separate header parsing from bitset decoding

Add an internal descriptor:

```rust
#[derive(Debug, Clone)]
pub(crate) struct BloomFilterReadInfo {
    pub(crate) filter_offset: u64,
    pub(crate) header_len: usize,
    pub(crate) bitset_range: Range<u64>,
    pub(crate) total_len: usize,
}
```

Add:

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

`decode_bloom_filter_header` must:

1. Decode the Thrift header.
2. Validate algorithm `BLOCK`, compression `UNCOMPRESSED`, and hash `XXHASH`.
3. Fallibly convert `header.num_bytes` and reject negative values.
4. Validate the bitset length through step 1.1.
5. Use `checked_add` for every offset or length calculation.
6. If `bloom_filter_length` exists, require it to equal `header_len + num_bytes`. If interoperability fixtures demonstrate legal padding, document that evidence before relaxing the check.
7. Return errors containing the invalid offset or length.

`decode_bloom_filter_bitset` must require exactly the expected number of bytes.

### 1.3 Centralize metadata-to-range conversion

Add:

```rust
pub(crate) fn bloom_filter_location(
    column: &ColumnChunkMetaData,
) -> Result<Option<(u64, Option<usize>)>>;
```

It must:

- Return `Ok(None)` only when the offset is absent.
- Reject negative offsets.
- Reject negative `bloom_filter_length`; never use `as usize` or `as u64`.
- Validate `offset + length` with checked arithmetic.

### 1.4 Make the synchronous reader delegate

Keep `Sbbf::read_from_column_chunk` public and replace its internals with the helpers above.

- No offset: `Ok(None)`.
- Malformed or truncated advertised filter: `Err`.
- Valid filter: `Ok(Some(filter))`.

Do not document malformed filters as `Ok(None)`.

Known total length needs one read. A missing total length needs a header-estimate read followed by the exact bitset read.

### 1.5 Replace duplicate single-filter decoding

- Sync Arrow method delegates to `Sbbf::read_from_column_chunk`.
- Async Arrow retains async I/O but delegates parsing to the shared helpers.
- `parquet-concat` delegates while preserving its existing error policy.

There must be no independent copies of header validation or `Sbbf::new(&bitset)` in readers.

### 1.6 Tests

Add tests for:

- Valid known-length and unknown-length filters.
- Missing and negative offsets.
- Negative advertised length.
- Overflow in `offset + length`.
- Negative `header.num_bytes`.
- Bitset smaller or larger than advertised.
- Bitset below minimum, above maximum, or not divisible by 32.
- Total-length mismatch.
- Truncated header and bitset.
- Equal sync and async errors for the same corruption.

Acceptance checks:

```bash
cargo test -p parquet --lib bloom_filter
cargo test -p parquet --test arrow_reader bloom_filter
```

---

## Phase 2: Add the Provider and Default Cache

### Goal

Represent complete and partial caches without confusing absence with a cache miss.

### 2.1 Create and export the module

New file: `parquet/src/file/metadata/bloom_filter.rs`

Export from `metadata/mod.rs`:

```rust
pub mod bloom_filter;
pub use bloom_filter::{
    BloomFilterIndex, BloomFilterIndexBuilder, BloomFilterProvider,
    BloomFilterStatus, RowGroupBloomFilters,
};
```

### 2.2 Define the provider

```rust
pub trait BloomFilterProvider: Send + Sync + Debug {
    fn status(
        &self,
        row_group_idx: usize,
        column_idx: usize,
    ) -> BloomFilterStatus<'_>;

    fn bloom_filter(&self, row_group_idx: usize, column_idx: usize) -> Option<&Sbbf> {
        match self.status(row_group_idx, column_idx) {
            BloomFilterStatus::Present(filter) => Some(filter),
            BloomFilterStatus::Absent | BloomFilterStatus::NotLoaded => None,
        }
    }

    fn has_bloom_filters(&self) -> bool;
    fn as_any(&self) -> &dyn Any;
}
```

Document that fallback-I/O callers must use `status`, not `bloom_filter`.

### 2.3 Implement `BloomFilterIndex`

Use sparse storage recording present and absent loaded coordinates:

```rust
pub struct BloomFilterIndex {
    entries: HashMap<(usize, usize), Option<Sbbf>>,
    num_row_groups: usize,
    num_columns: usize,
}
```

Semantics:

- Missing key: `NotLoaded`.
- Key mapped to `None`: `Absent`.
- Key mapped to `Some`: `Present`.
- Out of bounds: `NotLoaded`.

Provide:

- `len()` for covered coordinates, including absences.
- `num_filters()` for present filters.
- `is_empty()` based on coverage.
- `is_complete()` using checked multiplication.
- Dimension accessors.
- Iterators over present entries and all covered entries.

Do not call a partial cache complete merely because it contains every filter present in the file.

### 2.4 Implement the builder

`try_new` validates dimensions against `i32::MAX`, matching page-index builders. `new` may panic by delegating to `try_new`.

Provide:

```rust
put_filter(filter, row_group_idx, column_idx) -> Result<&mut Self>
put_absent(row_group_idx, column_idx) -> Result<&mut Self>
build() -> BloomFilterIndex
```

Reject out-of-bounds and duplicate coordinates. Do not silently discard programming errors.

### 2.5 Add the row-group accessor

`RowGroupBloomFilters` owns `Option<Arc<dyn BloomFilterProvider>>` and a row-group index. Provide:

```rust
pub fn status(&self, column_idx: usize) -> BloomFilterStatus<'_>;
pub fn bloom_filter(&self, column_idx: usize) -> Option<&Sbbf>;
```

No provider means `NotLoaded`, not `Absent`.

### 2.6 Equality and memory accounting

- Implement `PartialEq` for `BloomFilterIndex` after `Sbbf` supports equality.
- Implement `HeapSize` using the existing generic `HashMap` accounting where possible.
- `heap_size` excludes inline `size_of::<Self>()`; its caller adds that.
- Count the `Arc` allocation overhead once in `ParquetMetaData::memory_size`.
- Custom providers may report zero, consistently with the current `PageIndexProvider` limitation.

### 2.7 Tests

Test present, absent, not-loaded, out-of-bounds, partial and complete coverage, duplicates, dimension errors, zero dimensions, equality, memory growth, and row-group access.

---

## Phase 3: Attach the Provider to `ParquetMetaData`

### 3.1 Add the field

Add to `ParquetMetaData` and `ParquetMetaDataBuilder`:

```rust
bloom_filters: Option<Arc<dyn BloomFilterProvider>>,
```

Update every constructor, destructuring pattern, `From` implementation, and `build` method. Find construction sites with:

```bash
rg "ParquetMetaData \\{" parquet
rg "ParquetMetaDataBuilder \\{" parquet
```

### 3.2 Add metadata accessors

```rust
pub fn bloom_filters(&self) -> Option<&Arc<dyn BloomFilterProvider>>;
pub fn bloom_filters_for_row_group(&self, row_group_idx: usize) -> RowGroupBloomFilters;
pub(crate) fn set_bloom_filters(
    &mut self,
    provider: Option<Arc<dyn BloomFilterProvider>>,
);
```

### 3.3 Add builder APIs

```rust
pub fn set_bloom_filters(
    self,
    provider: Option<Arc<dyn BloomFilterProvider>>,
) -> Self;
pub fn take_bloom_filters(&mut self) -> Option<Arc<dyn BloomFilterProvider>>;
pub fn bloom_filters(&self) -> Option<&dyn BloomFilterProvider>;
```

Changing row-group order after attaching coordinate-based providers can invalidate them. Either clear both page-index and Bloom providers automatically or document that callers must replace both. Choose and test one consistent behavior.

### 3.4 Equality and memory size

- Compare default `BloomFilterIndex` providers by value.
- For unknown provider types, use `Arc::ptr_eq`, matching page indexes.
- Extend `memory_size` through the same downcast pattern as `PageIndex`.

### 3.5 Tests

Test shared allocation on clone, builder round trip, take/set APIs, default-provider value equality, custom-provider pointer equality, coordinate mutation behavior, and memory accounting.

---

## Phase 4: Implement Masked Batch Loading

### 4.1 Add reader configuration

Add to `ParquetMetaDataReader`:

```rust
bloom_filter_policy: BloomFilterPolicy,
bloom_filter_mask: ColumnChunkMask,
```

Defaults are `Skip` and `ColumnChunkMask::all()`.

Add `with_bloom_filter_policy` and `with_bloom_filter_mask`. The mask selects leaf columns. Iterate each row group's actual `num_columns()` when handling metadata.

### 4.2 Add multi-range metadata fetching

Extend `MetadataFetch` compatibly with a default method:

```rust
fn fetch_ranges(
    &mut self,
    ranges: Vec<Range<u64>>,
) -> BoxFuture<'_, Result<Vec<Bytes>>>;
```

The default may fetch sequentially. Override the implementation for `&mut T where T: AsyncFileReader` by delegating to `get_byte_ranges`.

Requirements:

- Buffers correspond one-for-one and in order with ranges.
- Reject an incorrect buffer count or length.
- Do not coalesce widely separated filters into a span containing column data.
- A coalescer may merge overlapping or immediately adjacent ranges using a small documented threshold.

### 4.3 Integrate `ParquetMetaDataPushDecoder`

Add Bloom policy/mask fields, setters, a decode path returning multiple `NeedsData` ranges, and two-round decoding for missing lengths.

Suggested method:

```rust
pub fn try_decode_bloom_filters(
    &mut self,
) -> Result<DecodeResult<(ParquetMetaData, Option<BloomFilterIndex>)>>;
```

Algorithm:

1. Visit every selected coordinate.
2. If no offset exists, record `Absent`.
3. If offset and total length exist, request the exact full range.
4. If length is absent, request the header-estimate range.
5. Return all missing first-round ranges together.
6. Decode headers using Phase 1 helpers.
7. Request exact unknown-length bitset ranges not already buffered.
8. Return all missing second-round ranges together.
9. Decode exact bitsets and insert `Present` entries.
10. Return `None` only for Skip or a mask selecting no coordinates. Optional all-absent loads return a cache so negative entries are retained.

Ensure normal `try_decode` can attach the cache when eager loading is configured. Skip must preserve an existing provider.

### 4.4 Add synchronous reader APIs

```rust
pub fn read_bloom_filters<R: ChunkReader>(&mut self, reader: &R) -> Result<()>;

pub fn read_bloom_filter_and_finish<R: ChunkReader>(
    self,
    reader: &R,
) -> Result<(ParquetMetaData, Option<BloomFilterIndex>)>;
```

The first attaches the cache. The second returns it separately and leaves an existing attached provider unchanged. Fetch exact disjoint ranges requested by the push decoder.

If a `&mut self` operation fails after taking metadata, restore the metadata before returning, matching page-index error-state expectations.

### 4.5 Add asynchronous APIs

```rust
pub async fn load_bloom_filters<F: MetadataFetch>(
    &mut self,
    fetch: F,
) -> Result<()>;

pub async fn load_bloom_filter_and_finish<F: MetadataFetch>(
    self,
    fetch: F,
) -> Result<(ParquetMetaData, Option<BloomFilterIndex>)>;
```

Drive `NeedsData` with `fetch_ranges`, not a loop awaiting one filter at a time.

### 4.6 Integrate normal entry points

For non-Skip policy, integrate `try_parse`, sized variants, `parse_and_finish`, `try_load`, `try_load_via_suffix`, and finish variants. Reuse bytes already held by the push decoder. Under Skip, retain existing request counts.

### 4.7 Policy and range tests

Build a matrix with all/some/no filters, empty/subset masks, known/unknown/mixed lengths, and both policies.

Assert:

- Exact `Present`/`Absent`/`NotLoaded` states.
- Optional never suppresses corruption.
- Skip performs no extra I/O and preserves providers.
- Sync and async results match.
- Known-length async loads use one range round after footer metadata.
- Unknown-length loads use at most two rounds.
- `AsyncFileReader::get_byte_ranges` is actually used.

---

## Phase 5: Propagate Arrow Reader Options

### 5.1 Extend `ArrowReaderOptions`

Add policy and mask fields, setters, and accessors. Default policy is Skip:

```rust
pub fn with_bloom_filter_policy(self, policy: BloomFilterPolicy) -> Self;
pub fn with_bloom_filter_mask(self, mask: ColumnChunkMask) -> Self;
pub fn bloom_filter_policy(&self) -> BloomFilterPolicy;
pub fn bloom_filter_mask(&self) -> &ColumnChunkMask;
```

### 5.2 Update the canonical option bridge

Update `ParquetMetaDataReader::with_arrow_reader_options`. Do not accidentally overwrite a policy configured directly on the metadata reader when Arrow options are default. Document and test precedence.

### 5.3 Audit every metadata path

Update:

- `ArrowReaderMetadata::load`
- `ArrowReaderMetadata::load_async`
- `ArrowReaderMetadata::try_new`
- Sync and async `try_new_with_options`
- `AsyncFileReader::get_metadata` implementations
- `ParquetObjectReader` preload compatibility paths
- Test/mock readers constructing `ParquetMetaDataReader`

Audit with:

```bash
rg "with_arrow_reader_options|ArrowReaderMetadata::|get_metadata" parquet
```

Rules for preexisting metadata:

- Skip uses the provider as-is.
- Complete selected coverage causes no I/O.
- Incomplete coverage plus an input reader loads only missing coordinates and merges with an existing default cache.
- Do not assume a custom provider can be converted to the default implementation. If retaining both is required, implement a small composite provider; otherwise document replacement behavior.
- `try_new` paths without an input cannot fill `NotLoaded`. Optional retains partial coverage
  without error; a later loading path with an input reader may fill it.

### 5.4 Tests

Verify default construction performs no Bloom I/O, Optional attaches a cache, masks propagate
through sync/async paths, complete caches cause no requests, and partial caches request only
missing coordinates.

---

## Phase 6: Preserve On-Demand APIs and Add Cache Lookup

### 6.1 Synchronous Arrow builder

Keep the public signature unchanged:

```rust
pub fn get_row_group_column_bloom_filter(
    &self,
    row_group_idx: usize,
    column_idx: usize,
) -> Result<Option<Sbbf>>;
```

Implementation:

1. Access metadata as today.
2. Consult the provider.
3. `Present`: clone and return, preserving the owned return type.
4. `Absent`: return `None` without I/O.
5. `NotLoaded` or no provider: call `Sbbf::read_from_column_chunk`.

The metadata provider is the borrowed zero-copy path. Do not remove or deprecate the owned method in the same release.

### 6.2 Asynchronous Arrow builder

Keep the existing signature and apply the same status logic. On `NotLoaded`, use the shared async decoding helpers.

### 6.3 Serialized reader

Do not change the public `SerializedRowGroupReader::new` signature. Add a private/internal constructor accepting `RowGroupBloomFilters`; the public constructor delegates without a provider and retains current behavior.

When `read_bloom_filter` is false, access remains disabled. When enabled:

- `Present`: retain provider access; do not clone into the fallback vector.
- `Absent`: do not read.
- `NotLoaded`: perform the legacy row-group-open read and retain it locally.

Possible representation:

```rust
struct RowGroupBloomFilterState {
    cached: RowGroupBloomFilters,
    loaded_fallback: Vec<Option<Sbbf>>,
}
```

`get_column_bloom_filter` checks cached state first, then fallback. Do not add an unused row-group index field.

### 6.4 Tests

With counting readers verify cache hits and cached absences perform no reads, misses preserve legacy reads, async matches sync, serialized reading touches only opened row groups, provider hits are not cloned into fallback storage, disabled reading remains disabled, and public API tests compile unchanged.

---

## Phase 7: External Cache Workflow and Documentation

### 7.1 Add a compiling example

Show how to:

1. Load footer metadata under Skip.
2. Create a metadata reader with that metadata.
3. Set Optional plus a mask.
4. Call the sync or async `*_and_finish` API.
5. Store the returned `BloomFilterIndex` externally.
6. Attach it later through `ParquetMetaDataBuilder::set_bloom_filters`.

Use distinct names for the I/O source and metadata reader; do not repeat the variable-shadowing bug from the old plan.

### 7.2 Document partial caches

Explain that `Present` is a hit, `Absent` is a negative-cache hit, and `NotLoaded` permits I/O. State that convenience `bloom_filter()` collapses Absent and NotLoaded.

### 7.3 Add an end-to-end test

Write a multi-row-group file with filters on some columns, load a mask, reattach it to freshly decoded metadata, prove present and absent cache entries cause no reads, prove uncovered entries fall back, and repeat asynchronously with a counting multi-range reader.

---

## Phase 8: Performance and Compatibility Validation

### Request-count validation

Measure legacy one-filter access, known-length preload, unknown-length preload, complete-cache reuse, and partial fallback.

Expected:

- Default construction has no new requests.
- One on-demand miss is no worse than current behavior.
- Known-length async preload uses one batch round.
- Unknown-length preload uses at most two batch rounds after footer metadata.
- Complete-cache reuse performs zero Bloom I/O.

### Storage benchmark

Compare sparse `HashMap<(usize, usize), Option<Sbbf>>` with a dense row-major vector for sparse masks, dense filters, lookup throughput, and memory. A dense representation with explicit NotLoaded state is acceptable if materially better; preserve public semantics.

### Feature and platform checks

Run:

```bash
cargo fmt --all -- --check
cargo clippy -p parquet --all-targets --all-features -- -D warnings
cargo test -p parquet --all-features
cargo test -p parquet --no-default-features
```

Also verify:

- Async-disabled builds do not import futures or `MetadataFetch`.
- Arrow-disabled builds retain synchronous support.
- Encryption builds compile. Do not claim encrypted Bloom-filter support without an interoperability test.
- Checked conversions work on 32-bit targets; run a 32-bit `cargo check` if CI supports it.

---

## File-by-File Checklist

### `parquet/src/bloom_filter/mod.rs`

- Add fallible bitset construction and validation helpers.
- Delegate `Sbbf::read_from_column_chunk`.
- Add equality and memory support.
- Add malformed-input tests.

### `parquet/src/file/metadata/bloom_filter.rs` (new)

- Add status, provider, default index, builder, row-group accessor, and tests.

### `parquet/src/file/metadata/mod.rs`

- Export types; attach provider to metadata/builder; add APIs, equality, and memory accounting.

### `parquet/src/file/metadata/reader.rs`

- Add policy/mask, sync/async preload APIs, push-decoder driving, and Arrow option propagation.

### `parquet/src/file/metadata/push_decoder.rs`

- Add Bloom configuration and multi-range two-round decoding.

### `parquet/src/arrow/async_reader/metadata.rs`

- Add compatible `fetch_ranges`; delegate to `get_byte_ranges` where available.

### `parquet/src/arrow/arrow_reader/mod.rs`

- Extend options and metadata construction; keep provider-aware public on-demand method.

### `parquet/src/arrow/async_reader/mod.rs`

- Keep provider-aware public async method; use batch ranges for preload.

### `parquet/src/file/serialized_reader.rs`

- Preserve public constructors and lazy row-group behavior; consult cache before fallback.

### `parquet/src/file/properties.rs`

- Correct inaccurate default documentation.

### Tests and examples

- Extend existing Arrow Bloom tests.
- Add provider/policy unit tests near implementations.
- Add one external-cache integration test.
- Add counting sync and async readers.

---

## Pull Request Breakdown

1. **Validated shared decoding**: Phase 1 only; no behavioral change.
2. **Provider and metadata attachment**: Phases 2–3; no automatic loading.
3. **Masked batch loader and push decoder**: Phase 4.
4. **Arrow option propagation**: Phase 5.
5. **Provider-aware existing readers**: Phase 6; preserve APIs.
6. **Documentation, integration tests, and benchmarks**: Phases 7–8.

Each commit must compile and pass relevant tests. Never land a change that makes default reader construction eager.

---

## Definition of Done

1. All sync and async paths use one validated decoder.
2. Partial caches distinguish `Absent` from `NotLoaded`.
3. Existing public on-demand signatures remain available.
4. Default metadata and Arrow construction performs no Bloom I/O.
5. Explicit masked loading works synchronously and asynchronously.
6. Async preload uses multi-range fetching when supported.
7. Known-length preload needs one range round; unknown-length needs at most two after footer metadata.
8. Skip and Optional semantics are tested and documented.
9. External caches can be returned separately and reattached through the builder.
10. Cached present and absent entries suppress I/O; uncovered entries retain fallback behavior.
11. Malformed offsets, lengths, headers, and bitsets error without panics, truncation, or unchecked casts.
12. Sync, async, Arrow-disabled, async-disabled, all-feature, and encryption-feature builds pass.
