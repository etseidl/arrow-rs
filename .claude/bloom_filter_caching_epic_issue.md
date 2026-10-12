# EPIC: Add cacheable Bloom filter access to Parquet metadata

## Summary

Add cacheable Bloom filter access to the Rust Parquet implementation while preserving the existing
lazy-I/O behavior of the Arrow and serialized readers.

This work will centralize Bloom filter decoding, add an explicit provider interface to
`ParquetMetaData`, support partial and external caches, add opt-in masked preloading, and use
batched range reads for asynchronous sources.

## Motivation

Bloom filter reading is currently implemented independently in several places:

- `Sbbf::read_from_column_chunk`
- `SerializedRowGroupReader`
- The synchronous Arrow reader builder
- The asynchronous Arrow stream builder
- Supporting utilities such as `parquet-concat`

These paths duplicate parsing and validation, and decoded filters cannot currently be attached to
`ParquetMetaData` for reuse. Applications that cache Parquet metadata must therefore either reread
Bloom filters or maintain a separate application-specific access path.

Page indexes provide a useful precedent for storing optional auxiliary structures behind a
provider, but Bloom filters need explicit cache-coverage semantics. A missing cache entry must
distinguish:

- A decoded filter is present.
- The coordinate was loaded and the file contains no filter.
- The coordinate was not loaded and on-demand I/O may still be useful.

Without this distinction, a partial cache cannot safely suppress or trigger fallback I/O.

## Goals

- Use one validated decoder for all synchronous and asynchronous Bloom filter reads.
- Attach a shareable Bloom filter provider to `ParquetMetaData`.
- Represent `Present`, `Absent`, and `NotLoaded` cache states.
- Preserve all existing public on-demand APIs and signatures.
- Preserve lazy loading by default.
- Add explicit, masked preload APIs for synchronous and asynchronous sources.
- Batch asynchronous range requests when the underlying reader supports them.
- Allow decoded filters to be returned separately for external cache population.
- Propagate Bloom filter loading options through every Arrow metadata-loading path.
- Reject malformed advertised filters consistently without panics or unchecked conversions.

## Non-goals

- Implementing cache eviction, LRU behavior, or a global cache.
- Making `BloomFilterProvider` perform synchronous or asynchronous I/O.
- Populating immutable metadata caches as a side effect of an on-demand cache miss.
- Removing or renaming existing public Bloom filter APIs.
- Loading Bloom filters eagerly by default.
- Adding a `Required` loading policy.

## Proposed design

### Loading policy

Add a two-variant policy:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BloomFilterPolicy {
    #[default]
    Skip,
    Optional,
}
```

- `Skip` performs no Bloom filter I/O and preserves a provider already attached to metadata.
- `Optional` loads selected coordinates and records missing filters as known `Absent` entries.
- If the footer advertises a filter but its location, header, or bitset is malformed, loading
  returns an error.

Unlike `PageIndexPolicy`, this policy intentionally has no `Required` variant.
`PageIndexPolicy::Required` is a historical compatibility mechanism from when offset-index
storage could not represent a missing index. Bloom filter caching will explicitly represent
absence and does not need the same behavior.

### Cache status

The provider exposes three states:

```rust
pub enum BloomFilterStatus<'a> {
    Present(&'a Sbbf),
    Absent,
    NotLoaded,
}
```

- `Present` is a cache hit.
- `Absent` is a negative-cache hit and suppresses I/O.
- `NotLoaded` permits an existing on-demand API to fall back to file I/O.

An `Option<&Sbbf>` convenience method may remain available, but callers deciding whether to issue
I/O must use the status API.

### Default provider

Add an immutable default provider, tentatively named `BloomFilterIndex`, with sparse storage for
loaded coordinates:

```rust
HashMap<(usize, usize), Option<Sbbf>>
```

- Missing key: `NotLoaded`
- `None` value: `Absent`
- `Some(Sbbf)`: `Present`

A builder will validate dimensions and reject duplicate or out-of-bounds insertion.

### Metadata integration

Add the following runtime-only field to `ParquetMetaData` and its builder:

```rust
bloom_filters: Option<Arc<dyn BloomFilterProvider>>
```

Add direct and row-group-scoped accessors, builder set/take methods, equality behavior matching the
page-index provider, and memory accounting for the default implementation.

### Loading and range I/O

`ParquetMetaDataReader` and `ParquetMetaDataPushDecoder` will support policy and
`ColumnChunkMask` configuration.

Known-length filters can be fetched in one range round. Legacy filters without
`bloom_filter_length` require at most two rounds after footer metadata is available:

1. Fetch and decode all selected headers.
2. Fetch the exact bitset ranges computed from those headers.

Extend `MetadataFetch` with a compatible default multi-range method. Implementations backed by
`AsyncFileReader` should delegate to `get_byte_ranges`.

### Existing readers

Existing reader methods remain on demand:

- Provider `Present`: return the cached filter, cloning only where the existing API returns an
  owned `Sbbf`.
- Provider `Absent`: return `None` without I/O.
- Provider `NotLoaded` or no provider: use the existing file-read behavior.

`SerializedFileReader` must continue to read Bloom filters only when enabled and when a row group
is opened. Attaching a provider must not make file construction eager.

## Implementation phases

### Phase 1: Centralize and harden decoding

- Add validated, fallible bitset construction.
- Separate header parsing/range calculation from bitset decoding.
- Centralize conversion of footer offsets and lengths.
- Make all existing sync and async single-filter paths delegate to shared helpers.
- Add malformed-input and sync/async parity tests.

This phase must not introduce provider APIs or alter I/O behavior.

### Phase 2: Add the provider and default cache

- Add `BloomFilterStatus` and `BloomFilterProvider`.
- Add the default immutable cache and builder.
- Record both positive and negative cache entries.
- Add row-group-scoped access.
- Add coverage, equality, and memory tests.

### Phase 3: Attach the provider to `ParquetMetaData`

- Add the provider field to metadata and its builder.
- Add accessors and builder set/take methods.
- Define metadata equality for default and custom providers.
- Add default-provider memory accounting.
- Document behavior when row-group coordinates are reordered.

### Phase 4: Add masked batch loading

- Add policy and mask configuration to metadata reader and push decoder.
- Add multi-range fetch support.
- Support known-length and two-round unknown-length loading.
- Add sync/async preload and `*_and_finish` APIs.
- Integrate regular metadata parse/load entry points.
- Test policies, masks, cache states, errors, and request counts.

### Phase 5: Propagate Arrow reader options

- Add policy and mask to `ArrowReaderOptions`.
- Update `with_arrow_reader_options`.
- Audit sync, async, object-store, cached-metadata, and custom-reader paths.
- Avoid I/O for complete cache coverage and load only uncovered coordinates.

### Phase 6: Make existing readers provider-aware

- Preserve synchronous and asynchronous Arrow method signatures.
- Consult cache status before fallback I/O.
- Preserve `SerializedRowGroupReader::new` and lazy row-group loading.
- Add counting-reader regression tests.

### Phase 7: Document external caching

- Add a compiling cache population and reattachment example.
- Explain positive, negative, and uncovered cache states.
- Add an end-to-end sync/async external-cache test.

### Phase 8: Validate performance and compatibility

- Verify request counts for lazy, preload, and cached paths.
- Compare sparse and dense storage for realistic workloads.
- Run formatting, Clippy, feature-matrix, encryption, and platform checks.

## Compatibility requirements

- Existing public Bloom filter methods remain available with unchanged signatures.
- Default reader and metadata construction performs no new Bloom filter I/O.
- `ReaderProperties::read_bloom_filter()` continues to control serialized-reader access.
- Existing files with and without `bloom_filter_length` remain readable.
- `Skip` never clears an attached provider implicitly.
- Optional absence is not treated as corruption.
- Advertised-but-malformed filters continue to return errors.

## Testing strategy

Tests should include:

- Valid filters with and without advertised length.
- Missing, negative, overflowing, truncated, and inconsistent locations.
- Invalid headers and bitset sizes.
- `Present`, `Absent`, and `NotLoaded` cache states.
- Complete and partial masks.
- Sync/async result and error parity.
- Counting readers proving default and cached paths avoid I/O.
- One-round known-length and at-most-two-round unknown-length async preload.
- External cache extraction and reattachment.
- Existing public API compile coverage.

## Suggested PR sequence

1. Shared validated decoding — Phase 1
2. Provider and metadata attachment — Phases 2 and 3
3. Masked batch loader and push decoder — Phase 4
4. Arrow option propagation — Phase 5
5. Provider-aware existing readers — Phase 6
6. Documentation, integration tests, and benchmarks — Phases 7 and 8

Each PR should compile and pass its relevant tests independently. No intermediate PR should make
default reader construction eager.

## Definition of done

- All sync and async paths use one validated decoder.
- Partial caches distinguish `Absent` from `NotLoaded`.
- Existing public on-demand APIs remain unchanged.
- Default construction performs no Bloom filter I/O.
- Explicit masked loading works synchronously and asynchronously.
- Async preload uses batch range fetching when supported.
- Known-length loading needs one range round; legacy loading needs at most two after footer
  metadata.
- `Skip` and `Optional` semantics are tested and documented.
- External caches can be returned separately and reattached.
- Cached present and absent entries suppress I/O; uncovered entries retain fallback behavior.
- Malformed data returns errors without panics, truncation, or unchecked casts.
- Relevant feature combinations and Clippy pass.

## Related issues

- Phase 1 issue: centralize and harden Bloom filter decoding (filed separately)
- Page-index cache/provider work that establishes the metadata provider pattern

