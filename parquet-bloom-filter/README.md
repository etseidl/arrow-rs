<!---
  SPDX-License-Identifier: Apache-2.0
  SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
  SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Parquet Bloom Filters

Standalone split block Bloom filters compatible with the [Parquet Bloom filter
specification](https://github.com/apache/parquet-format/blob/master/BloomFilter.md).

This crate provides xxHash64 seed-0 byte hashing, byte and prehashed insertion and
membership checks, sizing, raw little-endian bitset import/export, false positive
probability estimation, folding, and heap allocation accounting. Its only dependency
is `twox-hash`; it has no Arrow or Parquet dependency.

Use [`parquet::bloom_filter::Sbbf`](https://docs.rs/parquet/latest/parquet/bloom_filter/struct.Sbbf.html)
for Parquet headers, file reading, and `AsBytes` support. Consuming conversions between
the standalone filter and the Parquet wrapper move the bitset without copying it.

For Parquet compatibility, hash the plain-encoded physical value bytes with
`hash_bytes` (without length prefixes for byte arrays). This crate does not
convert logical or Arrow values into their Parquet physical representation.
Precomputed hashes can use another well-mixed 64-bit hash function, provided
insertion and lookup use the same encoding and hash function; such filters
are not necessarily Parquet-compatible.

The raw `Sbbf::new` constructor imports only complete 32-byte blocks and ignores
trailing bytes. Fewer than 32 bytes create a zero-block filter, on which insertion
and membership checks panic. Use a power-of-two block count for folding; the target
FPP passed to folding is not validated.

See the [API documentation](https://docs.rs/parquet-bloom-filter/latest) for details.

## License

Licensed under the Apache License, Version 2.0: <http://www.apache.org/licenses/LICENSE-2.0>.
