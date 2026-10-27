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

//! Shared, transport-independent single-filter read planning.

use super::*;
use crate::parquet_thrift::{ThriftProtocolError, ThriftProtocolResult};
use std::ops::Range;

/// Resource limits for loading a single Bloom filter.
///
/// Defaults allow a 64 KiB header and a 128 MiB bitset. These are resource
/// policies, not wire-format maxima. A larger valid filter requires explicit
/// opt-in. Limits cover this loader's requests, not allocations inside a source.
#[derive(Debug, Clone)]
pub struct BloomFilterReadOptions {
    max_header_bytes: usize,
    max_bitset_bytes: usize,
    file_length: Option<u64>,
}

impl Default for BloomFilterReadOptions {
    fn default() -> Self {
        Self {
            max_header_bytes: 64 * 1024,
            max_bitset_bytes: BITSET_MAX_LENGTH,
            file_length: None,
        }
    }
}

impl BloomFilterReadOptions {
    /// Set the maximum serialized header size.
    pub fn with_max_header_bytes(mut self, limit: usize) -> Self {
        self.max_header_bytes = limit;
        self
    }

    /// Set the maximum decoded bitset size.
    pub fn with_max_bitset_bytes(mut self, limit: usize) -> Self {
        self.max_bitset_bytes = limit;
        self
    }

    /// Set the known file length for range validation, especially for async I/O.
    /// Without it, EOF detection is the range source's responsibility.
    pub fn with_file_length(mut self, length: u64) -> Self {
        self.file_length = Some(length);
        self
    }

    /// Maximum serialized header size.
    pub fn max_header_bytes(&self) -> usize {
        self.max_header_bytes
    }

    /// Maximum decoded bitset size.
    pub fn max_bitset_bytes(&self) -> usize {
        self.max_bitset_bytes
    }

    /// Known file length, if provided.
    pub fn file_length(&self) -> Option<u64> {
        self.file_length
    }
}

// Charge primitive reads and skipped payload bytes across *all* prefix attempts.
// Collection traversal must consume input, and the shared protocol enforces a
// nesting-depth bound. This conservative budget also limits repeated parsing.
const HEADER_WORK_LIMIT: usize = 262_144;

struct HeaderProtocol<'a, 'b> {
    inner: ThriftSliceInputProtocol<'a>,
    remaining_work: &'b mut usize,
    exhausted: bool,
}

impl HeaderProtocol<'_, '_> {
    fn charge(&mut self, work: usize) -> ThriftProtocolResult<()> {
        let Some(remaining) = self.remaining_work.checked_sub(work.max(1)) else {
            self.exhausted = true;
            return Err(ThriftProtocolError::IntegerOverflow);
        };
        *self.remaining_work = remaining;
        Ok(())
    }
}

impl<'a> ThriftCompactInputProtocol<'a> for HeaderProtocol<'a, '_> {
    fn read_byte(&mut self) -> ThriftProtocolResult<u8> {
        self.charge(1)?;
        self.inner.read_byte()
    }

    fn read_bytes(&mut self) -> ThriftProtocolResult<&'a [u8]> {
        let len = usize::try_from(self.read_vlq()?)?;
        self.charge(len)?;
        let bytes = self
            .inner
            .as_slice()
            .get(..len)
            .ok_or(ThriftProtocolError::Eof)?;
        self.inner.skip_bytes(len)?;
        Ok(bytes)
    }

    fn read_bytes_owned(&mut self) -> ThriftProtocolResult<Vec<u8>> {
        Ok(self.read_bytes()?.to_vec())
    }

    fn skip_bytes(&mut self, n: usize) -> ThriftProtocolResult<()> {
        self.charge(n)?;
        self.inner.skip_bytes(n)
    }

    fn read_double(&mut self) -> ThriftProtocolResult<f64> {
        self.charge(8)?;
        self.inner.read_double()
    }

    fn remaining_bytes(&self) -> Option<usize> {
        self.inner.remaining_bytes()
    }
}

// Only a protocol EOF is incomplete input; malformed unions/fields and work
// exhaustion must not trigger another request.
fn parse_header(
    bytes: &[u8],
    remaining_work: &mut usize,
) -> Result<Option<(BloomFilterHeader, u64)>> {
    let mut protocol = HeaderProtocol {
        inner: ThriftSliceInputProtocol::new(bytes),
        remaining_work,
        exhausted: false,
    };
    let result = BloomFilterHeader::read_thrift(&mut protocol);
    if protocol.exhausted {
        return Err(general_err!("Bloom filter header work limit exceeded"));
    }
    match result {
        Ok(header) => Ok(Some((
            header,
            (bytes.len() - protocol.inner.as_slice().len()) as u64,
        ))),
        Err(ParquetError::EOF(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

enum ReadStage {
    Complete,
    Header,
    Bitset(BloomFilterReadInfo),
}

pub(super) struct BloomFilterReadPlan {
    offset: u64,
    total_len: Option<usize>,
    options: BloomFilterReadOptions,
    range: Range<u64>,
    stage: ReadStage,
    remaining_work: usize,
}

impl BloomFilterReadPlan {
    pub(super) fn new(
        column: &ColumnChunkMetaData,
        options: &BloomFilterReadOptions,
    ) -> Result<Option<Self>> {
        let Some(offset) = column.bloom_filter_offset() else {
            return Ok(None);
        };
        let offset =
            u64::try_from(offset).map_err(|_| general_err!("Bloom filter offset is invalid"))?;
        let total_len = column
            .bloom_filter_length()
            .map(usize::try_from)
            .transpose()
            .map_err(|_| general_err!("Bloom filter length is invalid"))?;
        if options.max_header_bytes == 0 || options.max_bitset_bytes < BITSET_MIN_LENGTH {
            return Err(general_err!("Bloom filter read limits are too small"));
        }
        let max_total = options
            .max_header_bytes
            .checked_add(options.max_bitset_bytes)
            .ok_or_else(|| general_err!("Bloom filter read limit overflow"))?;
        if let Some(total) = total_len {
            if total == 0 {
                return Err(general_err!("Bloom filter length is invalid"));
            }
            // Validate the entire advertised range even when discovering its header first.
            checked_range(offset, total, options.file_length)?;
            if total > max_total {
                return Err(general_err!(
                    "Bloom filter advertised length exceeds read limits"
                ));
            }
        }
        // Keep the common small-filter one-request path. Larger filters discover
        // the header first so a bogus bitset size cannot cause a large fetch.
        let complete = total_len.is_some_and(|total| total <= options.max_header_bytes);
        let stage = if complete {
            ReadStage::Complete
        } else {
            ReadStage::Header
        };
        let mut plan = Self {
            offset,
            total_len,
            options: options.clone(),
            range: offset..offset,
            stage,
            remaining_work: HEADER_WORK_LIMIT,
        };
        let len = if complete {
            total_len.unwrap()
        } else {
            plan.prefix_length(SBBF_HEADER_SIZE_ESTIMATE)?
        };
        plan.range = checked_range(offset, len, options.file_length)?;
        Ok(Some(plan))
    }

    fn prefix_length(&self, desired: usize) -> Result<usize> {
        let mut len = desired.min(self.options.max_header_bytes);
        if let Some(total) = self.total_len {
            len = len.min(total);
        }
        if let Some(file_len) = self.options.file_length {
            let remaining = file_len
                .checked_sub(self.offset)
                .ok_or_else(|| eof_err!("Bloom filter offset exceeds file length"))?;
            len = len.min(usize::try_from(remaining).unwrap_or(usize::MAX));
        }
        if len == 0 {
            return Err(eof_err!("Bloom filter header is truncated"));
        }
        Ok(len)
    }

    pub(super) fn range(&self) -> Range<u64> {
        self.range.clone()
    }

    pub(super) fn advance(&mut self, bytes: Bytes) -> Result<Option<Sbbf>> {
        let requested = usize::try_from(self.range.end - self.range.start)?;
        if bytes.len() != requested {
            return Err(general_err!(
                "Bloom filter read length mismatch: expected {requested} bytes, got {}",
                bytes.len()
            ));
        }
        if let ReadStage::Bitset(info) = &self.stage {
            return decode_bloom_filter_bitset(info, &bytes).map(Some);
        }
        let prefix = &bytes[..bytes.len().min(self.options.max_header_bytes)];
        let Some((header, header_len)) = parse_header(prefix, &mut self.remaining_work)? else {
            if matches!(self.stage, ReadStage::Complete) {
                return Err(eof_err!("Bloom filter header is truncated"));
            }
            let next_len = self.prefix_length(requested.saturating_mul(2))?;
            if next_len <= requested {
                return Err(general_err!(
                    "Bloom filter header byte limit or EOF reached"
                ));
            }
            self.range = checked_range(self.offset, next_len, self.options.file_length)?;
            return Ok(None);
        };
        let info = bloom_filter_read_info(self.offset, header, header_len, self.total_len)?;
        if info.bitset_len > self.options.max_bitset_bytes {
            return Err(general_err!("Bloom filter bitset exceeds read limit"));
        }
        if matches!(self.stage, ReadStage::Complete) {
            return decode_bloom_filter_bitset(&info, &bytes[info.header_len..]).map(Some);
        }
        self.range = checked_range(
            info.bitset_offset,
            info.bitset_len,
            self.options.file_length,
        )?;
        self.stage = ReadStage::Bitset(info);
        Ok(None)
    }
}

fn checked_range(offset: u64, len: usize, file_length: Option<u64>) -> Result<Range<u64>> {
    let end = offset
        .checked_add(u64::try_from(len)?)
        .ok_or_else(|| general_err!("Bloom filter read range overflow"))?;
    if file_length.is_some_and(|length| end > length) {
        return Err(eof_err!("Bloom filter read range exceeds file length"));
    }
    Ok(offset..end)
}
