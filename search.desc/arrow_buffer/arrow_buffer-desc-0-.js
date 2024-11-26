searchState.loadedDescShard("arrow_buffer", 0, "Low-level buffer abstractions for Apache Arrow Rust\nTrait expressing a Rust type that has the same in-memory …\nValue of an IntervalDayTime array\nValue of an IntervalMonthDayNano array\nAllows conversion from supported Arrow types to a byte …\nDefines the low-level <code>Allocation</code> API for shared memory …\nConvert to usize according to the <code>as</code> operator\nTypes for iterating over bitmasks in 64-bit chunks\nTypes for iterating over packed bitmasks\nUtils for working with packed bit masks\nUtils for working with bits\nTypes of shared memory region\nBuffer builders\nThis module contains an implementation of a contiguous …\nNumber of days\nNumber of days\nConvert native integer type from usize\nA signed 256-bit integer\nNumber of milliseconds\nNumber of months\nNumber of nanoseconds\nConverts this instance into a byte slice\nConvert native type to i64.\nConvert native type to isize.\nConvert native type to usize.\nConvert from usize according to the <code>as</code> operator\nCache and allocation multiple alignment size\nThe owner of an allocation. The trait implementation is …\nAn allocation from an external source like the FFI …\nMode of deallocating memory regions\nAn allocation using <code>std::alloc</code>\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCache and allocation multiple alignment size\nDerives <code>std::ops::$t</code> for <code>$ty</code> calling <code>$wrapping</code> or <code>$checked</code> …\nError returned by i256::DivRem\nDivision by zero\nDivision overflow\nThe maximum value that can be represented by this integer …\nThe minimum value that can be represented by this integer …\nThe multiplicative inverse for this integer type, i.e. <code>-1</code>.\nThe multiplicative identity for this integer type, i.e. <code>1</code>.\nAn opaque error similar to <code>std::num::ParseIntError</code>\nThe additive identity for this integer type, i.e. <code>0</code>.\nInterpret this <code>i256</code> as 4 <code>u64</code> digits, least significant …\nWraps this <code>i256</code> into an <code>i128</code>\nComputes the absolute value of this i256 returning <code>None</code> if …\nPerforms checked addition\nPerforms checked division\nPerforms checked multiplication\nNegates this i256 returning <code>None</code> if <code>Self == Self::MIN</code>\nPerforms checked exponentiation\nPerforms checked remainder\nPerforms checked subtraction\nN-digit division\nDivision operation, returns (quotient, remainder). This …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCreate an integer value from its representation as a byte …\nCreate an i256 from the provided <code>BigInt</code> returning a bool …\nInterpret 4 <code>u64</code> digits, least significant first, as a <code>i256</code>\nCreate an optional i256 from the provided <code>f64</code>. Returning …\nCreate an <code>i256</code> value from a 128-bit value.\nCreate an integer value from its representation as a byte …\nCreate an i256 from the provided low u128 and high i128\nCreate an integer value from its representation as string.\nA signed 256-bit integer\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns <code>true</code> if this <code>i256</code> is negative\nReturns <code>true</code> if this <code>i256</code> is positive\nPerforms an unsigned multiplication of <code>a * b</code> returning a …\nParse <code>s</code> with any sign and leading 0s removed\nReturns a number <code>i256</code> representing sign of this <code>i256</code>.\nTemporary workaround due to lack of stable const array …\nReturn the memory representation of this integer as a byte …\nConverts this <code>i256</code> into an <code>i128</code> returning <code>None</code> if this …\nReturn the memory representation of this integer as a byte …\nReturns this <code>i256</code> as a low u128 and high i128\nComputes the absolute value of this i256\nPerforms wrapping addition\nPerforms wrapping division\nPerforms wrapping multiplication\nNegates this i256\nPerforms wrapping exponentiation\nPerforms wrapping remainder\nPerforms wrapping subtraction\nAn array of N + 1 elements\nPerform <code>a += b</code>\nConverts an overflowing binary operation on scalars to one …\nReturn the least number of bits needed to represent the …\nUnsigned, little-endian, n-digit division with remainder\nUse Knuth Algorithm D to compute <code>numerator / divisor</code> …\nDivision of numerator by a u64 divisor\nPerform narrowing division of a u128 by a u64 divisor, …\nReturns the argument unchanged.\nWidening multiplication of an N-digit array with a u64\nWidening left shift of an N-digit array by at most 63 bits\nNarrowing right shift of an (N+1)-digit array by at most …\nCalls <code>U::from(self)</code>.\nLeft shift of an N-digit array by at most 63 bits\nPerform <code>a -= b</code>\nIterator over chunks of 64 bits represented as an u64\nIterates over an arbitrarily aligned byte buffer\nIterates over an arbitrarily aligned byte buffer\nIterator over an <code>UnalignedBitChunk</code>\noffset inside a byte, guaranteed to be between 0 and 7 …\nnumber of complete u64 chunks\nnumber of remaining bits, guaranteed to be between 0 and …\nAn iterator of <code>usize</code> whose index in a provided bitmask is …\nIterator over the bits within a packed bitmask\nIterator of contiguous ranges of set bits within a …\nCalls the provided closure for each index in the provided …\nUtil function to set bits in a slice of bytes.\nReturns the ceil of <code>value</code>/<code>divisor</code>\nReturns whether bit at position <code>i</code> in <code>data</code> is set or not\nReturns whether bit at position <code>i</code> in <code>data</code> is set or not.\nReturns the nearest number that is <code>&gt;=</code> than <code>num</code> and is a …\nReturns the nearest multiple of <code>factor</code> that is <code>&gt;=</code> than <code>num</code>…\nSets bit at position <code>i</code> for <code>data</code> to 1\nSets bit at position <code>i</code> for <code>data</code>\nSets bit at position <code>i</code> for <code>data</code> to 0\nSets bit at position <code>i</code> for <code>data</code> to 0\nA slice-able <code>Buffer</code> containing bit-packed booleans\nBuffer represents a contiguous memory region that can be …\nA <code>MutableBuffer</code> is Arrow’s interface to build a <code>Buffer</code> …\nA <code>BooleanBuffer</code> used to encode validity for arrow arrays\nA non-empty buffer of monotonically increasing, positive …\nA slice-able buffer of monotonically increasing, positive …\nA strongly-typed <code>Buffer</code> supporting zero-copy cloning and …\nApply a bitwise operation <code>op</code> to two inputs and return the …\nApply a bitwise operation <code>op</code> to four inputs and return the …\nApply a bitwise operation <code>op</code> to one input and return the …\nUnderlying data buffer\nApply a bitwise and to two inputs and return the result as …\nApply a bitwise and_not to two inputs and return the …\nApply a bitwise or to two inputs and return the result as …\nApply a bitwise xor to two inputs and return the result as …\nApply a bitwise not to one input and return the result as …\nthe internal byte buffer.\nByte length of the buffer.\nPointer into <code>data</code> valid\nA slice-able <code>Buffer</code> containing bit-packed booleans\nReturns a <code>BitChunks</code> instance which can be used to iterate …\nInvokes <code>f</code> with indexes <code>0..len</code> collecting the boolean …\nReturns the number of set bits in this buffer\nReturns the argument unchanged.\nReturns the inner <code>Buffer</code>\nCalls <code>U::from(self)</code>.\nReturns the inner <code>Buffer</code>, consuming self\nReturns true if this <code>BooleanBuffer</code> is empty\nReturns an iterator over the bits in this <code>BooleanBuffer</code>\nReturns the length of this <code>BooleanBuffer</code> in bits\nCreate a new <code>BooleanBuffer</code> from a <code>Buffer</code>, an <code>offset</code> and …\nCreate a new <code>BooleanBuffer</code> of <code>length</code> where all values are …\nCreate a new <code>BooleanBuffer</code> of <code>length</code> where all values are …\nReturns the offset of this <code>BooleanBuffer</code> in bits\nReturns true if this <code>BooleanBuffer</code> is equal to <code>other</code>, …\nReturns an iterator over the set bit positions in this …\nReturns a <code>BitSliceIterator</code> yielding contiguous ranges of …\nSlices this <code>BooleanBuffer</code> by the provided <code>offset</code> and <code>length</code>\nReturns a <code>Buffer</code> containing the sliced contents of this …\nReturns the boolean value at index <code>i</code>.\nReturns the boolean value at index <code>i</code>.\nReturns the packed values of this <code>BooleanBuffer</code> not …\nBuffer represents a contiguous memory region that can be …\nIncreases the offset of this buffer by <code>offset</code>\nReturns a pointer to the start of this buffer.\nReturns the byte slice stored in this buffer\nReturns a <code>BitChunks</code> instance which can be used to iterate …\nReturns a slice of this buffer starting at a certain bit …\nAuxiliary method to create a new Buffer\nReturns the capacity of this buffer. For externally owned …\nReturns the number of 1-bits in this buffer, starting from …\nthe internal byte buffer.\nReturns the pointer to the start of the buffer without the …\nReturns the argument unchanged.\nAuxiliary method to create a new Buffer\nCreates a buffer from an existing memory region. Ownership …\nCreates a buffer from an existing aligned memory region …\nInitializes a Buffer from a slice of items.\nCreates a <code>Buffer</code> from an <code>Iterator</code> with a trusted (upper) …\nCreate a <code>Buffer</code> from the provided <code>Vec</code> without copying\nCalls <code>U::from(self)</code>.\nReturns <code>MutableBuffer</code> for mutating the buffer if this …\nReturns <code>Vec</code> for mutating the buffer\nReturns whether the buffer is empty.\nReturns the number of bytes in the buffer\nByte length of the buffer.\nPointer into <code>data</code> valid\nReturns true if this <code>Buffer</code> is equal to <code>other</code>, using …\nReturns the offset, in bytes, of <code>Self::ptr</code> to <code>Self::data</code>\nReturns a new Buffer that is a slice of this buffer …\nReturns a new Buffer that is a slice of this buffer …\nCreates a <code>Buffer</code> from an <code>Iterator</code> with a trusted (upper) …\nView buffer as a slice of a specific type.\nA <code>MutableBuffer</code> is Arrow’s interface to build a <code>Buffer</code> …\nReturns a mutable raw pointer to this buffer’s internal …\nReturns a raw pointer to this buffer’s internal memory …\nReturns the data stored in this buffer as a slice.\nReturns the data stored in this buffer as a mutable slice.\nReturns the total capacity in this buffer. The invariant …\nClear all existing data from this buffer.\nInvokes <code>f</code> with values <code>0..len</code> collecting the boolean …\nExtends this buffer from a slice of items that can be …\nExtends the buffer by <code>additional</code> bytes equal to <code>0u8</code>, …\nReturns the argument unchanged.\nReturns the argument unchanged.\nAllocates a new MutableBuffer from given <code>Bytes</code>.\nAllocates a new MutableBuffer with <code>len</code> and capacity to be …\nCreates a <code>MutableBuffer</code> from an <code>Iterator</code> with a trusted …\nCreates a <code>MutableBuffer</code> from a boolean <code>Iterator</code> with a …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns whether this buffer is empty or not.\nReturns the length (the number of bytes written) in this …\nAllocate a new MutableBuffer with initial capacity to be …\ncreates a new MutableBuffer with capacity and length …\nExtends the buffer with a new item, increasing its …\nExtends the buffer with a new item, without checking for …\nEnsures that this buffer has at least <code>self.len + additional</code>…\nResizes the buffer, either truncating its contents (with …\nSafety\nEnsure that <code>count</code> bytes from <code>start</code> contain zero bits\nShrinks the capacity of the buffer as much as possible. …\nTruncates this buffer to <code>len</code> bytes\nCreates a <code>MutableBuffer</code> from an <code>Iterator</code> with a trusted …\nView buffer as a immutable slice of a specific type.\nView this buffer as a mutable slice of a specific type.\nSet the bits in the range of <code>[0, end)</code> to 0 (if <code>val</code> is …\nAllocate a new MutableBuffer with initial capacity to be …\nA <code>BooleanBuffer</code> used to encode validity for arrow arrays\nReturns the underlying <code>Buffer</code>\nReturns true if all nulls in <code>other</code> also exist in self\nReturns a new <code>NullBuffer</code> where each bit in the current …\nReturns the argument unchanged.\nReturns the inner <code>BooleanBuffer</code>\nCalls <code>U::from(self)</code>.\nReturns the inner <code>BooleanBuffer</code>\nReturns true if this <code>NullBuffer</code> is empty\nReturns <code>true</code> if the value at <code>idx</code> is null\nReturns <code>true</code> if the value at <code>idx</code> is not null\nReturns an iterator over the bits in this <code>NullBuffer</code>\nReturns the length of this <code>NullBuffer</code>\nCreate a new <code>NullBuffer</code> computing the null count\nCreate a new <code>NullBuffer</code> of length <code>len</code> where all values are …\nCreate a new <code>NullBuffer</code> with the provided <code>buffer</code> and …\nCreate a new <code>NullBuffer</code> of length <code>len</code> where all values are …\nReturns the null count for this <code>NullBuffer</code>\nReturns the offset of this <code>NullBuffer</code> in bits\nSlices this <code>NullBuffer</code> by the provided <code>offset</code> and <code>length</code>\nCalls the provided closure for each index in this null …\nComputes the union of the nulls in two optional <code>NullBuffer</code>\nReturns a <code>BitIndexIterator</code> over the valid indices in this …\nReturns a <code>BitSliceIterator</code> yielding contiguous ranges of …\nReturns the packed validity of this <code>NullBuffer</code> not …\nA non-empty buffer of monotonically increasing, positive …\nReturns the argument unchanged.\nCreate a new <code>OffsetBuffer</code> from the iterator of slice …\nReturns the inner <code>ScalarBuffer</code>\nCalls <code>U::from(self)</code>.\nReturns the inner <code>ScalarBuffer</code>, consuming self\nCreate a new <code>OffsetBuffer</code> from the provided <code>ScalarBuffer</code>\nCreate a new <code>OffsetBuffer</code> containing a single 0 value\nCreate a new <code>OffsetBuffer</code> from the provided <code>ScalarBuffer</code>\nCreate a new <code>OffsetBuffer</code> containing <code>len + 1</code> <code>0</code> values\nReturns true if this <code>OffsetBuffer</code> is equal to <code>other</code>, using …\nReturns a zero-copy slice of this buffer with length <code>len</code> …\nApply a bitwise operation <code>op</code> to two inputs and return the …\nApply a bitwise operation <code>op</code> to four inputs and return the …\nApply a bitwise operation <code>op</code> to one input and return the …\nApply a bitwise and to two inputs and return the result as …\nApply a bitwise and_not to two inputs and return the …\nApply a bitwise or to two inputs and return the result as …\nApply a bitwise xor to two inputs and return the result as …\nApply a bitwise not to one input and return the result as …\nA slice-able buffer of monotonically increasing, positive …\nReturns the argument unchanged.\nReturns the physical index at which the logical array ends\nPerforms a binary search to find the physical index for …\nReturns the physical index at which the logical array …\nReturns the inner <code>ScalarBuffer</code>\nCalls <code>U::from(self)</code>.\nReturns the inner <code>ScalarBuffer</code>, consuming self\nReturns true if this buffer is empty\nReturns the logical length of the run-ends stored by this …\nReturns the maximum run-end encoded in the underlying …\nCreate a new <code>RunEndBuffer</code> from a <code>ScalarBuffer</code>, an <code>offset</code> …\nCreate a new <code>RunEndBuffer</code> from an <code>ScalarBuffer</code>, an <code>offset</code> …\nReturns the logical offset into the run-ends stored by …\nSlices this <code>RunEndBuffer</code> by the provided <code>offset</code> and <code>length</code>\nReturns the values of this <code>RunEndBuffer</code> not including any …\nA strongly-typed <code>Buffer</code> supporting zero-copy cloning and …\nUnderlying data buffer\nReturns the argument unchanged.\nReturns the inner <code>Buffer</code>\nCalls <code>U::from(self)</code>.\nReturns the inner <code>Buffer</code>, consuming self\nCreate a new <code>ScalarBuffer</code> from a <code>Buffer</code>, and an <code>offset</code> and …\nReturns true if this <code>ScalarBuffer</code> is equal to <code>other</code>, using …\nReturns a zero-copy slice of this buffer with length <code>len</code> …\nBuilder for <code>BooleanBuffer</code>\nBuilder for creating a Buffer object.\nBuilder for creating the null bit buffer.\nBuilder of <code>OffsetBuffer</code>\nIncreases the number of elements in the internal buffer by …\nAppends a value of type <code>T</code> into the builder, growing the …\nAppends a value of type <code>T</code> into the builder N times, …\nAppends <code>n</code>, zero-initialized values\nAppends a slice of type <code>T</code>, growing the internal buffer as …\nSafety\nView the contents of this buffer as a slice\nView the contents of this buffer as a mutable slice\nReturns the actual capacity (number of elements) of the …\nResets this builder and returns an immutable Buffer.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nReturns whether the internal buffer is empty.\nReturns the current number of array elements in the …\nStore the length of the buffer before materializing.\nCreates a new builder with initial capacity for <em>at least</em> …\nCreates a new builder from a <code>MutableBuffer</code>\nReserves memory for <em>at least</em> <code>n</code> more elements of type <code>T</code>.\nShorten this BufferBuilder to <code>len</code> items\nBuilder for <code>BooleanBuffer</code>\nAdvances the buffer by <code>additional</code> bits\nAppends a boolean <code>v</code> into the buffer\nAppend <code>BooleanBuffer</code> to this <code>BooleanBufferBuilder</code>\nAppends n <code>additional</code> bits of value <code>v</code> into the buffer\nAppend <code>range</code> bits from <code>to_set</code>\nAppends a slice of booleans into the buffer\nReturns the packed bits\nReturns the packed bits\nReturns the capacity of the buffer\nCreates a <code>BooleanBuffer</code>\nBuilds the BooleanBuffer without resetting the builder.\nReturns the argument unchanged.\nGets a bit in the buffer at <code>index</code>\nCalls <code>U::from(self)</code>.\nReturns true if empty\nReturns the length of the buffer\nCreates a new <code>BooleanBufferBuilder</code>\nCreates a new <code>BooleanBufferBuilder</code> from <code>MutableBuffer</code> of …\nReserve space to at least <code>additional</code> new bits. Capacity …\nResizes the buffer, either truncating its contents (with …\nSets a bit in the buffer at <code>index</code>\nTruncates the builder to the given length\nBuilder for creating the null bit buffer.\nReturn the allocated size of this builder, in bytes, …\nAppends a boolean value into the builder.\nAppends <code>n</code> <code>true</code>s into the builder to indicate that these <code>n</code> …\nAppends <code>n</code> <code>false</code>s into the builder to indicate that these <code>n</code> …\nAppends a <code>true</code> into the builder to indicate that this item …\nAppends a <code>false</code> into the builder to indicate that this …\nAppends a boolean slice into the builder to indicate the …\nReturns the inner bitmap builder as slice\nReturn a mutable reference to the inner bitmap slice.\nBuilds the null buffer and resets the builder. Returns <code>None</code>…\nBuilds the NullBuffer without resetting the builder.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCheck if the builder is empty.\nReturn the number of bits in the buffer.\nStore the length of the buffer before materializing.\nCreates a new empty builder. <code>capacity</code> is the number of …\nCreates a new builder from a <code>MutableBuffer</code>.\nCreates a new builder with given length.\nBuilder of <code>OffsetBuffer</code>\nTakes the builder itself and returns an <code>OffsetBuffer</code>\nBuilds the OffsetBuffer without resetting the builder.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCreate a new builder with space for <code>capacity + 1</code> offsets\nPush a slice of <code>length</code> bytes\nReserve space for at least <code>additional</code> further offsets\nA continuous, fixed-size, immutable memory region that …\nhow to deallocate this region\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nThe number of bytes visible to this region. This is always …\nTakes ownership of an allocated memory region,\nThe raw pointer to be beginning of the region\nValue of an IntervalDayTime array\nValue of an IntervalMonthDayNano array\nThe maximum value that can be represented\nThe maximum value that can be represented\nThe minimum value that can be represented\nThe minimum value that can be represented\nThe multiplicative inverse, i.e. <code>-1</code>.\nThe multiplicative inverse, i.e. <code>-1</code>.\nThe multiplicative identity, i.e. <code>1</code>.\nThe multiplicative identity, i.e. <code>1</code>.\nThe additive identity i.e. <code>0</code>.\nThe additive identity i.e. <code>0</code>.\nComputes the absolute value\nComputes the absolute value\nPerforms checked addition\nPerforms checked addition\nPerforms checked division\nPerforms checked division\nPerforms checked multiplication\nPerforms checked multiplication\nNegates the value\nNegates the value\nPerforms checked exponentiation\nPerforms checked exponentiation\nPerforms checked remainder\nPerforms checked remainder\nPerforms checked subtraction\nPerforms checked subtraction\nNumber of days\nNumber of days\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nNumber of milliseconds\nNumber of months\nNumber of nanoseconds\nCreate a new <code>IntervalMonthDayNano</code>\nCreate a new <code>IntervalDayTime</code>\nComputes the absolute value\nComputes the absolute value\nPerforms wrapping addition\nPerforms wrapping addition\nPerforms wrapping division\nPerforms wrapping division\nPerforms wrapping multiplication\nPerforms wrapping multiplication\nNegates the value\nNegates the value\nPerforms wrapping exponentiation\nPerforms wrapping exponentiation\nPerforms wrapping remainder\nPerforms wrapping remainder\nPerforms wrapping subtraction\nPerforms wrapping subtraction\nTrait expressing a Rust type that has the same in-memory …\nAllows conversion from supported Arrow types to a byte …\nConvert to usize according to the <code>as</code> operator\nConvert native integer type from usize\nReturns the byte width of this native type.\nReturns the byte width of this native type.\nConverts this instance into a byte slice\nConvert native type to i64.\nConvert native type to isize.\nConvert native type to usize.\nConvert from usize according to the <code>as</code> operator\nTypes for iterating over bitmasks in 64-bit chunks\nTypes for iterating over packed bitmasks\nUtils for working with packed bit masks\nUtils for working with bits\nIterator over chunks of 64 bits represented as an u64\nIterates over an arbitrarily aligned byte buffer\nIterates over an arbitrarily aligned byte buffer\nIterator over an <code>UnalignedBitChunk</code>\noffset inside a byte, guaranteed to be between 0 and 7 …\nReturns the number of chunks\nnumber of complete u64 chunks\nReturns reference to the chunks\nCounts the number of ones\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns an iterator over the chunks\nReturns an iterator over chunks of 64 bits represented as …\nReturns an iterator over chunks of 64 bits, with the …\nReturns the number of leading padding bits\nCreate a from a byte array, and and an offset and length …\nCreate a new <code>BitChunks</code> from a byte array, and an offset …\nReturns the prefix, if any\nReturns the bitmask of remaining bits\nReturns the number of remaining bits, guaranteed to be …\nnumber of remaining bits, guaranteed to be between 0 and …\nReturns the suffix, if any\nReturns the number of trailing padding bits\nAn iterator of <code>usize</code> whose index in a provided bitmask is …\nIterator over the bits within a packed bitmask\nIterator of contiguous ranges of set bits within a …\nReturns <code>Some((chunk_offset, bit_offset))</code> for the next …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCreate a new <code>BitIterator</code> from the provided <code>buffer</code>, and …\nCreate a new <code>BitSliceIterator</code> from the provided <code>buffer</code>, …\nCreate a new <code>BitIndexIterator</code> from the provide <code>buffer</code>, and …\nCalls the provided closure for each index in the provided …\nSimilar to <code>write_u64_bytes</code>, but this method ORs the offset …\nSafety\nUtil function to set bits in a slice of bytes.\nSimilar to <code>set_bits</code> but sets only upto 64 bits, actual …\nSafety\nReturns the ceil of <code>value</code>/<code>divisor</code>\nReturns whether bit at position <code>i</code> in <code>data</code> is set or not\nReturns whether bit at position <code>i</code> in <code>data</code> is set or not.\nReturns the nearest number that is <code>&gt;=</code> than <code>num</code> and is a …\nReturns the nearest multiple of <code>factor</code> that is <code>&gt;=</code> than <code>num</code>…\nSets bit at position <code>i</code> for <code>data</code> to 1\nSets bit at position <code>i</code> for <code>data</code>\nSets bit at position <code>i</code> for <code>data</code> to 0\nSets bit at position <code>i</code> for <code>data</code> to 0")