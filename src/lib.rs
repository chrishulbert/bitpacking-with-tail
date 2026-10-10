mod scalar;

use bitpacking::{BitPacker, BitPacker8x};

pub fn num_bits(data: &[u32]) -> u8 {
    let combined = data.iter().fold(0u32, |acc, &v| acc | v);
    ((32 - combined.leading_zeros()) as u8).max(1)
}

// This packs in a hybrid fashion: whatever is possible is packed with bitpacking,
// the remaining tail is packed in a scalar fashion.
// Note: The packer should keep track of the number of elements, because if num_bits is small, it can be vague
// to get the number of elements from a packed buffer length.
pub fn pack(data: &[u32], num_bits: u8) -> Vec<u8> {
    assert!((1..=32).contains(&num_bits));

    let total_bits = data.len() * num_bits as usize;
    let bitpacker = BitPacker8x::new();
    let chunks = data.chunks_exact(BitPacker8x::BLOCK_LEN);
    let remainder = chunks.remainder();
    let block_bytes = BitPacker8x::BLOCK_LEN * num_bits as usize / 8;
    let mut out: Vec<u8> = Vec::with_capacity(total_bits.div_ceil(8));
    out.resize(chunks.len() * block_bytes, 0); // Fill it with space for the full blocks.
    for (chunk, dst) in chunks.zip(out.chunks_exact_mut(block_bytes)) {
        bitpacker.compress(chunk, dst, num_bits);
    }
    if !remainder.is_empty() {
        scalar::pack(remainder, num_bits, &mut out);
    }
    out
}

// This unpacks the prior output of the pack function.
// Bitpacker is used for initial multiples of the block length,
// then a scaler [un]packer is used for the remaining tail.
// Count is required because if num_bits is small, it can't be determined accurately
// by dividing data length.
pub fn unpack(data: &[u8], num_bits: u8, count: usize) -> Vec<u32> {
    assert!((1..=32).contains(&num_bits));
    assert!(
        data.len() >= (count * num_bits as usize).div_ceil(8),
        "input too short for {count} elements of {num_bits} bits"
    );

    let block_len = BitPacker8x::BLOCK_LEN;
    let block_bytes = block_len * num_bits as usize / 8;
    let full_blocks = count / block_len;
    let tail_count = count % block_len;

    let mut out: Vec<u32> = Vec::with_capacity(count);
    out.resize(full_blocks * block_len, 0); // Space for the full blocks.

    let bitpacker = BitPacker8x::new();
    for (src, dst) in data
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(block_len))
    {
        bitpacker.decompress(src, dst, num_bits);
    }

    if tail_count > 0 {
        scalar::unpack(&data[full_blocks * block_bytes..], num_bits, tail_count, &mut out);
    }
    debug_assert_eq!(out.len(), count);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_LEN: usize = BitPacker8x::BLOCK_LEN;

    /// Lengths around the block boundaries: tail-only, exact blocks, blocks + tail.
    const LENS: [usize; 11] = [0, 1, 7, 255, 256, 257, 511, 512, 513, 1000, 1024];

    fn mask_for(num_bits: u8) -> u32 {
        u32::MAX >> (32 - num_bits)
    }

    fn block_bytes(num_bits: u8) -> usize {
        BLOCK_LEN * num_bits as usize / 8
    }

    /// Deterministic pseudo-random values that fit in `num_bits`.
    fn sample(num_bits: u8, len: usize) -> Vec<u32> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ (num_bits as u64);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as u32) & mask_for(num_bits)
            })
            .collect()
    }

    // ---------- round trips ----------

    #[test]
    fn roundtrip_all_widths_and_lengths() {
        for num_bits in 1..=32u8 {
            for len in LENS {
                let data = sample(num_bits, len);
                let packed = pack(&data, num_bits);
                assert_eq!(
                    unpack(&packed, num_bits, len),
                    data,
                    "num_bits={num_bits} len={len}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_max_values() {
        for num_bits in 1..=32u8 {
            let data = vec![mask_for(num_bits); BLOCK_LEN * 2 + 5];
            let packed = pack(&data, num_bits);
            assert_eq!(unpack(&packed, num_bits, data.len()), data, "num_bits={num_bits}");
        }
    }

    #[test]
    fn roundtrip_all_zeros() {
        for num_bits in 1..=32u8 {
            let data = vec![0u32; BLOCK_LEN + 3];
            let packed = pack(&data, num_bits);
            assert!(packed.iter().all(|&b| b == 0), "num_bits={num_bits}");
            assert_eq!(unpack(&packed, num_bits, data.len()), data);
        }
    }

    #[test]
    fn roundtrip_alternating_bit_patterns() {
        for num_bits in 1..=32u8 {
            let m = mask_for(num_bits);
            let data: Vec<u32> = (0..BLOCK_LEN * 2 + 17)
                .map(|i| if i % 2 == 0 { m } else { 0 })
                .collect();
            let packed = pack(&data, num_bits);
            assert_eq!(unpack(&packed, num_bits, data.len()), data, "num_bits={num_bits}");
        }
    }

    // ---------- pack: sizes ----------

    #[test]
    fn packed_length_is_ceil_of_total_bits() {
        for num_bits in 1..=32u8 {
            for len in LENS {
                let packed = pack(&sample(num_bits, len), num_bits);
                assert_eq!(
                    packed.len(),
                    (len * num_bits as usize).div_ceil(8),
                    "num_bits={num_bits} len={len}"
                );
            }
        }
    }

    #[test]
    fn empty_input() {
        for num_bits in 1..=32u8 {
            assert!(pack(&[], num_bits).is_empty());
            assert!(unpack(&[], num_bits, 0).is_empty());
        }
    }

    // ---------- pack: each half in isolation ----------

    #[test]
    fn short_input_matches_scalar_exactly() {
        for num_bits in 1..=32u8 {
            for len in [1usize, 2, 9, 100, BLOCK_LEN - 1] {
                let data = sample(num_bits, len);
                let mut expected = Vec::new();
                scalar::pack(&data, num_bits, &mut expected);
                assert_eq!(pack(&data, num_bits), expected, "num_bits={num_bits} len={len}");
            }
        }
    }

    #[test]
    fn exact_blocks_match_crate_output() {
        let bitpacker = BitPacker8x::new();
        for num_bits in 1..=32u8 {
            let data = sample(num_bits, BLOCK_LEN * 3);
            let bb = block_bytes(num_bits);

            let mut expected = vec![0u8; bb * 3];
            for (chunk, dst) in data.chunks_exact(BLOCK_LEN).zip(expected.chunks_exact_mut(bb)) {
                bitpacker.compress(chunk, dst, num_bits);
            }
            assert_eq!(pack(&data, num_bits), expected, "num_bits={num_bits}");
        }
    }

    #[test]
    fn blocks_then_tail_split_is_correct() {
        for num_bits in 1..=32u8 {
            for tail in [1usize, 3, 8, 100, BLOCK_LEN - 1] {
                let data = sample(num_bits, BLOCK_LEN * 2 + tail);
                let packed = pack(&data, num_bits);

                let split = 2 * block_bytes(num_bits);
                let head = pack(&data[..BLOCK_LEN * 2], num_bits);
                let mut tail_expected = Vec::new();
                scalar::pack(&data[BLOCK_LEN * 2..], num_bits, &mut tail_expected);

                assert_eq!(&packed[..split], &head[..], "head num_bits={num_bits} tail={tail}");
                assert_eq!(&packed[split..], &tail_expected[..], "tail num_bits={num_bits} tail={tail}");
            }
        }
    }

    #[test]
    fn blocks_are_independent_of_the_tail() {
        let num_bits = 11;
        let a = sample(num_bits, BLOCK_LEN + 10);
        let mut b = a.clone();
        for v in &mut b[BLOCK_LEN..] {
            *v ^= 0x7FF;
        }
        let bb = block_bytes(num_bits);
        let pa = pack(&a, num_bits);
        let pb = pack(&b, num_bits);
        assert_eq!(&pa[..bb], &pb[..bb]);
        assert_ne!(&pa[bb..], &pb[bb..]);
    }

    #[test]
    fn tail_only_known_layouts() {
        assert_eq!(pack(&[0x1, 0x2, 0x3], 4), vec![0x21, 0x03]);
        assert_eq!(pack(&[1, 2, 7], 3), vec![0xD1, 0x01]);
        assert_eq!(pack(&[0x1234_5678], 32), vec![0x78, 0x56, 0x34, 0x12]);
    }

    // ---------- unpack: each half in isolation ----------

    #[test]
    fn unpack_tail_only_known_bytes() {
        assert_eq!(unpack(&[0x21, 0x03], 4, 3), vec![0x1, 0x2, 0x3]);
        assert_eq!(unpack(&[0xBC, 0xFA, 0xDE], 12, 2), vec![0xABC, 0xDEF]);
        assert_eq!(unpack(&[0x78, 0x56, 0x34, 0x12], 32, 1), vec![0x1234_5678]);
    }

    #[test]
    fn unpack_exact_blocks_from_crate_output() {
        // Feed unpack bytes produced directly by the crate, independent of our pack.
        let bitpacker = BitPacker8x::new();
        for num_bits in 1..=32u8 {
            let data = sample(num_bits, BLOCK_LEN * 2);
            let bb = block_bytes(num_bits);
            let mut compressed = vec![0u8; bb * 2];
            for (chunk, dst) in data.chunks_exact(BLOCK_LEN).zip(compressed.chunks_exact_mut(bb)) {
                bitpacker.compress(chunk, dst, num_bits);
            }
            assert_eq!(unpack(&compressed, num_bits, data.len()), data, "num_bits={num_bits}");
        }
    }

    #[test]
    fn unpack_blocks_then_scalar_tail_from_independent_parts() {
        // Build the input from the two reference encoders, not from `pack`.
        let bitpacker = BitPacker8x::new();
        for num_bits in 1..=32u8 {
            let data = sample(num_bits, BLOCK_LEN + 37);
            let bb = block_bytes(num_bits);

            let mut bytes = vec![0u8; bb];
            bitpacker.compress(&data[..BLOCK_LEN], &mut bytes, num_bits);
            scalar::pack(&data[BLOCK_LEN..], num_bits, &mut bytes);

            assert_eq!(unpack(&bytes, num_bits, data.len()), data, "num_bits={num_bits}");
        }
    }

    // ---------- unpack: count and trailing bytes ----------

    #[test]
    fn unpack_ignores_trailing_bytes() {
        for num_bits in 1..=32u8 {
            for len in [5usize, BLOCK_LEN, BLOCK_LEN + 5] {
                let data = sample(num_bits, len);
                let mut packed = pack(&data, num_bits);
                packed.extend_from_slice(&[0xFF; 16]);
                assert_eq!(unpack(&packed, num_bits, len), data, "num_bits={num_bits} len={len}");
            }
        }
    }

    #[test]
    fn unpack_prefix_at_block_boundary() {
        // Only valid when `count` is a multiple of BLOCK_LEN: the unpacked prefix then
        // consists purely of whole blocks, which are independent of what follows.
        for num_bits in 1..=32u8 {
            let data = sample(num_bits, BLOCK_LEN * 3 + 9);
            let packed = pack(&data, num_bits);
            assert_eq!(
                unpack(&packed, num_bits, BLOCK_LEN * 2),
                data[..BLOCK_LEN * 2].to_vec(),
                "num_bits={num_bits}"
            );
        }
    }

    #[test]
    fn unpack_count_must_match_pack_count_off_block_boundary() {
        // Documents a format property: the tail is scalar-encoded, so unpacking a
        // different non-block-aligned count from a longer buffer does NOT yield a prefix,
        // because the bytes after the last full block are SIMD-encoded, not scalar.
        let num_bits = 9;
        let data = sample(num_bits, BLOCK_LEN * 2);
        let packed = pack(&data, num_bits);
        let wrong = unpack(&packed, num_bits, BLOCK_LEN + 10);
        assert_ne!(wrong[BLOCK_LEN..], data[BLOCK_LEN..BLOCK_LEN + 10]);
    }

    #[test]
    fn unpack_returns_exactly_count_elements() {
        for num_bits in 1..=32u8 {
            for len in LENS {
                let packed = pack(&sample(num_bits, len), num_bits);
                assert_eq!(unpack(&packed, num_bits, len).len(), len);
            }
        }
    }

    // ---------- panics ----------

    #[test]
    #[should_panic]
    fn pack_rejects_zero_bits() {
        pack(&[1], 0);
    }

    #[test]
    #[should_panic]
    fn pack_rejects_zero_bits_on_empty_input() {
        pack(&[], 0);
    }

    #[test]
    #[should_panic]
    fn pack_rejects_33_bits() {
        pack(&[1], 33);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_zero_bits() {
        unpack(&[], 0, 0);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_33_bits() {
        unpack(&[0; 8], 33, 1);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn unpack_rejects_short_input_tail_only() {
        // 3 * 3 = 9 bits needs 2 bytes, not 1
        unpack(&[0xFF], 3, 3);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn unpack_rejects_short_input_one_byte_under_block_plus_one() {
        // BLOCK_LEN + 1 elements at 8 bits need BLOCK_LEN + 1 bytes
        let short = vec![0u8; BLOCK_LEN];
        unpack(&short, 8, BLOCK_LEN + 1);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn unpack_rejects_short_input_missing_a_block() {
        let short = vec![0u8; block_bytes(5) - 1];
        unpack(&short, 5, BLOCK_LEN);
    }
}
