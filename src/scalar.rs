// This bit-packs the 'tail' elements in a way that does not require any particular chunk length like bitpacking does.
// Note: The packer should keep track of the number of elements, because if num_bits is small, it can be vague
// to get the number of elements from a packed buffer length.
pub fn pack(data: &[u32], num_bits: u8, out: &mut Vec<u8>) {
    assert!((1..=32).contains(&num_bits));
    let mask: u32 = u32::MAX >> (32 - num_bits);
    let mut buffer: u64 = 0;
    let mut bits_in_buffer: u8 = 0;
    for &v in data {
        buffer |= ((v & mask) as u64) << bits_in_buffer;
        bits_in_buffer += num_bits;
        while bits_in_buffer >= 8 {
            out.push(buffer as u8);
            buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }
    if bits_in_buffer > 0 {
        out.push(buffer as u8);
    }
}

pub fn unpack(data: &[u8], num_bits: u8, count: usize, out: &mut Vec<u32>) {
    assert!((1..=32).contains(&num_bits));
    assert!(
        data.len() >= (count * num_bits as usize).div_ceil(8),
        "input too short for {count} elements of {num_bits} bits"
    );

    let mask: u32 = u32::MAX >> (32 - num_bits);
    let mut bytes = data.iter();
    let mut buffer: u64 = 0;
    let mut bits_in_buffer: u8 = 0;

    for _ in 0..count {
        while bits_in_buffer < num_bits {
            // length was checked above, so this can't run dry
            buffer |= (*bytes.next().unwrap() as u64) << bits_in_buffer;
            bits_in_buffer += 8;
        }
        out.push(buffer as u32 & mask);
        buffer >>= num_bits;
        bits_in_buffer -= num_bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_for(num_bits: u8) -> u32 {
        u32::MAX >> (32 - num_bits)
    }

    /// Pack into a fresh Vec.
    fn pack_vec(data: &[u32], num_bits: u8) -> Vec<u8> {
        let mut out = Vec::new();
        pack(data, num_bits, &mut out);
        out
    }

    /// Unpack into a fresh Vec.
    fn unpack_vec(data: &[u8], num_bits: u8, count: usize) -> Vec<u32> {
        let mut out = Vec::new();
        unpack(data, num_bits, count, &mut out);
        out
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

    // ---------- known layouts (LSB-first) ----------

    #[test]
    fn pack_8_bits_is_identity() {
        let data = [0x00, 0x01, 0x7F, 0x80, 0xFF];
        assert_eq!(pack_vec(&data, 8), vec![0x00, 0x01, 0x7F, 0x80, 0xFF]);
    }

    #[test]
    fn pack_4_bits_low_nibble_first() {
        // 0x1 in the low nibble, 0x2 in the high nibble, then 0x3 alone
        assert_eq!(pack_vec(&[0x1, 0x2, 0x3], 4), vec![0x21, 0x03]);
    }

    #[test]
    fn pack_1_bit() {
        // bits 0..8 = 1,0,1,1,0,0,0,0 -> 0b0000_1101, then a lone 1
        let data = [1, 0, 1, 1, 0, 0, 0, 0, 1];
        assert_eq!(pack_vec(&data, 1), vec![0x0D, 0x01]);
    }

    #[test]
    fn pack_3_bits_straddles_bytes() {
        // 1 | 2<<3 | 7<<6 = 0x1D1
        assert_eq!(pack_vec(&[1, 2, 7], 3), vec![0xD1, 0x01]);
    }

    #[test]
    fn pack_12_bits() {
        // 0xABC | 0xDEF<<12 = 0xDEFABC, little-endian bytes
        assert_eq!(pack_vec(&[0xABC, 0xDEF], 12), vec![0xBC, 0xFA, 0xDE]);
    }

    #[test]
    fn pack_32_bits_is_little_endian() {
        assert_eq!(pack_vec(&[0x1234_5678], 32), vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn pack_masks_out_of_range_values() {
        // high bits of each value must not leak into its neighbour
        assert_eq!(pack_vec(&[0xFF, 0x12], 4), vec![0x2F]);
    }

    // ---------- sizes and empties ----------

    #[test]
    fn empty_input() {
        for num_bits in 1..=32u8 {
            assert!(pack_vec(&[], num_bits).is_empty());
            assert!(unpack_vec(&[], num_bits, 0).is_empty());
        }
    }

    #[test]
    fn packed_length_is_ceil_of_total_bits() {
        for num_bits in 1..=32u8 {
            for len in 0..40 {
                let packed = pack_vec(&sample(num_bits, len), num_bits);
                assert_eq!(
                    packed.len(),
                    (len * num_bits as usize).div_ceil(8),
                    "num_bits={num_bits} len={len}"
                );
            }
        }
    }

    #[test]
    fn padding_bits_are_zero() {
        // 3 elements * 3 bits = 9 bits; the top 7 bits of byte 1 must be zero
        let packed = pack_vec(&[7, 7, 7], 3);
        assert_eq!(packed, vec![0xFF, 0x01]);
    }

    // ---------- append semantics (out is &mut Vec) ----------

    #[test]
    fn pack_appends_and_preserves_existing_bytes() {
        let mut out = vec![0xAA, 0xBB];
        pack(&[0x1, 0x2], 4, &mut out);
        assert_eq!(out, vec![0xAA, 0xBB, 0x21]);
    }

    #[test]
    fn pack_empty_input_leaves_out_untouched() {
        let mut out = vec![1, 2, 3];
        pack(&[], 5, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn pack_each_call_starts_on_a_byte_boundary() {
        // Padding is flushed at the end of every call, so two calls are NOT
        // the same as one call over the concatenated data.
        let mut split = Vec::new();
        pack(&[1], 4, &mut split);
        pack(&[2], 4, &mut split);
        assert_eq!(split, vec![0x01, 0x02]);

        assert_eq!(pack_vec(&[1, 2], 4), vec![0x21]);
    }

    #[test]
    fn pack_two_calls_roundtrip_separately() {
        // This is the hybrid use case: independent segments appended to one buffer.
        let a = sample(5, 13);
        let b = sample(5, 9);
        let mut out = Vec::new();
        pack(&a, 5, &mut out);
        let split = out.len();
        pack(&b, 5, &mut out);

        assert_eq!(split, (13usize * 5).div_ceil(8));
        assert_eq!(unpack_vec(&out[..split], 5, 13), a);
        assert_eq!(unpack_vec(&out[split..], 5, 9), b);
    }

    #[test]
    fn unpack_appends_and_preserves_existing_elements() {
        let mut out = vec![99u32];
        unpack(&[0x21], 4, 2, &mut out);
        assert_eq!(out, vec![99, 0x1, 0x2]);
    }

    #[test]
    fn unpack_count_zero_leaves_out_untouched() {
        let mut out = vec![7u32, 8];
        unpack(&[0xFF], 4, 0, &mut out);
        assert_eq!(out, vec![7, 8]);
    }

    #[test]
    fn unpack_appends_exactly_count_elements() {
        for num_bits in 1..=32u8 {
            let data = sample(num_bits, 20);
            let packed = pack_vec(&data, num_bits);
            let mut out = vec![0xDEAD_BEEF; 3];
            unpack(&packed, num_bits, 20, &mut out);
            assert_eq!(out.len(), 23, "num_bits={num_bits}");
            assert_eq!(&out[..3], &[0xDEAD_BEEF; 3]);
            assert_eq!(&out[3..], &data[..]);
        }
    }

    // ---------- round trips ----------

    #[test]
    fn roundtrip_all_widths_and_lengths() {
        for num_bits in 1..=32u8 {
            for len in [0usize, 1, 2, 3, 7, 8, 9, 31, 32, 33, 100, 1000] {
                let data = sample(num_bits, len);
                let packed = pack_vec(&data, num_bits);
                assert_eq!(
                    unpack_vec(&packed, num_bits, len),
                    data,
                    "num_bits={num_bits} len={len}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_max_values() {
        for num_bits in 1..=32u8 {
            let data = vec![mask_for(num_bits); 50];
            let packed = pack_vec(&data, num_bits);
            assert_eq!(unpack_vec(&packed, num_bits, 50), data, "num_bits={num_bits}");
        }
    }

    #[test]
    fn roundtrip_all_zeros() {
        for num_bits in 1..=32u8 {
            let data = vec![0u32; 50];
            let packed = pack_vec(&data, num_bits);
            assert!(packed.iter().all(|&b| b == 0));
            assert_eq!(unpack_vec(&packed, num_bits, 50), data);
        }
    }

    #[test]
    fn roundtrip_alternating_bit_patterns() {
        // catches shifts that bleed across element boundaries
        for num_bits in 1..=32u8 {
            let m = mask_for(num_bits);
            let data: Vec<u32> = (0..64)
                .map(|i| if i % 2 == 0 { m } else { 0 })
                .collect();
            let packed = pack_vec(&data, num_bits);
            assert_eq!(unpack_vec(&packed, num_bits, 64), data, "num_bits={num_bits}");
        }
    }

    // ---------- unpack specifics ----------

    #[test]
    fn unpack_known_bytes() {
        assert_eq!(unpack_vec(&[0x21, 0x03], 4, 3), vec![0x1, 0x2, 0x3]);
        assert_eq!(unpack_vec(&[0xBC, 0xFA, 0xDE], 12, 2), vec![0xABC, 0xDEF]);
        assert_eq!(unpack_vec(&[0x78, 0x56, 0x34, 0x12], 32, 1), vec![0x1234_5678]);
    }

    #[test]
    fn unpack_ignores_trailing_bytes() {
        // extra bytes beyond what `count` needs are never read
        assert_eq!(unpack_vec(&[0xFF, 0xFF, 0xFF], 4, 2), vec![0xF, 0xF]);
    }

    #[test]
    fn unpack_count_smaller_than_packed() {
        let data = sample(7, 20);
        let packed = pack_vec(&data, 7);
        assert_eq!(unpack_vec(&packed, 7, 5), data[..5].to_vec());
    }

    #[test]
    fn unpack_count_inferred_from_length_overshoots_for_small_widths() {
        // documents why the count must be stored: 3 * 3 bits = 2 bytes,
        // and 16 / 3 = 5 would decode two phantom zeros
        let packed = pack_vec(&[1, 2, 3], 3);
        let decoded = unpack_vec(&packed, 3, packed.len() * 8 / 3);
        assert_eq!(decoded, vec![1, 2, 3, 0, 0]);
    }

    // ---------- panics ----------

    #[test]
    #[should_panic]
    fn pack_rejects_zero_bits() {
        pack_vec(&[1], 0);
    }

    #[test]
    #[should_panic]
    fn pack_rejects_33_bits() {
        pack_vec(&[1], 33);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_zero_bits() {
        unpack_vec(&[0], 0, 1);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_33_bits() {
        unpack_vec(&[0; 8], 33, 1);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn unpack_rejects_short_input() {
        // 3 * 8 bits needs 3 bytes
        unpack_vec(&[0, 0], 8, 3);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn unpack_rejects_short_input_partial_byte() {
        // 3 * 3 = 9 bits needs 2 bytes, not 1
        unpack_vec(&[0xFF], 3, 3);
    }
}
