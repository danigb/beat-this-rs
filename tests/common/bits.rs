//! Bit-level comparison of float slices for the identity tests.
//!
//! Equality is always `to_bits()`: `==` on floats would hide a `-0.0` / `+0.0` swap and would
//! never equate NaNs.

/// How two float slices differ, compared by bit pattern over the shorter length.
#[derive(Debug, Clone)]
pub struct BitDiff {
    pub len_a: usize,
    pub len_b: usize,
    /// Number of positions whose bit patterns differ.
    pub differing: usize,
    /// First differing position.
    pub first: Option<usize>,
    /// Largest absolute difference over differing positions (NaN differences are ignored here).
    pub max_abs: f32,
    /// Largest distance in representable values over differing positions. `-0.0` and `+0.0`
    /// are 1 apart.
    pub max_ulp: u64,
}

impl BitDiff {
    pub fn is_identical(&self) -> bool {
        self.len_a == self.len_b && self.differing == 0
    }
}

/// Position of `x` on the number line of representable values: ascending, with `-0.0` just below
/// `+0.0`.
fn ordered(x: f32) -> i64 {
    let bits = x.to_bits();
    if bits >> 31 == 1 {
        -((bits & 0x7fff_ffff) as i64) - 1
    } else {
        bits as i64
    }
}

pub fn bit_diff(a: &[f32], b: &[f32]) -> BitDiff {
    let mut diff = BitDiff {
        len_a: a.len(),
        len_b: b.len(),
        differing: 0,
        first: None,
        max_abs: 0.0,
        max_ulp: 0,
    };
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            diff.differing += 1;
            diff.first.get_or_insert(i);
            diff.max_abs = diff.max_abs.max((x - y).abs());
            diff.max_ulp = diff.max_ulp.max(ordered(x).abs_diff(ordered(y)));
        }
    }
    diff
}

/// Panics unless `a` and `b` have the same length and the same bit pattern everywhere.
pub fn assert_bits_eq(what: &str, a: &[f32], b: &[f32]) {
    let d = bit_diff(a, b);
    assert!(
        d.is_identical(),
        "{what}: not bit-identical: {d:?} (first: {:?} vs {:?})",
        d.first.and_then(|i| a.get(i)),
        d.first.and_then(|i| b.get(i)),
    );
}
