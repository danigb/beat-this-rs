//! Shared helpers for the parity / cross-runtime tests.
//!
//! `common/mod.rs` is the standard shared-test-module layout: cargo does not
//! compile it as its own test binary, only as a module included via `mod common;`.
//! Each including test crate uses a different subset, so allow unused items here.
#![allow(dead_code)]

use serde::Deserialize;
use std::path::Path;

/// A golden output produced from the Python reference (`scripts/gen_golden.py`).
#[derive(Debug, Deserialize)]
pub struct Golden {
    pub beats: Vec<f32>,
    pub downbeats: Vec<f32>,
}

/// Load a committed golden JSON fixture, panicking with a clear message on failure.
pub fn load_golden(path: &str) -> Golden {
    let text = std::fs::read_to_string(Path::new(path))
        .unwrap_or_else(|e| panic!("failed to read golden {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("failed to parse golden {path}: {e}"))
}

/// Result of an F-measure comparison, including timing diagnostics.
pub struct Score {
    pub f_measure: f64,
    pub matched: usize,
    pub n_ref: usize,
    pub n_est: usize,
    pub max_matched_diff: f64,
    pub mean_matched_diff: f64,
}

/// Standard MIR beat F-measure with one-to-one matching inside `window` seconds.
///
/// Greedy two-pointer over sorted sequences: for near-identical inputs this equals
/// the optimal bipartite matching; if it ever under-matches it lowers F (fails loud),
/// never falsely inflating it.
pub fn f_measure(reference: &[f32], estimated: &[f32], window: f64) -> Score {
    let mut r: Vec<f64> = reference.iter().map(|&x| x as f64).collect();
    let mut e: Vec<f64> = estimated.iter().map(|&x| x as f64).collect();
    r.sort_by(|a, b| a.total_cmp(b));
    e.sort_by(|a, b| a.total_cmp(b));

    let (mut i, mut j, mut matched) = (0usize, 0usize, 0usize);
    let mut max_diff = 0.0f64;
    let mut sum_diff = 0.0f64;
    while i < r.len() && j < e.len() {
        let d = (r[i] - e[j]).abs();
        if d <= window {
            matched += 1;
            max_diff = max_diff.max(d);
            sum_diff += d;
            i += 1;
            j += 1;
        } else if r[i] < e[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    let precision = if e.is_empty() {
        0.0
    } else {
        matched as f64 / e.len() as f64
    };
    let recall = if r.is_empty() {
        0.0
    } else {
        matched as f64 / r.len() as f64
    };
    let f = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    Score {
        f_measure: f,
        matched,
        n_ref: r.len(),
        n_est: e.len(),
        max_matched_diff: if matched > 0 { max_diff } else { 0.0 },
        mean_matched_diff: if matched > 0 {
            sum_diff / matched as f64
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences_score_one() {
        let seq = vec![0.3, 0.54, 0.84, 1.12];
        let s = f_measure(&seq, &seq, 0.070);
        assert_eq!(s.f_measure, 1.0);
        assert_eq!(s.matched, 4);
        assert_eq!(s.max_matched_diff, 0.0);
    }

    #[test]
    fn within_window_still_matches() {
        let reference = vec![1.0, 2.0, 3.0];
        let estimated = vec![1.05, 1.96, 3.02]; // all within 70 ms
        let s = f_measure(&reference, &estimated, 0.070);
        assert_eq!(s.f_measure, 1.0);
        assert!(s.max_matched_diff <= 0.05 + 1e-9);
    }

    #[test]
    fn beat_outside_window_lowers_f() {
        let reference = vec![1.0, 2.0, 3.0];
        let estimated = vec![1.0, 2.0, 3.2]; // last is 200 ms off
        let s = f_measure(&reference, &estimated, 0.070);
        assert_eq!(s.matched, 2);
        assert!(s.f_measure < 1.0);
    }

    #[test]
    fn extra_estimate_lowers_precision() {
        let reference = vec![1.0, 2.0];
        let estimated = vec![1.0, 2.0, 5.0]; // spurious extra beat
        let s = f_measure(&reference, &estimated, 0.070);
        assert_eq!(s.matched, 2);
        assert!(s.f_measure < 1.0); // F = 2*1.0*(2/3)/(1.0+2/3) = 0.8
    }

    #[test]
    fn empty_sequences() {
        let s = f_measure(&[], &[], 0.070);
        assert_eq!(s.f_measure, 0.0);
        assert_eq!(s.matched, 0);
    }
}
