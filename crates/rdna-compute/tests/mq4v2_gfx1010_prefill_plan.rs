// SPDX-License-Identifier: Apache-2.0
// gfx1010 MQ4 v2 prefill launch plan. CPU-only: no HIP, no GPU.
// A 256-row chunk whose weight row fits in LDS is one tiled launch.
// A shorter leftover is ceil(rows/4) launches of the 4-row x-batch GEMV.
use rdna_compute::gemm::{
    mq4v2_gfx1010_prefill_tiled, MQ4V2_GFX1010_PREFILL_LDS_MAX, MQ4V2_GFX1010_PREFILL_TILE,
};

const GROUP_BYTES: usize = 136;
const XBATCH_ROWS: usize = 4;

fn launches(k: usize, batch: usize) -> usize {
    if batch == 0 {
        return 0;
    }
    if mq4v2_gfx1010_prefill_tiled(k, batch) {
        1
    } else {
        batch.div_ceil(XBATCH_ROWS)
    }
}

#[test]
fn tile_matches_the_gfx1010_chunk() {
    assert_eq!(MQ4V2_GFX1010_PREFILL_TILE, 256);
    assert_eq!(MQ4V2_GFX1010_PREFILL_LDS_MAX, 8 * 1024);
}

#[test]
fn qwen35_9b_chunk_is_one_launch() {
    // hidden K=4096 is 2176 B; intermediate K=12288 is 6528 B.
    for k in [4096_usize, 12288] {
        let row = k / 256 * GROUP_BYTES;
        assert!(row <= MQ4V2_GFX1010_PREFILL_LDS_MAX, "K={k} row {row}");
        assert!(mq4v2_gfx1010_prefill_tiled(k, 256));
        assert_eq!(launches(k, 256), 1, "K={k} must not be 64 x-batch launches");
        assert_eq!(launches(k, 512), 1, "K={k} covers the whole call");
    }
}

#[test]
fn shorter_than_the_tile_stays_on_xbatch() {
    let k = 12288;
    assert!(!mq4v2_gfx1010_prefill_tiled(k, 0));
    assert_eq!(launches(k, 0), 0);
    assert!(!mq4v2_gfx1010_prefill_tiled(k, 6));
    assert_eq!(launches(k, 6), 2);
    assert!(!mq4v2_gfx1010_prefill_tiled(k, 255));
    assert_eq!(launches(k, 255), 64);
    assert!(!mq4v2_gfx1010_prefill_tiled(k, 4));
    assert_eq!(launches(k, 4), 1);
}

#[test]
fn row_over_the_lds_cap_stays_on_xbatch() {
    // 60 groups is 8160 B and still tiles. 61 groups is 8296 B and does not.
    let fits = 60 * 256;
    let spills = 61 * 256;
    assert_eq!(fits / 256 * GROUP_BYTES, 8160);
    assert_eq!(spills / 256 * GROUP_BYTES, 8296);
    assert!(mq4v2_gfx1010_prefill_tiled(fits, 256));
    assert_eq!(launches(fits, 256), 1);
    assert!(!mq4v2_gfx1010_prefill_tiled(spills, 256));
    assert_eq!(launches(spills, 256), 64);
}

#[test]
fn k_not_a_group_multiple_does_not_tile() {
    assert!(!mq4v2_gfx1010_prefill_tiled(4097, 256));
}
