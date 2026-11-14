//! Grid of the dual-row MQ4 v2 residual GEMV. The kernel writes
//! `blockIdx.x * 2` and `blockIdx.x * 2 + 1`.

use rdna_compute::gemv::mq4v2_residual_workgroups;

#[test]
fn one_workgroup_covers_two_rows() {
    assert_eq!(mq4v2_residual_workgroups(0), 0);
    assert_eq!(mq4v2_residual_workgroups(1), 1);
    assert_eq!(mq4v2_residual_workgroups(2), 1);
    assert_eq!(mq4v2_residual_workgroups(5), 3);
    assert_eq!(mq4v2_residual_workgroups(4096), 2048);
    assert_eq!(mq4v2_residual_workgroups(12288), 6144);
}
