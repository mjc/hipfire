// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL
#define HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL gemm_mq4g256v2_prefill_gfx1010
#endif

// gfx1010 MQ4G256V2 prefill. One workgroup owns ROWS output rows, one wave
// each. Those waves share one K-group of the current 4-token activation
// step through LDS, then each wave replays the scalar GEMV against its own
// weight group. The inner width stays 4: a wider accumulator set spills on
// this arch and a spill changes the FMA order. Inside a group the eight
// nibble terms match gemv_mq4g256v2_xbatch (one expression per accumulator).
// Across a quad the four groups still land in acc0..acc3, the tail still
// lands in acc0..acc2, and the warp reduces with the same pairwise sum and
// shfl_down from 16. HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE selects y += acc.
//
// Weight groups are read from global once per group. Staging every weight
// row plus this activation tile does not fit the 8 KiB cap (two K=4096 rows
// are already 4352 B, and the 4-token group tile is 4096 B). The host still
// admits only a chunk whose weight row is under that cap.
//
// ROWS matches MQ4V2_GFX1010_PREFILL_ROWS. The host launches block (32, ROWS).

#define HIPFIRE_MQ4G256V2_PREFILL_STEP 4
#define HIPFIRE_MQ4G256V2_PREFILL_ROWS 8
#define HIPFIRE_MQ4G256V2_PREFILL_THREADS \
    (HIPFIRE_MQ4G256V2_PREFILL_ROWS * 32)

// __syncthreads() also emits buffer_gl0_inv. The waves only exchange the
// activation tile through LDS, so wait for that traffic and barrier.
__device__ __forceinline__ void mq4v2_lds_barrier() {
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)\n\ts_barrier" ::: "memory");
}

// Fill xtile with x[b, g*256 : g*256+256) for the live tokens, then add this
// group's eight nibble terms into acc[0..local_b). Every thread in the
// workgroup runs both barriers. g_index and acc_slot are each evaluated once
// per expansion, so g+1 is safe.
#define MQ4V2_PREFILL_GROUP(g_index, acc_slot)                                     \
    do {                                                                           \
        const int g_ = (g_index);                                                  \
        unsigned int pk_ = 0;                                                      \
        float sc_ = 0.0f;                                                          \
        float zp_ = 0.0f;                                                          \
        if (active) {                                                              \
            const char* gp_ = row_weights + (long long)g_ * 136;                   \
            const unsigned int hA_ = *(const unsigned int*)(gp_);                  \
            const unsigned int hB_ = *(const unsigned int*)(gp_ + 4);              \
            const unsigned int hs_ = (tid < 16) ? hA_ : hB_;                       \
            sc_ = __half2float(__ushort_as_half((unsigned short)(hs_ & 0xFFFFu)));\
            zp_ = __half2float(__ushort_as_half((unsigned short)(hs_ >> 16)));    \
            pk_ = *(const unsigned int*)(gp_ + 8 + boff);                          \
        }                                                                          \
        const int n4_ = local_b * 64;                                              \
        for (int i_ = linear; i_ < n4_; i_ += HIPFIRE_MQ4G256V2_PREFILL_THREADS) { \
            const int b_ = i_ >> 6;                                                \
            const int v_ = i_ & 63;                                                \
            const float* src_ =                                                    \
                x_tile + ((long long)b_ * K + (long long)g_ * 256 + (v_ << 2));   \
            const float4 got_ = *(const float4*)src_;                             \
            const int dst_ = (b_ << 8) + (v_ << 2);                                \
            xtile[dst_] = got_.x;                                                  \
            xtile[dst_ + 1] = got_.y;                                              \
            xtile[dst_ + 2] = got_.z;                                              \
            xtile[dst_ + 3] = got_.w;                                              \
        }                                                                          \
        mq4v2_lds_barrier();                                                           \
        if (active) {                                                              \
            _Pragma("unroll")                                                      \
            for (int b_ = 0; b_ < HIPFIRE_MQ4G256V2_PREFILL_STEP; b_++) {          \
                if (b_ >= local_b) break;                                          \
                const float* xb_ = xtile + (b_ << 8) + (tid << 3);                 \
                (acc_slot)[b_] +=                                                  \
                    (sc_ * (float)(pk_ & 0xFu) + zp_) * xb_[0]                     \
                    + (sc_ * (float)((pk_ >> 4) & 0xFu) + zp_) * xb_[1]            \
                    + (sc_ * (float)((pk_ >> 8) & 0xFu) + zp_) * xb_[2]            \
                    + (sc_ * (float)((pk_ >> 12) & 0xFu) + zp_) * xb_[3]           \
                    + (sc_ * (float)((pk_ >> 16) & 0xFu) + zp_) * xb_[4]           \
                    + (sc_ * (float)((pk_ >> 20) & 0xFu) + zp_) * xb_[5]           \
                    + (sc_ * (float)((pk_ >> 24) & 0xFu) + zp_) * xb_[6]           \
                    + (sc_ * (float)((pk_ >> 28) & 0xFu) + zp_) * xb_[7];          \
            }                                                                      \
        }                                                                          \
        mq4v2_lds_barrier();                                                           \
    } while (0)

__launch_bounds__(HIPFIRE_MQ4G256V2_PREFILL_THREADS, 1)
extern "C" __global__ void HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    __shared__ float xtile[HIPFIRE_MQ4G256V2_PREFILL_STEP * 256];

    const int tid = threadIdx.x;
    const int wave = threadIdx.y;
    const int row = blockIdx.x * HIPFIRE_MQ4G256V2_PREFILL_ROWS + wave;
    const bool active = row < M;
    const int groups_per_row = K >> 8;
    const char* row_weights = A + (active ? (long long)row * groups_per_row * 136 : 0);
    const int boff = tid * 4;
    const int linear = wave * 32 + tid;

    for (int b0 = 0; b0 < B; b0 += HIPFIRE_MQ4G256V2_PREFILL_STEP) {
        int local_b = B - b0;
        if (local_b > HIPFIRE_MQ4G256V2_PREFILL_STEP) {
            local_b = HIPFIRE_MQ4G256V2_PREFILL_STEP;
        }
        const float* x_tile = x + (long long)b0 * K;
        float* y_tile = y + (long long)b0 * M;

        float acc0[HIPFIRE_MQ4G256V2_PREFILL_STEP];
        float acc1[HIPFIRE_MQ4G256V2_PREFILL_STEP];
        float acc2[HIPFIRE_MQ4G256V2_PREFILL_STEP];
        float acc3[HIPFIRE_MQ4G256V2_PREFILL_STEP];
        #pragma unroll
        for (int b = 0; b < HIPFIRE_MQ4G256V2_PREFILL_STEP; b++) {
            acc0[b] = 0.0f;
            acc1[b] = 0.0f;
            acc2[b] = 0.0f;
            acc3[b] = 0.0f;
        }

        const int quads = groups_per_row >> 2;
        const int tail = groups_per_row & 3;

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            MQ4V2_PREFILL_GROUP(g, acc0);
            MQ4V2_PREFILL_GROUP(g + 1, acc1);
            MQ4V2_PREFILL_GROUP(g + 2, acc2);
            MQ4V2_PREFILL_GROUP(g + 3, acc3);
        }
        for (int t = 0; t < tail; t++) {
            const int g = (quads << 2) + t;
            if (t == 0) {
                MQ4V2_PREFILL_GROUP(g, acc0);
            } else if (t == 1) {
                MQ4V2_PREFILL_GROUP(g, acc1);
            } else {
                MQ4V2_PREFILL_GROUP(g, acc2);
            }
        }

        if (active) {
            #pragma unroll
            for (int b = 0; b < HIPFIRE_MQ4G256V2_PREFILL_STEP; b++) {
                if (b >= local_b) break;
                float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_down(acc, offset);
#if defined(HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE)
                if (tid == 0) y_tile[(long long)b * M + row] += acc;
#else
                if (tid == 0) y_tile[(long long)b * M + row] = acc;
#endif
            }
        }
    }
}
