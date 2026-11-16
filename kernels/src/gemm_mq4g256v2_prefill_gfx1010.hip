// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL
#define HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL gemm_mq4g256v2_prefill_gfx1010
#endif

// gfx1010 MQ4G256V2 prefill. One wave per output row copies that weight row
// into LDS once, then replays gemv_mq4g256v2_xbatch for activation rows in
// steps of 4. The inner width stays 4: a wider accumulator set spills on
// this arch and a spill changes the FMA order. Inside a step the terms match
// that kernel (XDOG across a quad, tail into acc0..acc2, pairwise sum,
// shfl_down from 16). HIPFIRE_MQ4G256V2_RESIDUAL_EPILOGUE selects y += acc.
//
// The host launches this for a chunk whose weight row fits the LDS cap.
// A shorter leftover stays on gemv_mq4g256v2_xbatch.

#define HIPFIRE_MQ4G256V2_PREFILL_STEP 4

__launch_bounds__(32, 8)
extern "C" __global__ void HIPFIRE_MQ4G256V2_PREFILL_GFX1010_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    extern __shared__ unsigned int smem[];
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const unsigned int* src =
        (const unsigned int*)(A + (long long)row * groups_per_row * 136);
    const int n_u32 = groups_per_row * (136 / 4);
    for (int i = tid; i < n_u32; i += 32) {
        smem[i] = src[i];
    }
    __syncthreads();
    const char* row_ptr = (const char*)smem;
    const int boff = tid * 4;

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
            acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
        }

        const int quads = groups_per_row >> 2;
        const int tail = groups_per_row & 3;

        for (int q = 0; q < quads; q++) {
            const int g = q << 2;
            const char* gp0 = row_ptr + g * 136;
            const char* gp1 = gp0 + 136;
            const char* gp2 = gp1 + 136;
            const char* gp3 = gp2 + 136;

            const unsigned int hA0 = *(const unsigned int*)(gp0);
            const unsigned int hB0 = *(const unsigned int*)(gp0 + 4);
            const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
            float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
            float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
            const unsigned int hA1 = *(const unsigned int*)(gp1);
            const unsigned int hB1 = *(const unsigned int*)(gp1 + 4);
            const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
            float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
            float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
            const unsigned int hA2 = *(const unsigned int*)(gp2);
            const unsigned int hB2 = *(const unsigned int*)(gp2 + 4);
            const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
            float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
            float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
            const unsigned int hA3 = *(const unsigned int*)(gp3);
            const unsigned int hB3 = *(const unsigned int*)(gp3 + 4);
            const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
            float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
            float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

            unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

            const int base = g * 256 + tid * 8;

            #define XDOG(pk, sc, zp, bs, a) \
                (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
                     + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
                     + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
                     + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
                     + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
                     + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
                     + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
                     + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

            #pragma unroll
            for (int b = 0; b < HIPFIRE_MQ4G256V2_PREFILL_STEP; b++) {
                if (b >= local_b) break;
                const float* xb = x_tile + (long long)b * K;
                XDOG(pk0, sc0, zp0, base,       acc0[b]);
                XDOG(pk1, sc1, zp1, base + 256, acc1[b]);
                XDOG(pk2, sc2, zp2, base + 512, acc2[b]);
                XDOG(pk3, sc3, zp3, base + 768, acc3[b]);
            }
            #undef XDOG
        }

        #define XTAIL(pk, sc, zp, bs, a) \
            (a) += (sc * (float)((pk) & 0xFu)         + zp) * xb[(bs)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * xb[(bs) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * xb[(bs) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * xb[(bs) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * xb[(bs) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * xb[(bs) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * xb[(bs) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * xb[(bs) + 7]

        for (int t = 0; t < tail; t++) {
            const int g = (quads << 2) + t;
            const char* gp = row_ptr + g * 136;
            const unsigned int hA = *(const unsigned int*)(gp);
            const unsigned int hB = *(const unsigned int*)(gp + 4);
            const unsigned int hs = (tid < 16) ? hA : hB;
            float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
            const int base = g * 256 + tid * 8;
            #pragma unroll
            for (int b = 0; b < HIPFIRE_MQ4G256V2_PREFILL_STEP; b++) {
                if (b >= local_b) break;
                const float* xb = x_tile + (long long)b * K;
                if (t == 0)      { XTAIL(pk, sc, zp, base, acc0[b]); }
                else if (t == 1) { XTAIL(pk, sc, zp, base, acc1[b]); }
                else             { XTAIL(pk, sc, zp, base, acc2[b]); }
            }
        }
        #undef XTAIL

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
