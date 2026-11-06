// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// Batched Lloyd MQ3 projection with residual accumulation for gfx1010.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BATCH_TILE 8

extern "C" __global__ __launch_bounds__(32, 6) void gemm_mq3g256_lloyd_residual_gfx1010(
    const char* __restrict__ weights, const float* __restrict__ input,
    float* __restrict__ output, int rows, int k, int batch_size
) {
    const int row = blockIdx.x;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int lane = threadIdx.x;
    if (row >= rows || batch_start >= batch_size) return;
    const int groups = k / 256;
    const char* row_ptr = weights + (long long)row * groups * 112;
    float sum[BATCH_TILE] = {};
    for (int group = 0; group < groups; ++group) {
        const char* block = row_ptr + group * 112;
        const unsigned char* packed = (const unsigned char*)(block + 16 + lane * 3);
        const unsigned int bits = (unsigned int)packed[0]
            | ((unsigned int)packed[1] << 8) | ((unsigned int)packed[2] << 16);
        const __half* codebook = (const __half*)block;
        float decoded[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            decoded[i] = __half2float(codebook[(bits >> (3 * i)) & 7u]);
        const int offset = group * 256 + lane * 8;
        #pragma unroll
        for (int b = 0; b < BATCH_TILE; ++b) {
            if (batch_start + b >= batch_size) break;
            const float* x = input + (long long)(batch_start + b) * k + offset;
            #pragma unroll
            for (int i = 0; i < 8; ++i) sum[b] += decoded[i] * x[i];
        }
    }
    #pragma unroll
    for (int b = 0; b < BATCH_TILE; ++b) {
        if (batch_start + b >= batch_size) break;
        float value = sum[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down(value, offset);
        if (lane == 0) output[(long long)(batch_start + b) * rows + row] += value;
    }
}
