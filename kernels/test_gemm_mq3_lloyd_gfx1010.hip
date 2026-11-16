// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// Run from the repository root with hipcc --offload-arch=gfx1010 -I.
#include "kernels/src/gemm_mq3g256_lloyd_residual.gfx1010.hip"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>
#define CHECK(x)                                                               \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e));                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main() {
  int count = 0, dev = -1;
  CHECK(hipGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, i));
    if (strstr(p.gcnArchName, "gfx1010"))
      dev = i;
  }
  if (dev < 0) {
    fprintf(stderr, "gfx1010 unavailable\n");
    return 1;
  }
  CHECK(hipSetDevice(dev));
  for (int groups : {1, 2, 5})
    for (int batches : {1, 7, 8, 9, 17}) {
      int rows = 3, k = groups * 256;
      std::vector<unsigned char> w(rows * groups * 112);
      std::vector<float> x(batches * k), y(batches * rows), expected(y.size());
      for (int r = 0; r < rows; r++)
        for (int g = 0; g < groups; g++) {
          auto p = w.data() + (r * groups + g) * 112;
          for (int j = 0; j < 8; j++) {
            __half h = __float2half((j - 3) * 0.125f + g * 0.03125f);
            memcpy(p + 2 * j, &h, 2);
          }
          for (int l = 0; l < 32; l++) {
            unsigned bits = 0;
            for (int j = 0; j < 8; j++)
              bits |= ((r + g + l + j) % 8) << (3 * j);
            for (int j = 0; j < 3; j++)
              p[16 + l * 3 + j] = (bits >> (8 * j)) & 255;
          }
        }
      for (size_t i = 0; i < x.size(); i++)
        x[i] = ((int)(i % 23) - 11) * 0.0625f;
      for (size_t i = 0; i < y.size(); i++)
        y[i] = expected[i] = (float)i * 0.01f;
      for (int b = 0; b < batches; b++)
        for (int r = 0; r < rows; r++)
          for (int g = 0; g < groups; g++)
            for (int l = 0; l < 32; l++)
              for (int j = 0; j < 8; j++)
                expected[b * rows + r] +=
                    ((r + g + l + j) % 8 - 3) * 0.125f *
                        x[b * k + g * 256 + l * 8 + j] +
                    g * 0.03125f * x[b * k + g * 256 + l * 8 + j];
      unsigned char *dw;
      float *dx, *dy;
      CHECK(hipMalloc(&dw, w.size()));
      CHECK(hipMalloc(&dx, x.size() * 4));
      CHECK(hipMalloc(&dy, y.size() * 4));
      CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dy, y.data(), y.size() * 4, hipMemcpyHostToDevice));
      hipLaunchKernelGGL(gemm_mq3g256_lloyd_residual_gfx1010,
                         dim3(rows, (batches + 7) / 8), dim3(32), 0, 0,
                         (const char *)dw, dx, dy, rows, k, batches);
      CHECK(hipGetLastError());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < y.size(); i++)
        if (fabsf(y[i] - expected[i]) > 0.002f) {
          fprintf(
              stderr,
              "mismatch groups=%d batch=%d index=%zu actual=%f expected=%f\n",
              groups, batches, i, y[i], expected[i]);
          return 1;
        }
      hipLaunchKernelGGL(gemm_mq3g256_lloyd_gfx1010,
                         dim3(rows, (batches + 7) / 8), dim3(32), 0, 0,
                         (const char *)dw, dx, dy, rows, k, batches);
      CHECK(hipGetLastError());
      CHECK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < y.size(); i++)
        if (fabsf(y[i] - (expected[i] - (float)i * 0.01f)) > 0.002f) {
          fprintf(stderr, "set mismatch groups=%d batch=%d index=%zu\n", groups,
                  batches, i);
          return 1;
        }
      CHECK(hipFree(dw));
      CHECK(hipFree(dx));
      CHECK(hipFree(dy));
    }
  puts("30 gfx1010 set and residual cases passed");
}
