// The stock decode's float arithmetic, one rounding at a time: PyTorch's eager CUDA kernels
// (torch 2.14, ATen/native/cuda/Reduce.cuh for the means) and fla's Triton step (its PTX).
// Every step that rounds is an explicit intrinsic, which nvcc never contracts or reorders.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace mk {

typedef uint16_t bf16;  // raw bits

__device__ __forceinline__ float f(bf16 b) { return __uint_as_float(((uint32_t)b) << 16); }
__device__ __forceinline__ bf16 rb(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float rbf(float x) { return f(rb(x)); }

// --- PyTorch eager, fp32 opmath ---------------------------------------------------------------
__device__ __forceinline__ float silu(float x) {
  return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}
__device__ __forceinline__ float sigmoid(float x) {
  return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));
}
__device__ __forceinline__ float softplus(float x) {  // beta 1, threshold 20
  return x > 20.0f ? x : log1pf(expf(x));
}
__device__ __forceinline__ float t_rsqrt(float x) { return rsqrtf(x); }
// bf16 * bf16 -> bf16, bf16 + bf16 -> bf16 (computed in fp32, rounded once)
__device__ __forceinline__ bf16 mul_b(bf16 a, bf16 b) { return rb(__fmul_rn(f(a), f(b))); }
__device__ __forceinline__ bf16 add_b(bf16 a, bf16 b) { return rb(__fadd_rn(f(a), f(b))); }

// --- Triton (fla) lowerings -------------------------------------------------------------------
__device__ __forceinline__ float tri_exp(float x) {
  float y, t = __fmul_rn(x, 1.44269502162933349609375f);  // 0x3FB8AA3B
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(t));
  return y;
}
__device__ __forceinline__ float tri_sqrt(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tri_div(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}

// --- PyTorch's mean of squares ----------------------------------------------------------------
// x.pow(2).mean(-1) over one row of 2560 fp32 (Reduce.cuh, vectorized input, 512 threads): each
// thread sums float4 vectors t and t + 512 into 4 accumulators, combined in order; a shared-
// memory tree at offsets 256..32, then shfl_down 16..1. Call with all 512 threads; red holds
// 512 floats. Returns the sum (not yet times 1/2560) in every thread.
__device__ __forceinline__ float torch_sumsq_2560(const float* x, float* red) {
  const int t = threadIdx.x;
  float a[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float v = x[4 * t + i];
    a[i] = __fmul_rn(v, v);
  }
  if (t < 128) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float v = x[4 * (t + 512) + i];
      a[i] = __fadd_rn(a[i], __fmul_rn(v, v));
    }
  }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
  red[t] = s;
#pragma unroll
  for (int off = 256; off >= 32; off >>= 1) {
    __syncthreads();
    if (t < off) {
      s = __fadd_rn(s, red[t + off]);
      red[t] = s;
    }
  }
  __syncthreads();
  if (t < 32) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
    if (t == 0) red[0] = s;
  }
  __syncthreads();
  s = red[0];
  __syncthreads();
  return s;
}

// The same over a row of 128 (the gated norm: rows of 128 are vectorized too, block width 32):
// lane l sums float4 vector l in order, then shfl_down 16..1. One full warp; the sum in lane 0.
__device__ __forceinline__ float torch_sumsq_128(const float* x) {
  const int l = threadIdx.x & 31;
  float a[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float v = x[4 * l + i];
    a[i] = __fmul_rn(v, v);
  }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(a[0], a[1]), a[2]), a[3]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
  return s;
}

// --- fla's recurrent step ---------------------------------------------------------------------
// tl.sum(x * x) over 256 (one warp, 8 consecutive elements per lane): in-lane pairs
// fma(x_odd, x_odd, x_even^2) (FMA_ODD) or fma(x_even, x_even, x_odd^2), a tree over the four
// pairs, then a butterfly at 16..1. Every lane gets the sum.
template <bool FMA_ODD>
__device__ __forceinline__ float fla_sumsq_256(const float* x) {
  const int l = threadIdx.x & 31;
  float p[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float e = x[8 * l + 2 * i], o = x[8 * l + 2 * i + 1];
    p[i] = FMA_ODD ? __fmaf_rn(o, o, __fmul_rn(e, e)) : __fmaf_rn(e, e, __fmul_rn(o, o));
  }
  float s = __fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  return s;
}

// tl.sum(h * w[:, None], 0) for one column of the [256, 8] state tile, over only the rows where
// w is nonzero (the others add exact zeros). The tile's layout (sizePerThread [1, 4],
// threadsPerWarp [16, 2]) sums row bits in the order 4, 5, 6, 7 (in a thread: the first level
// fused, fma(h_lo, w_lo, h_hi * w_hi)), then 3, 2, 1, 0 (the lanes' butterfly). pos[i] is row
// i's place in that order (tree_pos), ascending; n <= N.
__device__ __host__ __forceinline__ int tree_pos(int row) {
  return ((row >> 4) & 15) | (((row >> 3) & 1) << 4) | (((row >> 2) & 1) << 5) |
         (((row >> 1) & 1) << 6) | ((row & 1) << 7);
}
__device__ __host__ __forceinline__ int tree_row(int p) {
  return ((p & 15) << 4) | (((p >> 4) & 1) << 3) | (((p >> 5) & 1) << 2) |
         (((p >> 6) & 1) << 1) | ((p >> 7) & 1);
}

template <int N>
__device__ __forceinline__ float fla_dot(int n, const int* pos, const float* h, const float* w) {
  float val[N];
  int key[N];
  int m = 0;
  for (int i = 0; i < n;) {
    if (i + 1 < n && (pos[i] >> 1) == (pos[i + 1] >> 1)) {
      val[m] = __fmaf_rn(h[i], w[i], __fmul_rn(h[i + 1], w[i + 1]));
      i += 2;
    } else {
      val[m] = __fmul_rn(h[i], w[i]);
      i += 1;
    }
    key[m++] = pos[i - 1] >> 1;
  }
  for (int lev = 1; lev < 8; lev++) {
    int m2 = 0;
    for (int i = 0; i < m;) {
      if (i + 1 < m && (key[i] >> 1) == (key[i + 1] >> 1)) {
        val[m2] = __fadd_rn(val[i], val[i + 1]);
        i += 2;
      } else {
        val[m2] = val[i];
        i += 1;
      }
      key[m2++] = key[i - 1] >> 1;
    }
    m = m2;
  }
  return m ? val[0] : 0.0f;
}

}  // namespace mk
