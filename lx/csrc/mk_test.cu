// Unit tests of mk_math.cuh against torch and fla (tests/check_mk_math.py).
#include <cuda_runtime.h>

#include "mk_math.cuh"

using namespace mk;

// mean of squares over rows of 2560, and rsqrt(mean + eps)
__global__ void k_norm2560(const float* x, float* mean, float* r, float eps) {
  __shared__ float red[512];
  const float* xr = x + (size_t)blockIdx.x * 2560;
  float s = torch_sumsq_2560(xr, red);
  if (threadIdx.x == 0) {
    float m = __fmul_rn(s, 1.0f / 2560.0f);
    mean[blockIdx.x] = m;
    r[blockIdx.x] = t_rsqrt(__fadd_rn(m, eps));
  }
}

// Qwen3_5RMSNormGated on rows of 128: one warp per row
__global__ void k_gated(const bf16* o, const bf16* z, const bf16* w, bf16* out, float eps) {
  const int row = blockIdx.x, l = threadIdx.x;
  __shared__ float x[128];
  for (int j = l; j < 128; j += 32) x[j] = f(o[row * 128 + j]);
  __syncwarp();
  float s = torch_sumsq_128(x);
  s = __shfl_sync(0xffffffffu, s, 0);
  float r = t_rsqrt(__fadd_rn(__fmul_rn(s, 1.0f / 128.0f), eps));
  for (int j = l; j < 128; j += 32) {
    bf16 t = rb(__fmul_rn(x[j], r));
    bf16 u = mul_b(w[j], t);
    out[row * 128 + j] = rb(__fmul_rn(f(u), silu(f(z[row * 128 + j]))));
  }
}

__global__ void k_elem(const float* x, float* out, int n, int which) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i], y = 0.0f;
  switch (which) {
    case 0: y = silu(v); break;
    case 1: y = sigmoid(v); break;
    case 2: y = softplus(v); break;
    case 3: y = t_rsqrt(v); break;
    case 4: y = expf(v); break;
    case 5: y = tri_exp(v); break;
  }
  out[i] = y;
}

// l2 sums of q and k as fla's step takes them (both in-lane pairings), one warp per row
__global__ void k_l2(const bf16* q, float* out0, float* out1) {
  __shared__ float x[256];
  for (int j = threadIdx.x; j < 256; j += 32) x[j] = f(q[blockIdx.x * 256 + j]);
  __syncwarp();
  float a = fla_sumsq_256<true>(x), b = fla_sumsq_256<false>(x);
  if (threadIdx.x == 0) {
    out0[blockIdx.x] = a;
    out1[blockIdx.x] = b;
  }
}

// fla's recurrent step for one value head per block (its own query and key), 128 threads (one
// per state column), over the nonzero rows of q and k in tree order
// VARIANT 1 and 2 are wrong on purpose (sums in row order: the l2 norms, the dot products), to
// show that the test tells the orders apart
__device__ float seq_dot(int n, const int* rows, const float* h, const float* w) {
  float s = 0.0f;
  for (int a = 0; a < n; a++) s = __fmaf_rn(h[a], w[a], s);
  return s;
}

template <bool FMA_ODD, int VARIANT>
__global__ void k_rec(const bf16* q, const bf16* k, const bf16* v, const bf16* beta,
                      const float* g, float* h, bf16* o, float scale) {
  const int hv = blockIdx.x, j = threadIdx.x;
  __shared__ float xq[256], xk[256], sums[2];
  __shared__ int nq, nk, rq[256], rk[256], pq[256], pk[256];
  __shared__ float wq[256], wk[256];
  for (int i = j; i < 256; i += 128) {
    xq[i] = f(q[hv * 256 + i]);
    xk[i] = f(k[hv * 256 + i]);
  }
  __syncthreads();
  if (j < 32) {
    float s = fla_sumsq_256<FMA_ODD>(xq);
    if (j == 0) sums[0] = s;
  } else if (j < 64) {
    float s = fla_sumsq_256<FMA_ODD>(xk);
    if (j == 32) sums[1] = s;
  }
  __syncthreads();
  if (VARIANT == 1 && j == 0) {
    float a = 0.0f, b = 0.0f;
    for (int i = 0; i < 256; i++) {
      a = __fmaf_rn(xq[i], xq[i], a);
      b = __fmaf_rn(xk[i], xk[i], b);
    }
    sums[0] = a;
    sums[1] = b;
  }
  __syncthreads();
  if (j == 0) {
    float dq = tri_sqrt(__fadd_rn(sums[0], 1e-6f)), dk = tri_sqrt(__fadd_rn(sums[1], 1e-6f));
    int a = 0, b = 0;
    for (int p = 0; p < 256; p++) {
      int r = tree_row(p);
      if (xq[r] != 0.0f) {
        rq[a] = r;
        pq[a] = p;
        wq[a++] = __fmul_rn(tri_div(xq[r], dq), scale);
      }
      if (xk[r] != 0.0f) {
        rk[b] = r;
        pk[b] = p;
        wk[b++] = tri_div(xk[r], dk);
      }
    }
    nq = a;
    nk = b;
  }
  __syncthreads();
  float* hh = h + (size_t)hv * 256 * 128;
  float e = g ? tri_exp(g[hv]) : 1.0f;
  if (g)
    for (int i = 0; i < 256; i++) hh[i * 128 + j] = __fmul_rn(hh[i * 128 + j], e);
  float hk[256];
  for (int a = 0; a < nk; a++) hk[a] = hh[rk[a] * 128 + j];
  float s = VARIANT == 2 ? seq_dot(nk, rk, hk, wk) : fla_dot<256>(nk, pk, hk, wk);
  float vn = __fmul_rn(__fsub_rn(f(v[hv * 128 + j]), s), f(beta[hv]));
  for (int a = 0; a < nk; a++) hh[rk[a] * 128 + j] = __fmaf_rn(wk[a], vn, hk[a]);
  for (int a = 0; a < nq; a++) hk[a] = hh[rq[a] * 128 + j];
  o[hv * 128 + j] = rb(VARIANT == 2 ? seq_dot(nq, rq, hk, wq) : fla_dot<256>(nq, pq, hk, wq));
}

extern "C" {
void t_norm2560(const float* x, float* mean, float* r, float eps, int rows, cudaStream_t st) {
  k_norm2560<<<rows, 512, 0, st>>>(x, mean, r, eps);
}
void t_gated(const bf16* o, const bf16* z, const bf16* w, bf16* out, float eps, int rows,
             cudaStream_t st) {
  k_gated<<<rows, 32, 0, st>>>(o, z, w, out, eps);
}
void t_elem(const float* x, float* out, int n, int which, cudaStream_t st) {
  k_elem<<<(n + 255) / 256, 256, 0, st>>>(x, out, n, which);
}
void t_l2(const bf16* q, float* out0, float* out1, int rows, cudaStream_t st) {
  k_l2<<<rows, 32, 0, st>>>(q, out0, out1);
}
void t_rec(const bf16* q, const bf16* k, const bf16* v, const bf16* beta, const float* g,
           float* h, bf16* o, float scale, int nh, int variant, cudaStream_t st) {
  switch (variant) {
    case 0: k_rec<true, 0><<<nh, 128, 0, st>>>(q, k, v, beta, g, h, o, scale); break;
    case 3: k_rec<false, 0><<<nh, 128, 0, st>>>(q, k, v, beta, g, h, o, scale); break;
    case 1: k_rec<true, 1><<<nh, 128, 0, st>>>(q, k, v, beta, g, h, o, scale); break;
    case 2: k_rec<true, 2><<<nh, 128, 0, st>>>(q, k, v, beta, g, h, o, scale); break;
  }
}
int t_last_error() { return (int)cudaGetLastError(); }
}
