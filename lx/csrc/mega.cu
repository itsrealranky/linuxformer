// The chip layers' decode as one persistent kernel (lx/mega.py): every CTA loops over the tokens,
// the phases of a token (each layer's GDN, then its MLP, then the output head) separated by grid
// barriers. The argmax and the next token's embedding are taken on the GPU. The arithmetic is
// lx.fast's decode (the stock decode on the nonzero parts), rounding as torch and fla round
// (mk_math.cuh); sums over exact terms are taken in any order.
//
// Residual stream: every CTA keeps the whole of it in shared memory. A phase adds its outputs
// into acc[q % 3] (fp32 atomics; the terms are exact, so their order does not matter); at the
// start of the next phase every CTA rounds them to bf16 and adds them to its rows, as the
// decode's index_add_ of a bf16 matmul does. acc[(q + 1) % 3] is zeroed in phase q.
#include <cuda_runtime.h>

#include "mk_math.cuh"

using namespace mk;

#define NT 512
#define HIDDEN 2560
#define MAXCU 2048
#define HEAD_ROWS 3584
#define KD 256
#define VD 128
#define R 8
#define DBGW (2 * KD + R * VD + 2 * R + 2 * R * VD + 8)

struct Phase {
  int kind;  // 0 GDN, 1 MLP, 2 head
  int layer;
  float eps;
  int cu_off, n_cu;    // the norm's columns (cols, w1)
  int upd_off, n_upd;  // the residual rows this phase adds to (upd_rows)
  int item_off, n_items;  // GDN: its items (one per live key head, or conv channels only)
  int n_units, row_off, d_off;  // MLP: units; gate rows row_off + u, up rows + n_units; down d_off + u
  int f64;  // MLP: gate/up sums in fp64 (rows whose fp32 sum may depend on the order)
  int grp_n;  // > 0: the first of grp_n GDN phases run optimistically together (mega's loop)
  int pair;   // GDN: with the next phase (its layer's MLP) optimistically (mega's loop)
};

// A GDN item's resident blob (shared memory): Item, padded to 64 bytes; float vtab[64] (its
// weights and conv taps); uint64 meta[n_ch]; int wbase[2][16], wcnt[2][16]; uint16 ent[].
// Channels by count, descending: thread t's are t and (if < n_ch) 2 * NT - 1 - t. A round's warp w
// has a block of entries at wbase[round][w], wcnt (a multiple of 4) a lane: entry e of lane l at
// e * 32 + l, code * n_cu + column (T's index; past a channel's count TZERO, a zero).
// meta: count (5 bits) | slot << 19 (11, 0x7ff none) | tap index << 30 (2) | tap code << 32 (6)
// | taps from chans << 38 (1) | conv channel << 39 (15)
#define IT_VTAB 64
#define IT_META 320
#define TZERO 4095
#define UPR 4  // a phase's residual rows: at most UPR * NT
struct Item {
  int idx, ch_off, n_ch, nvh;
  int kh;                      // key head: value heads kh * R + r in the recurrent state
  int beta_row, a_row, z_row;  // CSR rows of r = 0 (a_row -1: the layer does not decay)
  int hv_off;                  // a_neg, dt_bias of r = 0
  int dim_off;                 // out_proj column (dptr) of (r, j) = dim_off + r * VD + j
  int bm_off;                  // nonzero-row bitmaps (8 words per value head), -1 none
  int gw_off;                  // gated norm weight (VD bf16)
  float geps;
  int n_cu, nv;  // the norm's columns; vtab's values (T: nv x n_cu)
};

// An MLP part's resident blob: int a, b, n_ent, 0; uint64 meta[b - a]: first entry (16 bits) | gate
// count << 16 (8) | up count << 24 (8) | residual row << 32 (16) | bf16 down weight << 48;
// uint32 ent[] (column | bf16 weight << 16): each unit's gate row, then its up row.
#define MAXPH 64
#define MAXDENSE 32
#define MAXSPARSE 256

struct Chan {
  int row, ch, slot;  // CSR row of its projection, conv channel, place in the item's q|k|v
  uint32_t w01, w23;  // conv taps (bf16 pairs)
};

struct Params {
  const Phase* phases;
  int n_phases;
  const Chan* chans;  // by item (ch_off + c): the conv taps of channels with more than one
  const unsigned char* blob;  // every CTA's resident data, from res_off[cta], res_len[cta] bytes
  const int* res_off;
  const int* res_len;
  const int* sched;  // [phase * ncta + cta]: the CTA's blob in its resident data, -1 none
  const int* rowptr;
  const uint32_t* nz;  // CSR entries: column (into the phase's cols) | bf16 weight << 16
  const int16_t* cols;
  const float* w1;
  const int16_t* upd_rows;
  const int* dptr;
  const uint32_t* dnz;  // CSC entries: residual row | bf16 weight << 16
  const float* a_neg;
  const bf16* dt_bias;
  const bf16* gnorm_w;
  uint32_t* bitmaps;
  float* const* rs;
  bf16* const* cs;
  int n_layers;
  // the output head: rows split over the CTAs. Dense columns as 2-bit codes into htab, in the
  // resident data at sched (words [c * ceil(head_rows / 16) + k / 16]); the others by column
  int n_dense, n_sparse;
  const int16_t* hdense;  // the dense columns' places in the head's columns
  const int16_t* hsparse;
  const float* htab;
  const int* hptr;   // [cta * (n_sparse + 1) + j]
  const uint32_t* hnz;  // local row | bf16 weight << 16
  int head_rows;     // rows per CTA
  int n_vocab;
  const bf16* embed;  // the token embedding, row offset + t
  int offset;
  // the run
  const int* forced;  // inputs of the first n_forced steps (the host's)
  int n_forced, n_tokens;
  int* preds;
  float* acc;  // 3 x HIDDEN
  unsigned long long* bar;
  float* part_v;
  int* part_i;
  float* dbg;  // if set: the residual stream at each phase's start, at step dbg_tok
  int dbg_tok;
  float* dbg_items;  // if set: each GDN item's q | k | v, beta, g at step dbg_tok (beta if q|k live)
  unsigned long long* prof;  // if set: per phase and CTA, cycles of work and of the whole phase
  uint2* wsave;  // per item channel (chans' index): its window before an optimistic phase
  float* accs;   // 3 x HIDDEN: an optimistic MLP's sums
  int* inputs;   // if set: each step's input token, as embedded (n_tokens)
};

struct Smem {
  float xs[HIDDEN];
  float red[NT];
  float xn[MAXCU];
  union {
    struct {
      float buf[2 * KD + R * VD];  // the item's q | k | v
      union {
        float T[4096];  // the conv rows' products: xn[column] * vtab[code]
        struct {
          float ob[R][VD];  // o (bf16 values)
          float zb[R][VD];  // z, then the gated norm's output
          float qw[KD], kw[KD], qraw[KD], kraw[KD];
          int qrow[KD], qpos[KD], krow[KD], kpos[KD];
        };
      };
      float beta[R], g[R], l2[2];
      int nq, nk, any_o[R];
    } g;
    struct {
      float lg[HEAD_ROWS];
      float2 pt[MAXDENSE][16];  // per nonzero dense column, two rows' products by their codes
      float wv[NT / 32];
      int wi[NT / 32];
      int dact[MAXDENSE], ndact;
    } h;
  } u;
  int tok, wrote, out, outb;
  unsigned long long gmask;
  bf16* cs[64];
  unsigned long long barhi[3];
  Phase phs[MAXPH];
  int sched[MAXPH];
  long long c0, sub[8];
  int nact, act[MAXSPARSE];
  int hd[MAXDENSE], hs[MAXSPARSE];
  float htab[4];
};

#define SMEM_BASE ((sizeof(Smem) + 15) / 16 * 16)

// MK_SUBPROF builds: cycles from the phase's start to checkpoint i, after prof's per-phase pairs
#ifdef MK_SUBPROF
#define SUB(i)                                              \
  do {                                                      \
    if (threadIdx.x == 0) sm.sub[i] = clock64() - sm.c0;    \
  } while (0)
#else
#define SUB(i) \
  do {         \
  } while (0)
#endif

__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// The grid barrier of phase q: counter q % 3 (the next two phases use the others, so its value
// once all have arrived is this phase's alone). Low 40 bits: arrivals; then, mod 2^24, CTAs that
// added to acc (class 0) plus 4096 x those that added to accs (class 1, cls). Returns whether any
// of class 0 did (every thread); sm.outb class 1's.
__device__ __forceinline__ bool grid_sync(Smem& sm, unsigned long long* bar, unsigned long long q,
                                          int cls = 0) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const int j = (int)(q % 3);
    unsigned long long* b = bar + 16 * j;
    const unsigned long long target = (q / 3 + 1) * gridDim.x;
    __threadfence();
    atomicAdd(b, 1ull + (sm.wrote ? 1ull << (cls ? 52 : 40) : 0ull));
    unsigned long long v;
    while (((v = ld_acquire(b)) & ((1ull << 40) - 1)) < target) {
    }
    const unsigned hi = (unsigned)(v >> 40), d = (hi - sm.barhi[j]) & 0xffffffu;
    sm.out = (d & 0xfff) != 0;
    sm.outb = (d >> 12) != 0;
    sm.barhi[j] = hi;
    sm.wrote = 0;
  }
  __syncthreads();
  return sm.out;
}

__device__ __forceinline__ float csr_dot(const Params& p, int row, const float* xn) {
  int a = p.rowptr[row], b = p.rowptr[row + 1];
  float s = 0.0f;
  for (int e = a; e < b; e++) {
    uint32_t w = __ldg(p.nz + e);
    s = __fadd_rn(s, __fmul_rn(xn[w & 0xffff], f((bf16)(w >> 16))));
  }
  return s;
}

// Qwen3_5RMSNorm(x)[cols] (bf16 values) into xn, all threads. A resident blob starts with its norm's
// columns: int n, body (bytes), 0, 0; int16 cols[n] and float w1[n], each padded to 16 bytes.
// The sum of squares is kept while the residual stream is unchanged (dirty: some row changed).
__device__ const unsigned char* norm_blob(Smem& sm, float eps, const unsigned char* blob,
                                          bool& dirty, float& ssum) {
  const int* hd = reinterpret_cast<const int*>(blob);
  const int n = hd[0];
  const int16_t* cols = reinterpret_cast<const int16_t*>(blob + 16);
  const float* w1 = reinterpret_cast<const float*>(blob + 16 + ((2 * n + 15) & ~15));
  if (dirty) {
    ssum = torch_sumsq_2560(sm.xs, sm.red);
    dirty = false;
  }
  const float r = t_rsqrt(__fadd_rn(__fmul_rn(ssum, 1.0f / 2560.0f), eps));
  for (int k = threadIdx.x; k < n; k += NT)
    sm.xn[k] = rbf(__fmul_rn(__fmul_rn(sm.xs[cols[k]], r), w1[k]));
  __syncthreads();
  return blob + hd[1];
}

__device__ __noinline__ float fla_dot_big(int n, const int* pos, const float* h, const float* w) {
  return fla_dot<KD>(n, pos, h, w);
}

// tl.sum(h * w) over the n listed rows of one state column (h_col: the column, row stride VD)
__device__ __forceinline__ float col_dot(int n, const int* rows, const int* pos, const float* w,
                                         const float* h_col) {
  if (n == 0) return 0.0f;
  if (n == 1) return __fmul_rn(h_col[rows[0] * VD], w[0]);
  if (n <= 8) {
    float hv[8];
    for (int a = 0; a < n; a++) hv[a] = h_col[rows[a] * VD];
    return fla_dot<8>(n, pos, hv, w);
  }
  float hv[KD];
  for (int a = 0; a < n; a++) hv[a] = h_col[rows[a] * VD];
  return fla_dot_big(n, pos, hv, w);
}

// a channel's new window (its projection's sum s appended) and, if in q | k | v, its conv output
__device__ __forceinline__ void conv_out(const Params& p, Smem& sm, const Item& it, const float* vt,
                                         int c, uint64_t m, uint2 win, uint2* wp, float s,
                                         bool save) {
  const bf16 P = rb(s);
  const bf16 x0 = (bf16)(win.x >> 16), x1 = (bf16)(win.y & 0xffff), x2 = (bf16)(win.y >> 16);
  if (save) p.wsave[it.ch_off + c] = win;
  *wp = make_uint2((uint32_t)x0 | ((uint32_t)x1 << 16), (uint32_t)x2 | ((uint32_t)P << 16));
  const int slot = (int)((m >> 19) & 0x7ff);
  if (slot == 0x7ff) return;
  float y = 0.0f;
  if ((m >> 38) & 1) {
    const Chan ch = p.chans[it.ch_off + c];
    y = __fadd_rn(y, __fmul_rn(f((bf16)(ch.w01 & 0xffff)), f(x0)));
    y = __fadd_rn(y, __fmul_rn(f((bf16)(ch.w01 >> 16)), f(x1)));
    y = __fadd_rn(y, __fmul_rn(f((bf16)(ch.w23 & 0xffff)), f(x2)));
    y = __fadd_rn(y, __fmul_rn(f((bf16)(ch.w23 >> 16)), f(P)));
  } else {  // one nonzero tap: the other products are zeros (y differs at most in a zero's sign)
    const int ti = (int)((m >> 30) & 3);
    const bf16 xv = ti == 0 ? x0 : ti == 1 ? x1 : ti == 2 ? x2 : P;
    y = __fadd_rn(y, __fmul_rn(vt[(m >> 32) & 63], f(xv)));
  }
  sm.u.g.buf[slot] = rbf(silu(rbf(y)));
}

// gm set: optimistic (the residual may be stale): the windows' old values to wsave, and past the
// conv windows (the only change of a step with q, k all zeros and no decay) only bit spec of gm
// (and wrote) set.
__device__ void gdn_item(const Params& p, Smem& sm, const unsigned char* blob, int L, float* acc,
                         int step, unsigned long long* gm = nullptr, int spec = 0) {
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const Item& it = *reinterpret_cast<const Item*>(blob);
  const float* vt = reinterpret_cast<const float*>(blob + IT_VTAB);
  const uint64_t* meta = reinterpret_cast<const uint64_t*>(blob + IT_META);
  const int* wbase = reinterpret_cast<const int*>(blob + IT_META + 8 * it.n_ch);
  const int* wcnt = wbase + 32;
  const uint16_t* ent = reinterpret_cast<const uint16_t*>(blob + IT_META + 8 * it.n_ch + 256);
  auto& G = sm.u.g;
  bf16* cs = sm.cs[L];
  // channels t and c1 (its count is at most t's): their windows' loads first
  const int c1 = 2 * NT - 1 - t;
  const bool one = t < it.n_ch, two = c1 < it.n_ch;
  const uint64_t m0 = one ? meta[t] : 0, m1 = two ? meta[c1] : 0;
  uint2* wp0 = reinterpret_cast<uint2*>(cs + (size_t)((m0 >> 39) & 0x7fff) * 4);
  uint2* wp1 = reinterpret_cast<uint2*>(cs + (size_t)((m1 >> 39) & 0x7fff) * 4);
  const uint2 win0 = one ? *wp0 : make_uint2(0, 0);
  const uint2 win1 = two ? *wp1 : make_uint2(0, 0);
  for (int k = t; k < 2 * KD + R * VD; k += NT) G.buf[k] = 0.0f;
  if (t < R) G.any_o[t] = 0;
  {
    const int nc = it.n_cu;
    const float inv = 1.0f / (float)nc;  // k / nc for k < 4096: the quotient's fraction is >= 0.5 / nc
    for (int k = t; k < it.nv * nc; k += NT) {
      const int c = (int)(((float)k + 0.5f) * inv);
      G.T[k] = __fmul_rn(sm.xn[k - c * nc], vt[c]);
    }
    if (t == 0) G.T[TZERO] = 0.0f;
  }
  __syncthreads();
  // in_proj_qkv rows, then the conv window and silu, per channel
  {
    const uint16_t* e0 = ent + wbase[warp] + lane;
    const uint16_t* e1 = ent + wbase[16 + warp] + lane;
    const int M0 = wcnt[warp], M1 = wcnt[16 + warp];  // M1 <= M0; zeros past a channel's count
    float s0 = 0.0f, s1 = 0.0f;
    for (int e = 0; e < M0; e += 4) {
      const float a0 = G.T[e0[e * 32]], a1 = G.T[e0[e * 32 + 32]], a2 = G.T[e0[e * 32 + 64]],
                  a3 = G.T[e0[e * 32 + 96]];
      if (e < M1) {
        const float b0 = G.T[e1[e * 32]], b1 = G.T[e1[e * 32 + 32]], b2 = G.T[e1[e * 32 + 64]],
                    b3 = G.T[e1[e * 32 + 96]];
        s1 = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(s1, b0), b1), b2), b3);
      }
      s0 = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(s0, a0), a1), a2), a3);
    }
    if (one) conv_out(p, sm, it, vt, t, m0, win0, wp0, s0, gm != nullptr);
    if (two) conv_out(p, sm, it, vt, c1, m1, win1, wp1, s1, gm != nullptr);
  }
  __syncthreads();
  SUB(1);
  if (it.nvh == 0) return;
  if (t < R) {
    float g = 0.0f;
    if (it.a_row >= 0) {
      float a = rbf(csr_dot(p, it.a_row + t, sm.xn));
      g = __fmul_rn(p.a_neg[it.hv_off + t], softplus(__fadd_rn(a, f(p.dt_bias[it.hv_off + t]))));
    }
    G.g[t] = g;
  }
  int live = 0;
  for (int k = t; k < 2 * KD; k += NT) live |= G.buf[k] != 0.0f;
  const int qk = __syncthreads_or(live);
  if (gm) {
    bool any = qk;
    for (int r = 0; r < R; r++) any |= G.g[r] != 0.0f;
    if (any && t == 0) {
      atomicOr(gm, 1ull << spec);
      sm.wrote = 1;
    }
    return;
  }
  const bool dbg = p.dbg_items && step == p.dbg_tok;
  if (qk && t >= R && t < 2 * R)  // beta is used only with a nonzero k
    G.beta[t - R] = rbf(sigmoid(rbf(csr_dot(p, it.beta_row + t - R, sm.xn))));
  if (dbg) {
    __syncthreads();
    float* d = p.dbg_items + (size_t)it.idx * DBGW;
    for (int k = t; k < 2 * KD + R * VD; k += NT) d[k] = G.buf[k];
    if (t < R) {
      d[2 * KD + R * VD + t] = qk ? G.beta[t] : 0.0f;
      d[2 * KD + R * VD + R + t] = G.g[t];
    }
  }
  int anyg = 0;
  for (int r = 0; r < R; r++) anyg |= G.g[r] != 0.0f;
  SUB(2);
#ifdef MK_SUBPROF
  if (p.prof && t == 0 && (qk || anyg))
    atomicOr(p.prof + (size_t)p.n_phases * gridDim.x * 10 + 2 * step, 1ull << L);
#endif
  if (!qk && !anyg) return;  // fla's step changes nothing: o = 0, the state as it is
  // l2 sums (warps 0, 1); the nonzero rows of q and k in the sum tree's order (warps 2, 3)
  if (warp < 2) {
    float s = fla_sumsq_256<true>(G.buf + warp * KD);
    if (lane == 0) G.l2[warp] = s;
  } else if (warp < 4) {
    const bool isk = warp == 3;
    const float* x = G.buf + (isk ? KD : 0);
    int n = 0;
    for (int base = 0; base < KD; base += 32) {
      int pp = base + lane, row = tree_row(pp);
      float v = x[row];
      unsigned m = __ballot_sync(0xffffffffu, v != 0.0f);
      if (v != 0.0f) {
        int k = n + __popc(m & ((1u << lane) - 1));
        (isk ? G.krow : G.qrow)[k] = row;
        (isk ? G.kpos : G.qpos)[k] = pp;
        (isk ? G.kraw : G.qraw)[k] = v;
      }
      n += __popc(m);
    }
    if (lane == 0) (isk ? G.nk : G.nq) = n;
  }
  __syncthreads();
  {
    const float dq = tri_sqrt(__fadd_rn(G.l2[0], 1e-6f)), dk = tri_sqrt(__fadd_rn(G.l2[1], 1e-6f));
    for (int k = t; k < G.nq; k += NT) G.qw[k] = __fmul_rn(tri_div(G.qraw[k], dq), 0.0625f);
    for (int k = t; k < G.nk; k += NT) G.kw[k] = tri_div(G.kraw[k], dk);
  }
  __syncthreads();
  SUB(3);
  // the delta rule, one state column per thread
  const int nq = G.nq, nk = G.nk;
  for (int r = t >> 7; r < R; r += NT / VD) {
    const int j = t & (VD - 1);
    const float g = G.g[r];
    float o = 0.0f;
    if (qk || g != 0.0f) {
      float* h = p.rs[L] + (size_t)(it.kh * R + r) * KD * VD + j;
      if (g != 0.0f) {  // the nonzero rows (the others stay zero), four loads in flight
        const float e = tri_exp(g);
        const uint4* bq = reinterpret_cast<const uint4*>(p.bitmaps + it.bm_off + r * 8);
        const uint4 b0 = bq[0], b1 = bq[1];
        const uint32_t bw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int w = 0; w < 8; w++) {
          uint32_t m = bw[w];
          float* hb = h + w * 32 * VD;
          while (m) {
            int ix[4];
#pragma unroll
            for (int a = 0; a < 4; a++) {
              ix[a] = m ? __ffs(m) - 1 : -1;
              m &= m - 1;
            }
            float v[4];
#pragma unroll
            for (int a = 0; a < 4; a++) v[a] = ix[a] >= 0 ? hb[ix[a] * VD] : 0.0f;
#pragma unroll
            for (int a = 0; a < 4; a++)
              if (ix[a] >= 0) hb[ix[a] * VD] = __fmul_rn(v[a], e);
          }
        }
      }
      if (nk) {
        float s = col_dot(nk, G.krow, G.kpos, G.kw, h);
        float vn = __fmul_rn(__fsub_rn(G.buf[2 * KD + r * VD + j], s), G.beta[r]);
        if (p.dbg_items && step == p.dbg_tok) {
          float* d = p.dbg_items + (size_t)it.idx * DBGW + 2 * KD + R * VD + 2 * R;
          d[r * VD + j] = s;
          d[R * VD + r * VD + j] = vn;
          if (r == 0 && j == 0) {
            d[2 * R * VD] = nk;
            d[2 * R * VD + 1] = nq;
            d[2 * R * VD + 2] = G.kw[0];
            d[2 * R * VD + 3] = G.l2[1];
            d[2 * R * VD + 4] = G.krow[0];
          }
        }
        for (int a = 0; a < nk; a++) {
          float* hp = h + G.krow[a] * VD;
          *hp = __fmaf_rn(G.kw[a], vn, *hp);
        }
      }
      o = rbf(col_dot(nq, G.qrow, G.qpos, G.qw, h));
    }
    G.ob[r][j] = o;
    if (o != 0.0f) G.any_o[r] = 1;
  }
  __syncthreads();
  SUB(4);
  if (it.bm_off >= 0 && t < R && (qk || G.g[t] != 0.0f)) {
    uint32_t* bm = p.bitmaps + it.bm_off + t * 8;
    for (int a = 0; a < nk; a++) bm[G.krow[a] >> 5] |= 1u << (G.krow[a] & 31);
  }
  // value heads with output: in_proj_z, the gated norm, out_proj
  if (!__syncthreads_or(t < R ? G.any_o[t] : 0)) return;
  SUB(5);
  for (int k = t; k < R * VD; k += NT)
    if (G.any_o[k >> 7]) G.zb[k >> 7][k & (VD - 1)] = rbf(csr_dot(p, it.z_row + k, sm.xn));
  __syncthreads();
  SUB(6);
  if (warp < R && G.any_o[warp]) {
    const int r = warp;
    float s = __shfl_sync(0xffffffffu, torch_sumsq_128(G.ob[r]), 0);
    float rr = t_rsqrt(__fadd_rn(__fmul_rn(s, 1.0f / 128.0f), it.geps));
    const bf16* gw = p.gnorm_w + it.gw_off;
    for (int j = lane; j < VD; j += 32) {
      bf16 u = mul_b(gw[j], rb(__fmul_rn(G.ob[r][j], rr)));
      G.zb[r][j] = rbf(__fmul_rn(f(u), silu(G.zb[r][j])));
    }
  }
  __syncthreads();
  for (int k = t; k < R * VD; k += NT) {
    const int r = k >> 7, j = k & (VD - 1);
    if (!G.any_o[r]) continue;
    const float on = G.zb[r][j];
    if (on == 0.0f) continue;
    const int d = it.dim_off + k;
    for (int e = p.dptr[d]; e < p.dptr[d + 1]; e++) {
      uint32_t w = __ldg(p.dnz + e);
      atomicAdd(acc + (w & 0xffff), __fmul_rn(on, f((bf16)(w >> 16))));
      sm.wrote = 1;
    }
  }
  __syncthreads();
  SUB(7);
}

__device__ __forceinline__ float ent_dot(const uint32_t* ent, int n, const float* xn, bool f64) {
  if (f64) {
    double s = 0.0;
    for (int e = 0; e < n; e++) {
      const uint32_t w = ent[e];
      s = __dadd_rn(s, __dmul_rn((double)xn[w & 0xffff], (double)f((bf16)(w >> 16))));
    }
    return __double2float_rn(s);
  }
  float s = 0.0f;
  for (int e = 0; e < n; e++) {
    const uint32_t w = ent[e];
    s = __fadd_rn(s, __fmul_rn(xn[w & 0xffff], f((bf16)(w >> 16))));
  }
  return s;
}

__device__ void mlp_part(const Params& p, Smem& sm, const Phase& ph, const unsigned char* blob,
                         float* acc) {
  const int* hd = reinterpret_cast<const int*>(blob);
  const int n = hd[1] - hd[0];
  const uint64_t* meta = reinterpret_cast<const uint64_t*>(blob + 16);
  const uint32_t* ent = reinterpret_cast<const uint32_t*>(blob + 16 + 8 * n);
  for (int u = threadIdx.x; u < n; u += NT) {
    const uint64_t m = meta[u];
    const int e0 = (int)(m & 0xffff), ng = (int)((m >> 16) & 0xff), nu = (int)((m >> 24) & 0xff);
    const float g = rbf(ent_dot(ent + e0, ng, sm.xn, ph.f64));
    const float up = rbf(ent_dot(ent + e0 + ng, nu, sm.xn, ph.f64));
    const float act = rbf(__fmul_rn(rbf(silu(g)), up));
    if (act == 0.0f) continue;
    atomicAdd(acc + ((m >> 32) & 0xffff), __fmul_rn(act, f((bf16)(m >> 48))));
    sm.wrote = 1;
  }
}

// the output head over this CTA's rows: its bf16 logits' first maximum into part_v/part_i
__device__ void head_part(const Params& p, Smem& sm, const uint32_t* codes) {
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  auto& H = sm.u.h;
  const int r0 = blockIdx.x * p.head_rows;
  const int nr = max(0, min(p.head_rows, p.n_vocab - r0));
  const int W = (p.head_rows + 15) >> 4, nd = p.n_dense, ns = p.n_sparse;
  if (t == 0) sm.nact = 0;
  // the dense columns with x != 0, in order (a zero column adds only zeros)
  if (warp == 0) {
    const float x = lane < nd ? sm.xn[sm.hd[lane]] : 0.0f;
    const unsigned m = __ballot_sync(0xffffffffu, x != 0.0f);
    if (x != 0.0f) H.dact[__popc(m & ((1u << lane) - 1))] = lane;
    if (lane == 0) H.ndact = __popc(m);
  }
  __syncthreads();
  const int na = H.ndact;
  for (int q = t; q < na * 16; q += NT) {
    const float x = sm.xn[sm.hd[H.dact[q >> 4]]];
    H.pt[q >> 4][q & 15] =
        make_float2(__fmul_rn(x, sm.htab[q & 3]), __fmul_rn(x, sm.htab[(q >> 2) & 3]));
  }
  __syncthreads();
  // rows 8t .. 8t + 7 (the codes' word t / 2, half t % 2); per row the columns in order
  if (8 * t < nr) {
    float a[8];
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = 0.0f;
    const int sh = (t & 1) * 16;
    for (int k = 0; k < na; k++) {
      const uint32_t wd = codes[H.dact[k] * W + (t >> 1)] >> sh;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float2 v = H.pt[k][(wd >> (4 * j)) & 15];
        a[2 * j] = __fadd_rn(a[2 * j], v.x);
        a[2 * j + 1] = __fadd_rn(a[2 * j + 1], v.y);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; i++)
      if (8 * t + i < nr) H.lg[8 * t + i] = a[i];
  }
  __syncthreads();
  if (t < ns && sm.xn[sm.hs[t]] != 0.0f) sm.act[atomicAdd(&sm.nact, 1)] = t;
  __syncthreads();
  SUB(1);
  const int* hp = p.hptr + (size_t)blockIdx.x * (ns + 1);
  for (int a = 0; a < sm.nact; a++) {
    const int j = sm.act[a];
    const float x = sm.xn[sm.hs[j]];
    for (int e = hp[j] + t; e < hp[j + 1]; e += NT) {
      uint32_t w = __ldg(p.hnz + e);
      atomicAdd(&H.lg[w & 0xffff], __fmul_rn(x, f((bf16)(w >> 16))));
    }
  }
  __syncthreads();
  SUB(2);
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int k = t; k < nr; k += NT) {
    float v = rbf(H.lg[k]);
    if (v > bv) {
      bv = v;
      bi = r0 + k;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, bv, off);
    int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if ((t & 31) == 0) {
    H.wv[t >> 5] = bv;
    H.wi[t >> 5] = bi;
  }
  __syncthreads();
  if (t == 0) {
    for (int w = 1; w < NT / 32; w++)
      if (H.wv[w] > bv || (H.wv[w] == bv && H.wi[w] < bi)) {
        bv = H.wv[w];
        bi = H.wi[w];
      }
    __stcg(p.part_v + blockIdx.x, bv);
    __stcg(p.part_i + blockIdx.x, bi);
  }
}

// an optimistic gdn_item's windows back (each thread its own channels, as it saved them)
__device__ void gdn_rollback(const Params& p, Smem& sm, const unsigned char* blob, int L) {
  const unsigned char* body = blob + reinterpret_cast<const int*>(blob)[1];
  const Item& it = *reinterpret_cast<const Item*>(body);
  const uint64_t* meta = reinterpret_cast<const uint64_t*>(body + IT_META);
  bf16* cs = sm.cs[L];
  const int t = threadIdx.x, c1 = 2 * NT - 1 - t;
  if (t < it.n_ch)
    *reinterpret_cast<uint2*>(cs + (size_t)((meta[t] >> 39) & 0x7fff) * 4) = p.wsave[it.ch_off + t];
  if (c1 < it.n_ch)
    *reinterpret_cast<uint2*>(cs + (size_t)((meta[c1] >> 39) & 0x7fff) * 4) =
        p.wsave[it.ch_off + c1];
}

__device__ void load_embed(const Params& p, Smem& sm, int tok) {
  const bf16* e = p.embed + (size_t)(tok + p.offset) * HIDDEN;
  for (int k = threadIdx.x; k < HIDDEN; k += NT) sm.xs[k] = f(e[k]);
  __syncthreads();
}

__global__ void __launch_bounds__(NT, 1) mega(Params p) {
  extern __shared__ __align__(16) unsigned char dsm[];
  Smem& sm = *reinterpret_cast<Smem*>(dsm);
  unsigned char* res = dsm + SMEM_BASE;
  const int t = threadIdx.x;
  {
    const uint4* src = reinterpret_cast<const uint4*>(p.blob + p.res_off[blockIdx.x]);
    uint4* dst = reinterpret_cast<uint4*>(res);
    for (int k = t; k < p.res_len[blockIdx.x] / 16; k += NT) dst[k] = src[k];
    for (int k = t; k < p.n_phases; k += NT) sm.sched[k] = p.sched[k * gridDim.x + blockIdx.x];
    for (int k = t; k < p.n_dense; k += NT) sm.hd[k] = p.hdense[k];
    for (int k = t; k < p.n_sparse; k += NT) sm.hs[k] = p.hsparse[k];
    if (t < 4) sm.htab[t] = p.htab[t];
    const int* ps = reinterpret_cast<const int*>(p.phases);
    int* pd = reinterpret_cast<int*>(sm.phs);
    for (int k = t; k < p.n_phases * (int)(sizeof(Phase) / 4); k += NT) pd[k] = ps[k];
    if (t < 3) sm.barhi[t] = 0;
    if (t < 64) sm.cs[t] = t < p.n_layers ? p.cs[t] : nullptr;
    if (t == 0) sm.wrote = 0;
  }
  unsigned long long q = 0;  // phases run (acc buffer and barrier counter q % 3)
  const float* prev_acc = p.acc;  // the last phase's sums
  int urow[UPR];  // the last phase's residual rows (thread t's: t + u NT), -1 none
  for (int u = 0; u < UPR; u++) urow[u] = -1;
  bool prev_out = false;  // whether the last phase added to acc
  bool dirty = true;
  float ssum = 0.0f;
  load_embed(p, sm, p.forced[0]);
  if (p.inputs && blockIdx.x == 0 && t == 0) p.inputs[0] = p.forced[0];
  for (int i = 0; i < p.n_tokens; i++) {
    // a phase: the last one's sums into the residual, then this one's work, then the barrier
    auto begin = [&](int k) {
      int chg = 0;
      if (prev_out) {  // else the last sums are all zeros: x + 0 is x (but for a zero's sign)
        float y[UPR];
        for (int u = 0; u < UPR; u++)
          y[u] = urow[u] >= 0 ? __ldcg(prev_acc + urow[u]) : 0.0f;
        for (int u = 0; u < UPR; u++)
          if (urow[u] >= 0) {
            sm.xs[urow[u]] = rbf(__fadd_rn(sm.xs[urow[u]], rbf(y[u])));
            chg |= y[u] != 0.0f;
          }
      }
      {
        const int z0 = blockIdx.x * ((HIDDEN + gridDim.x - 1) / gridDim.x);
        const int z1 = min(HIDDEN, z0 + (HIDDEN + gridDim.x - 1) / gridDim.x);
        float* zb = p.acc + ((q + 1) % 3) * HIDDEN;
        float* zs = p.accs + ((q + 1) % 3) * HIDDEN;
        for (int a = z0 + t; a < z1; a += NT) {
          zb[a] = 0.0f;
          zs[a] = 0.0f;
        }
        if (blockIdx.x == 0 && t == 0) p.bar[64 + 16 * ((q + 1) % 3)] = 0;
      }
      if (__syncthreads_or(chg)) dirty = true;
#ifdef MK_SUBPROF
      if (t == 0) {
        for (int a = 0; a < 8; a++) sm.sub[a] = 0;
        sm.c0 = clock64();
      }
#endif
      if (p.dbg && i == p.dbg_tok && blockIdx.x == 0)
        for (int a = t; a < HIDDEN; a += NT) p.dbg[k * HIDDEN + a] = sm.xs[a];
    };
    auto end = [&](int k, long long c0, int upd_off, int n_upd, int cls = 0) {
      long long c1 = clock64();
#ifdef MK_SUBPROF
      if (p.prof && t == 0)
        for (int a = 0; a < 8; a++)
          p.prof[(size_t)p.n_phases * gridDim.x * 2 + ((size_t)k * gridDim.x + blockIdx.x) * 8 + a] +=
              sm.sub[a];
#endif
      for (int u = 0; u < UPR; u++)  // loaded while waiting
        urow[u] = t + u * NT < n_upd ? p.upd_rows[upd_off + t + u * NT] : -1;
      prev_out = grid_sync(sm, p.bar, q, cls);
#ifdef MK_SUBPROF
      if (p.prof && t == 0 && blockIdx.x == 0 && prev_out)
        p.prof[(size_t)p.n_phases * gridDim.x * 10 + 2 * i + 1] |= 1ull << k;
#endif
      if (p.prof && t == 0) {
        long long c2 = clock64();
        unsigned long long* pr = p.prof + ((size_t)k * gridDim.x + blockIdx.x) * 2;
        pr[0] += (unsigned long long)(c1 - c0);
        pr[1] += (unsigned long long)(c2 - c0);
      }
      prev_acc = p.acc + (q % 3) * HIDDEN;
      q++;
    };
    auto normal = [&](int k) {
      const Phase& ph = sm.phs[k];
      begin(k);
      const long long c0 = clock64();
      float* acc = p.acc + (q % 3) * HIDDEN;
      const int so = sm.sched[k];
      if (so >= 0) {
        const unsigned char* body = norm_blob(sm, ph.eps, res + so, dirty, ssum);
        SUB(0);
        if (ph.kind == 0)
          gdn_item(p, sm, body, ph.layer, acc, i);
        else if (ph.kind == 1)
          mlp_part(p, sm, ph, body, acc);
        else
          head_part(p, sm, reinterpret_cast<const uint32_t*>(body));
      }
      end(k, c0, ph.upd_off, ph.n_upd);
    };
    for (int k = 0; k < p.n_phases;) {
      if (sm.phs[k].pair) {
        // GDN phase k and its layer's MLP (phase k + 1) together, the MLP (its sums to accs)
        // assuming the GDN step leaves the residual as it is; if it does not, the MLP again
        // (no CTA has both)
        const Phase& ph = sm.phs[k];
        const Phase& pm = sm.phs[k + 1];
        begin(k);
        const long long c0 = clock64();
        float* ga = p.acc + (q % 3) * HIDDEN;
        float* ma = p.accs + (q % 3) * HIDDEN;
        const int so = sm.sched[k], som = sm.sched[k + 1];
        if (so >= 0) {
          const unsigned char* body = norm_blob(sm, ph.eps, res + so, dirty, ssum);
          gdn_item(p, sm, body, ph.layer, ga, i);
        } else if (som >= 0) {
          const unsigned char* body = norm_blob(sm, pm.eps, res + som, dirty, ssum);
          mlp_part(p, sm, pm, body, ma);
        }
        end(k, c0, pm.upd_off, pm.n_upd, so < 0 ? 1 : 0);
        if (!prev_out) {
          prev_out = sm.outb;
          prev_acc = ma;
          k += 2;
        } else {
          for (int u = 0; u < UPR; u++)
            urow[u] = t + u * NT < ph.n_upd ? p.upd_rows[ph.upd_off + t + u * NT] : -1;
          k += 1;
        }
        continue;
      }
      const int gn = sm.phs[k].grp_n;
      if (gn <= 0) {
        normal(k++);
        continue;
      }
      // GDN phases k .. k1 - 1 on the residual as it is, assuming each before leaves it so (a
      // step changing only its conv windows). If one does more, phases from the first such j
      // are undone (their windows back), j run alone, the rest again so.
      const int k1 = k + gn;
      for (int lo = k; lo < k1;) {
        begin(lo);
        const long long c0 = clock64();
        unsigned long long* gm = p.bar + 64 + 16 * (q % 3);
        for (int kk = lo; kk < k1; kk++) {
          const int so = sm.sched[kk];
          if (so >= 0) {
            const unsigned char* body = norm_blob(sm, sm.phs[kk].eps, res + so, dirty, ssum);
            gdn_item(p, sm, body, sm.phs[kk].layer, nullptr, i, gm, kk - k);
          }
        }
        end(lo, c0, 0, 0);
        if (!prev_out) break;
        prev_out = false;  // the barrier's flag was gm's
        if (t == 0) sm.gmask = ld_acquire(gm);
        __syncthreads();
#ifdef MK_SUBPROF
        if (p.prof && t == 0 && blockIdx.x == 0)
          p.prof[(size_t)p.n_phases * gridDim.x * 10 + 2 * i] += 1ull << 48;
#endif
        const int j = k + __ffsll((long long)sm.gmask) - 1;
        for (int kk = j; kk < k1; kk++) {
          const int so = sm.sched[kk];
          if (so >= 0) gdn_rollback(p, sm, res + so, sm.phs[kk].layer);
        }
        normal(j);
        lo = j + 1;
      }
      k = k1;
    }
    // the head's partial maxima: every CTA takes the first maximum (rows ascend with the CTA)
    if (t < 32) {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
      for (int c = t; c < (int)gridDim.x; c += 32) {
        float v = __ldcg(p.part_v + c);
        int ii = __ldcg(p.part_i + c);
        if (v > bv || (v == bv && ii < bi)) {
          bv = v;
          bi = ii;
        }
      }
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (t == 0) sm.tok = bi;
    }
    __syncthreads();
    const int pred = sm.tok;
    if (blockIdx.x == 0 && t == 0) p.preds[i] = pred;
    if (i + 1 < p.n_tokens) {
      const int nt = i + 1 < p.n_forced ? p.forced[i + 1] : pred;
      if (p.inputs && blockIdx.x == 0 && t == 0) p.inputs[i + 1] = nt;
      load_embed(p, sm, nt);
    }
    dirty = true;
    __syncthreads();
  }
}

extern "C" {
int mk_smem_base() { return (int)SMEM_BASE; }

int mk_run(const Params* prm, int ncta, int smem, cudaStream_t st) {
  Params p = *prm;
  void* args[] = {&p};
  cudaError_t e = cudaFuncSetAttribute(mega, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e) return (int)e;
  e = cudaLaunchCooperativeKernel((void*)mega, dim3(ncta), dim3(NT), args, smem, st);
  return (int)e;
}

int mk_max_ctas(int smem) {
  int n = 0, dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (cudaFuncSetAttribute(mega, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)) return 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mega, NT, smem);
  return n * sms;
}

int mk_max_smem() {
  int dev = 0, v = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return v;
}

int mk_last_error() { return (int)cudaGetLastError(); }
}
