// Recipe 01: Flash Attention — Forward Pass
//
// Standard attention computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// This requires materializing the full N×N attention matrix, using
// O(N^2) memory and O(N^2 * d) memory bandwidth.
//
// Flash Attention (Dao et al., 2022) computes the SAME result without
// ever materializing the full attention matrix.  It tiles Q, K, V into
// blocks and uses the online softmax trick to accumulate results in a
// single pass, using only O(N) extra memory.
//
// The key insight: softmax can be computed incrementally.  As we process
// each K/V block, we track the running maximum and running sum, and
// rescale previous partial results when a new maximum is found.
//
// This kernel implements the forward pass of Flash Attention for a
// single head, single batch element.  It demonstrates the algorithm
// clearly without the complexity of multi-head/multi-batch dispatch.
//
// Memory: O(N * d) for Q, K, V, O — no N×N attention matrix.
// Compute: O(N^2 * d) — same as standard attention.
// IO: O(N^2 * d / M) where M = shared memory size — the real win.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t e = (err);                                                     \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Tile sizes
const int BR = 32; // Q tile rows (queries processed per block)
const int BC = 32; // K/V tile cols (keys processed per inner loop step)

// ---------------------------------------------------------------------------
// Naive attention (reference) — materializes the full N×N matrix
// ---------------------------------------------------------------------------
__global__ void attention_naive(const float *Q, const float *K, const float *V,
                                float *O, int N, int d) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;

  const float scale = 1.0f / sqrtf((float)d);

  // Compute max for numerical stability
  float max_val = -FLT_MAX;
  for (int j = 0; j < N; ++j) {
    float score = 0.0f;
    for (int k = 0; k < d; ++k)
      score += Q[row * d + k] * K[j * d + k];
    score *= scale;
    if (score > max_val) max_val = score;
  }

  // Compute softmax denominator
  float sum_exp = 0.0f;
  for (int j = 0; j < N; ++j) {
    float score = 0.0f;
    for (int k = 0; k < d; ++k)
      score += Q[row * d + k] * K[j * d + k];
    score *= scale;
    sum_exp += expf(score - max_val);
  }

  // Compute output
  for (int k = 0; k < d; ++k) {
    float val = 0.0f;
    for (int j = 0; j < N; ++j) {
      float score = 0.0f;
      for (int kk = 0; kk < d; ++kk)
        score += Q[row * d + kk] * K[j * d + kk];
      score *= scale;
      float attn = expf(score - max_val) / sum_exp;
      val += attn * V[j * d + k];
    }
    O[row * d + k] = val;
  }
}

// ---------------------------------------------------------------------------
// Flash Attention — tiled, O(1) extra memory per query row
// ---------------------------------------------------------------------------
__global__ void flash_attention_fwd(const float *Q, const float *K,
                                    const float *V, float *O, int N, int d) {
  // Each block handles BR query rows
  int q_start = blockIdx.x * BR;

  extern __shared__ float smem[];
  // Shared memory layout:
  //   q_tile: BR × d
  //   k_tile: BC × d
  //   v_tile: BC × d
  //   scores: BR × BC
  float *q_tile = smem;
  float *k_tile = q_tile + BR * d;
  float *v_tile = k_tile + BC * d;
  float *scores = v_tile + BC * d;

  // Per-row online softmax state (in registers via shared mem)
  // m_i: running max, l_i: running sum of exp
  float *m_i = scores + BR * BC;
  float *l_i = m_i + BR;

  int tid = threadIdx.x;
  int num_threads = blockDim.x;

  // Initialize running max and sum
  for (int i = tid; i < BR; i += num_threads) {
    m_i[i] = -FLT_MAX;
    l_i[i] = 0.0f;
  }

  // Load Q tile: q_tile[BR][d]
  for (int i = tid; i < BR * d; i += num_threads) {
    int r = i / d, c = i % d;
    int global_r = q_start + r;
    q_tile[r * d + c] = (global_r < N) ? Q[global_r * d + c] : 0.0f;
  }

  // Initialize output accumulator to zero
  for (int i = tid; i < BR * d; i += num_threads) {
    int r = i / d, c = i % d;
    int global_r = q_start + r;
    if (global_r < N) O[global_r * d + c] = 0.0f;
  }

  __syncthreads();

  float scale = 1.0f / sqrtf((float)d);

  // Iterate over K/V blocks
  for (int kv_start = 0; kv_start < N; kv_start += BC) {
    // Load K tile: k_tile[BC][d]
    for (int i = tid; i < BC * d; i += num_threads) {
      int r = i / d, c = i % d;
      int global_r = kv_start + r;
      k_tile[r * d + c] = (global_r < N) ? K[global_r * d + c] : 0.0f;
    }

    // Load V tile: v_tile[BC][d]
    for (int i = tid; i < BC * d; i += num_threads) {
      int r = i / d, c = i % d;
      int global_r = kv_start + r;
      v_tile[r * d + c] = (global_r < N) ? V[global_r * d + c] : 0.0f;
    }

    __syncthreads();

    // Compute scores: S = Q_tile @ K_tile^T * scale
    // scores[BR][BC]
    for (int i = tid; i < BR * BC; i += num_threads) {
      int r = i / BC, c = i % BC;
      float sum = 0.0f;
      for (int k = 0; k < d; ++k)
        sum += q_tile[r * d + k] * k_tile[c * d + k];
      scores[r * BC + c] = sum * scale;

      // Mask out-of-bounds keys
      if (kv_start + c >= N) scores[r * BC + c] = -FLT_MAX;
    }

    __syncthreads();

    // Online softmax update + output accumulation
    // For each query row, update m_i, l_i, and O
    for (int r = tid; r < BR; r += num_threads) {
      if (q_start + r >= N) continue;

      // Find new block maximum
      float m_new = m_i[r];
      for (int c = 0; c < BC; ++c)
        if (scores[r * BC + c] > m_new) m_new = scores[r * BC + c];

      // Rescale previous accumulator: O *= exp(m_old - m_new)
      float rescale = expf(m_i[r] - m_new);
      for (int k = 0; k < d; ++k)
        O[(q_start + r) * d + k] *= rescale;

      // Rescale previous sum
      float l_new = l_i[r] * rescale;

      // Add new block's contribution
      for (int c = 0; c < BC; ++c) {
        float p = expf(scores[r * BC + c] - m_new);
        l_new += p;
        for (int k = 0; k < d; ++k)
          O[(q_start + r) * d + k] += p * v_tile[c * d + k];
      }

      m_i[r] = m_new;
      l_i[r] = l_new;
    }

    __syncthreads();
  }

  // Final normalization: O /= l_i
  for (int i = tid; i < BR * d; i += num_threads) {
    int r = i / d, c = i % d;
    int global_r = q_start + r;
    if (global_r < N && l_i[r] > 0.0f)
      O[global_r * d + c] /= l_i[r];
  }
}

// ---------------------------------------------------------------------------
// Benchmark and verification
// ---------------------------------------------------------------------------
void randomize(float *h, int n) {
  for (int i = 0; i < n; ++i)
    h[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
}

void verify(const float *ref, const float *test, int n, const char *label) {
  float max_err = 0.0f;
  double sum_err = 0.0;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    sum_err += diff;
    if (diff > max_err) max_err = diff;
  }
  float avg_err = (float)(sum_err / n);
  printf("  %-35s max=%.6f avg=%.8f %s\n", label, max_err, avg_err,
         max_err < 1e-2f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Recipe 01: Flash Attention (Forward Pass) ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  printf("Shared memory per block: %zu bytes\n\n", prop.sharedMemPerBlock);

  const int N = 2048; // sequence length
  const int d = 64;   // head dimension

  printf("Sequence length: N=%d, Head dim: d=%d\n", N, d);
  printf("Standard attention memory: %.1f MB (N×N matrix)\n",
         (float)N * N * sizeof(float) / (1024 * 1024));
  printf("Flash attention memory: %.1f KB (no N×N matrix)\n",
         (float)(BR + BC) * d * sizeof(float) / 1024);
  printf("Tile sizes: BR=%d, BC=%d\n\n", BR, BC);

  size_t bytes_QKV = (size_t)N * d * sizeof(float);
  float *hQ = (float *)malloc(bytes_QKV);
  float *hK = (float *)malloc(bytes_QKV);
  float *hV = (float *)malloc(bytes_QKV);
  float *hO_ref = (float *)malloc(bytes_QKV);
  float *hO_flash = (float *)malloc(bytes_QKV);

  srand(42);
  randomize(hQ, N * d);
  randomize(hK, N * d);
  randomize(hV, N * d);

  float *dQ, *dK, *dV, *dO;
  CUDA_CHECK(cudaMalloc(&dQ, bytes_QKV));
  CUDA_CHECK(cudaMalloc(&dK, bytes_QKV));
  CUDA_CHECK(cudaMalloc(&dV, bytes_QKV));
  CUDA_CHECK(cudaMalloc(&dO, bytes_QKV));

  CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes_QKV, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK, bytes_QKV, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV, bytes_QKV, cudaMemcpyHostToDevice));

  // --- Naive reference ---
  int naive_threads = 256;
  int naive_blocks = (N + naive_threads - 1) / naive_threads;

  attention_naive<<<naive_blocks, naive_threads>>>(dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hO_ref, dO, bytes_QKV, cudaMemcpyDeviceToHost));

  // Benchmark naive
  int warmup = 3, iters = 10;
  for (int i = 0; i < warmup; ++i)
    attention_naive<<<naive_blocks, naive_threads>>>(dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    attention_naive<<<naive_blocks, naive_threads>>>(dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float naive_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));
  naive_ms /= iters;

  // --- Flash Attention ---
  int flash_blocks = (N + BR - 1) / BR;
  int flash_threads = 128;
  size_t smem_size = (BR * d + BC * d + BC * d + BR * BC + BR + BR) * sizeof(float);
  printf("Flash SMEM per block: %zu bytes\n", smem_size);

  if (smem_size > prop.sharedMemPerBlock) {
    printf("ERROR: Need %zu bytes SMEM but only %zu available\n",
           smem_size, prop.sharedMemPerBlock);
    return 1;
  }

  CUDA_CHECK(cudaMemset(dO, 0, bytes_QKV));
  flash_attention_fwd<<<flash_blocks, flash_threads, smem_size>>>(
      dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hO_flash, dO, bytes_QKV, cudaMemcpyDeviceToHost));

  // Benchmark flash
  for (int i = 0; i < warmup; ++i)
    flash_attention_fwd<<<flash_blocks, flash_threads, smem_size>>>(
        dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    flash_attention_fwd<<<flash_blocks, flash_threads, smem_size>>>(
        dQ, dK, dV, dO, N, d);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float flash_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&flash_ms, start, stop));
  flash_ms /= iters;

  // --- Results ---
  double flops = 2.0 * N * N * d + 2.0 * N * N * d; // QK^T + attn@V
  double naive_tflops = flops / (naive_ms * 1e-3) * 1e-12;
  double flash_tflops = flops / (flash_ms * 1e-3) * 1e-12;

  printf("\n%-25s %8.2f ms  %7.3f TFLOPS\n", "Naive attention:", naive_ms,
         naive_tflops);
  printf("%-25s %8.2f ms  %7.3f TFLOPS  (%.1fx speedup)\n",
         "Flash attention:", flash_ms, flash_tflops, naive_ms / flash_ms);
  printf("\n");

  verify(hO_ref, hO_flash, N * d, "flash vs naive");

  free(hQ); free(hK); free(hV); free(hO_ref); free(hO_flash);
  CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dO));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
