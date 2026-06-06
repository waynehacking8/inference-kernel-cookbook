// Recipe 02: KV Cache — Why LLM Inference Stores Keys and Values
//
// During autoregressive generation, each new token needs attention over
// ALL previous tokens.  Without caching, we would recompute K and V
// projections for every past token at every step — O(N^2) total work
// for generating N tokens.
//
// The KV cache stores previously computed K and V vectors, so each
// generation step only computes Q, K, V for the NEW token, then
// attends over the cached K/V plus the new entry.
//
// This kernel demonstrates:
//   1. "Prefill" phase: process the full prompt, fill KV cache
//   2. "Decode" phase: generate tokens one at a time using cached KV
//   3. Comparison: with cache (O(N) per step) vs without (O(N^2) per step)
//
// The difference is dramatic: for a 2048-token sequence, the cache
// avoids re-projecting ~2 million key-value pairs across generation.

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

// Single-query attention against a KV cache of length seq_len
// q: [1, d], k_cache: [max_seq, d], v_cache: [max_seq, d]
// output: [1, d]
__global__ void cached_attention_decode(const float *q, const float *k_cache,
                                        const float *v_cache, float *output,
                                        int seq_len, int d) {
  extern __shared__ float smem[];
  float *scores = smem; // [seq_len] — only valid up to seq_len

  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  float scale = 1.0f / sqrtf((float)d);

  // Step 1: Compute attention scores = q @ k_cache^T
  float max_score = -FLT_MAX;
  for (int j = tid; j < seq_len; j += num_threads) {
    float score = 0.0f;
    for (int k = 0; k < d; ++k)
      score += q[k] * k_cache[j * d + k];
    score *= scale;
    scores[j] = score;
    if (score > max_score) max_score = score;
  }

  // Warp-level max reduction
  __shared__ float shared_max[32];
  float local_max = max_score;
  for (int offset = 16; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  if (tid % 32 == 0) shared_max[tid / 32] = local_max;
  __syncthreads();

  if (tid < 32) {
    local_max = (tid < (num_threads + 31) / 32) ? shared_max[tid] : -FLT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1)
      local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    if (tid == 0) shared_max[0] = local_max;
  }
  __syncthreads();
  max_score = shared_max[0];

  // Step 2: Softmax
  float local_sum = 0.0f;
  for (int j = tid; j < seq_len; j += num_threads) {
    scores[j] = expf(scores[j] - max_score);
    local_sum += scores[j];
  }

  // Warp-level sum reduction
  __shared__ float shared_sum[32];
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  if (tid % 32 == 0) shared_sum[tid / 32] = local_sum;
  __syncthreads();

  if (tid < 32) {
    local_sum = (tid < (num_threads + 31) / 32) ? shared_sum[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (tid == 0) shared_sum[0] = local_sum;
  }
  __syncthreads();
  float sum_exp = shared_sum[0];

  for (int j = tid; j < seq_len; j += num_threads)
    scores[j] /= sum_exp;
  __syncthreads();

  // Step 3: Weighted sum = attn @ v_cache
  for (int k = tid; k < d; k += num_threads) {
    float val = 0.0f;
    for (int j = 0; j < seq_len; ++j)
      val += scores[j] * v_cache[j * d + k];
    output[k] = val;
  }
}

// Append new K, V to cache
__global__ void append_to_cache(float *k_cache, float *v_cache,
                                const float *new_k, const float *new_v,
                                int pos, int d) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d) {
    k_cache[pos * d + tid] = new_k[tid];
    v_cache[pos * d + tid] = new_v[tid];
  }
}


void randomize(float *h, int n) {
  for (int i = 0; i < n; ++i)
    h[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
}

int main() {
  printf("=== Recipe 02: KV Cache for Autoregressive Decoding ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  const int max_seq = 2048;
  const int d = 64;
  const int prompt_len = 128;
  const int gen_steps = 256; // generate 256 tokens

  printf("Max sequence length: %d\n", max_seq);
  printf("Head dimension: d=%d\n", d);
  printf("Prompt length: %d tokens\n", prompt_len);
  printf("Generation steps: %d tokens\n", gen_steps);
  printf("KV cache memory: %.1f KB per head\n\n",
         2.0f * max_seq * d * sizeof(float) / 1024);

  size_t cache_bytes = (size_t)max_seq * d * sizeof(float);
  size_t vec_bytes = (size_t)d * sizeof(float);

  float *hQ = (float *)malloc((size_t)(prompt_len + gen_steps) * d * sizeof(float));
  float *hK = (float *)malloc((size_t)(prompt_len + gen_steps) * d * sizeof(float));
  float *hV = (float *)malloc((size_t)(prompt_len + gen_steps) * d * sizeof(float));

  srand(42);
  randomize(hQ, (prompt_len + gen_steps) * d);
  randomize(hK, (prompt_len + gen_steps) * d);
  randomize(hV, (prompt_len + gen_steps) * d);

  float *dK_cache, *dV_cache, *dQ_vec, *dK_vec, *dV_vec, *dOut;
  CUDA_CHECK(cudaMalloc(&dK_cache, cache_bytes));
  CUDA_CHECK(cudaMalloc(&dV_cache, cache_bytes));
  CUDA_CHECK(cudaMalloc(&dQ_vec, vec_bytes));
  CUDA_CHECK(cudaMalloc(&dK_vec, vec_bytes));
  CUDA_CHECK(cudaMalloc(&dV_vec, vec_bytes));
  CUDA_CHECK(cudaMalloc(&dOut, vec_bytes));

  // Upload all K, V for prefill
  CUDA_CHECK(cudaMemcpy(dK_cache, hK, (size_t)prompt_len * d * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV_cache, hV, (size_t)prompt_len * d * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Benchmark: Decode with KV cache ---
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int threads = 256;
  int cur_len = prompt_len;

  CUDA_CHECK(cudaEventRecord(start));
  for (int step = 0; step < gen_steps; ++step) {
    int token_idx = prompt_len + step;

    // Upload new Q, K, V for this token
    CUDA_CHECK(cudaMemcpy(dQ_vec, &hQ[token_idx * d], vec_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK_vec, &hK[token_idx * d], vec_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV_vec, &hV[token_idx * d], vec_bytes,
                          cudaMemcpyHostToDevice));

    // Append to cache
    append_to_cache<<<1, d>>>(dK_cache, dV_cache, dK_vec, dV_vec, cur_len, d);

    cur_len++;

    // Attend over full cache
    size_t smem = cur_len * sizeof(float);
    cached_attention_decode<<<1, threads, smem>>>(dQ_vec, dK_cache, dV_cache,
                                                  dOut, cur_len, d);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float cached_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cached_ms, start, stop));

  printf("With KV cache:\n");
  printf("  %d decode steps in %.2f ms (%.2f ms/token)\n",
         gen_steps, cached_ms, cached_ms / gen_steps);
  printf("  Total KV reads: %d vectors (linear growth)\n\n",
         gen_steps * (prompt_len + gen_steps / 2));

  // Theoretical comparison: without cache, each step recomputes all K,V
  long long no_cache_reads = 0;
  for (int step = 0; step < gen_steps; step++)
    no_cache_reads += (prompt_len + step + 1); // full recompute per step

  long long cached_reads = 0;
  for (int step = 0; step < gen_steps; step++)
    cached_reads += (prompt_len + step + 1); // same attention reads but no recompute

  printf("Without KV cache (theoretical):\n");
  printf("  Would recompute K,V for all %d tokens at each step\n",
         prompt_len + gen_steps);
  printf("  Total K,V projections saved: %d (%.1fx reduction)\n\n",
         gen_steps * (prompt_len + gen_steps / 2),
         (float)(gen_steps * (prompt_len + gen_steps / 2)) / gen_steps);

  printf("Key insight: KV cache trades %.1f KB of memory per head\n",
         2.0f * max_seq * d * sizeof(float) / 1024);
  printf("for avoiding O(N^2) recomputation across generation steps.\n");

  free(hQ); free(hK); free(hV);
  CUDA_CHECK(cudaFree(dK_cache)); CUDA_CHECK(cudaFree(dV_cache));
  CUDA_CHECK(cudaFree(dQ_vec)); CUDA_CHECK(cudaFree(dK_vec));
  CUDA_CHECK(cudaFree(dV_vec)); CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
