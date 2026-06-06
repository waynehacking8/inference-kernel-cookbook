// Recipe 03: Paged Attention — Virtual Memory for KV Caches
//
// Recipe 02's KV cache uses a contiguous buffer per sequence.  When
// serving many sequences with different lengths, this wastes memory:
// short sequences reserve max_seq_len slots, and finished sequences
// leave fragmented holes.
//
// PagedAttention (Kwon et al., 2023 — the core idea behind vLLM)
// solves this by splitting the KV cache into fixed-size PAGES
// (blocks), like an OS virtual memory system.  Each sequence has a
// page table mapping logical positions to physical block addresses.
//
// Benefits:
//   1. Near-zero internal fragmentation (only last page is partial)
//   2. Dynamic allocation: grow/shrink per sequence as needed
//   3. Enables techniques like prefix caching (shared pages)
//   4. Memory utilization goes from ~50% to ~95%+ in production
//
// This kernel implements the core paged attention decode operation:
// given a query vector and a page table, attend over KV blocks
// scattered across physical memory.

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

const int BLOCK_SIZE_KV = 16; // tokens per KV page

// Paged attention decode kernel
// q:           [d]           — single query vector
// k_cache:     [num_blocks, BLOCK_SIZE_KV, d] — all K pages in physical memory
// v_cache:     [num_blocks, BLOCK_SIZE_KV, d] — all V pages in physical memory
// page_table:  [max_pages]   — maps logical block idx to physical block idx
// output:      [d]
// seq_len:     actual sequence length (may not fill last page)
__global__ void paged_attention_decode(const float *q,
                                       const float *k_cache,
                                       const float *v_cache,
                                       const int *page_table,
                                       float *output,
                                       int seq_len, int d,
                                       int num_pages) {
  extern __shared__ float smem[];

  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  float scale = 1.0f / sqrtf((float)d);

  // Load query into shared memory
  float *q_shared = smem;
  for (int i = tid; i < d; i += num_threads)
    q_shared[i] = q[i];
  __syncthreads();

  // Phase 1: compute attention scores across all pages
  // We process one page at a time, accumulating softmax online
  float m_prev = -FLT_MAX; // running max
  float l_prev = 0.0f;     // running sum of exp
  float *o_local = smem + d; // partial output accumulator [d] in shared mem
  for (int i = tid; i < d; i += num_threads)
    o_local[i] = 0.0f;
  __syncthreads();

  for (int page_idx = 0; page_idx < num_pages; ++page_idx) {
    int phys_block = page_table[page_idx];
    int page_start = page_idx * BLOCK_SIZE_KV;
    int page_end = page_start + BLOCK_SIZE_KV;
    if (page_end > seq_len) page_end = seq_len;
    int page_len = page_end - page_start;
    if (page_len <= 0) break;

    // Compute scores for this page
    float *page_scores = smem + d + d; // [BLOCK_SIZE_KV]

    for (int j = tid; j < page_len; j += num_threads) {
      float score = 0.0f;
      int k_offset = phys_block * BLOCK_SIZE_KV * d + j * d;
      for (int k = 0; k < d; ++k)
        score += q_shared[k] * k_cache[k_offset + k];
      page_scores[j] = score * scale;
    }
    for (int j = page_len + tid; j < BLOCK_SIZE_KV; j += num_threads)
      page_scores[j] = -FLT_MAX;
    __syncthreads();

    // Find max in this page
    float m_page = -FLT_MAX;
    for (int j = 0; j < page_len; ++j)
      if (page_scores[j] > m_page) m_page = page_scores[j];

    // Online softmax update
    float m_new = fmaxf(m_prev, m_page);
    float rescale_prev = expf(m_prev - m_new);
    float rescale_page = expf(m_page - m_new);

    // Rescale previous accumulator
    for (int i = tid; i < d; i += num_threads)
      o_local[i] *= rescale_prev;

    // Compute exp scores and add V contribution
    float page_sum = 0.0f;
    for (int j = 0; j < page_len; ++j) {
      float p = expf(page_scores[j] - m_new);
      page_sum += p;
      int v_offset = phys_block * BLOCK_SIZE_KV * d + j * d;
      for (int i = tid; i < d; i += num_threads)
        o_local[i] += p * v_cache[v_offset + i];
    }

    l_prev = l_prev * rescale_prev + page_sum;
    m_prev = m_new;

    __syncthreads();
  }

  // Final normalization
  for (int i = tid; i < d; i += num_threads)
    output[i] = o_local[i] / l_prev;
}

// Contiguous attention (reference, like recipe 02)
__global__ void contiguous_attention_decode(const float *q, const float *k,
                                            const float *v, float *output,
                                            int seq_len, int d) {
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  float scale = 1.0f / sqrtf((float)d);

  extern __shared__ float smem[];
  float *scores = smem;

  float max_val = -FLT_MAX;
  for (int j = tid; j < seq_len; j += num_threads) {
    float score = 0.0f;
    for (int kk = 0; kk < d; ++kk)
      score += q[kk] * k[j * d + kk];
    score *= scale;
    scores[j] = score;
  }
  __syncthreads();

  // Find max
  float local_max = -FLT_MAX;
  for (int j = tid; j < seq_len; j += num_threads)
    if (scores[j] > local_max) local_max = scores[j];

  __shared__ float smax[32];
  float tmp = local_max;
  for (int offset = 16; offset > 0; offset >>= 1)
    tmp = fmaxf(tmp, __shfl_down_sync(0xffffffff, tmp, offset));
  if (tid % 32 == 0) smax[tid / 32] = tmp;
  __syncthreads();
  if (tid < 32) {
    tmp = (tid < (num_threads + 31) / 32) ? smax[tid] : -FLT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1)
      tmp = fmaxf(tmp, __shfl_down_sync(0xffffffff, tmp, offset));
    if (tid == 0) smax[0] = tmp;
  }
  __syncthreads();
  max_val = smax[0];

  // Softmax
  float local_sum = 0.0f;
  for (int j = tid; j < seq_len; j += num_threads) {
    scores[j] = expf(scores[j] - max_val);
    local_sum += scores[j];
  }

  __shared__ float ssum[32];
  tmp = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1)
    tmp += __shfl_down_sync(0xffffffff, tmp, offset);
  if (tid % 32 == 0) ssum[tid / 32] = tmp;
  __syncthreads();
  if (tid < 32) {
    tmp = (tid < (num_threads + 31) / 32) ? ssum[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      tmp += __shfl_down_sync(0xffffffff, tmp, offset);
    if (tid == 0) ssum[0] = tmp;
  }
  __syncthreads();
  float sum_exp = ssum[0];

  for (int j = tid; j < seq_len; j += num_threads)
    scores[j] /= sum_exp;
  __syncthreads();

  // Output
  for (int kk = tid; kk < d; kk += num_threads) {
    float val = 0.0f;
    for (int j = 0; j < seq_len; ++j)
      val += scores[j] * v[j * d + kk];
    output[kk] = val;
  }
}

void randomize(float *h, int n) {
  for (int i = 0; i < n; ++i)
    h[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
}

void verify(const float *ref, const float *test, int n, const char *label) {
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    if (diff > max_err) max_err = diff;
  }
  printf("  %-35s max_err=%.6f %s\n", label, max_err,
         max_err < 1e-3f ? "[PASS]" : "[FAIL]");
}

int main() {
  printf("=== Recipe 03: Paged Attention ===\n\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

  const int seq_len = 1024;
  const int d = 64;
  const int total_blocks = 256; // physical block pool
  const int num_pages = (seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;

  printf("Sequence length: %d\n", seq_len);
  printf("Head dimension: d=%d\n", d);
  printf("Page size: %d tokens\n", BLOCK_SIZE_KV);
  printf("Pages needed: %d (out of %d physical blocks)\n", num_pages,
         total_blocks);
  printf("Contiguous cache: %.1f KB\n",
         (float)seq_len * d * 2 * sizeof(float) / 1024);
  printf("Paged cache: %.1f KB (same data, scattered in physical memory)\n\n",
         (float)num_pages * BLOCK_SIZE_KV * d * 2 * sizeof(float) / 1024);

  // Allocate contiguous K,V for reference
  size_t kv_bytes = (size_t)seq_len * d * sizeof(float);
  float *hK = (float *)malloc(kv_bytes);
  float *hV = (float *)malloc(kv_bytes);
  float *hQ = (float *)malloc(d * sizeof(float));
  float *hO_ref = (float *)malloc(d * sizeof(float));
  float *hO_paged = (float *)malloc(d * sizeof(float));

  srand(42);
  randomize(hQ, d);
  randomize(hK, seq_len * d);
  randomize(hV, seq_len * d);

  // Create a SCRAMBLED page table to simulate real paging
  int *hPageTable = (int *)malloc(num_pages * sizeof(int));
  {
    // Shuffle physical block assignments
    int *pool = (int *)malloc(total_blocks * sizeof(int));
    for (int i = 0; i < total_blocks; ++i) pool[i] = i;
    for (int i = total_blocks - 1; i > 0; --i) {
      int j = rand() % (i + 1);
      int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }
    for (int i = 0; i < num_pages; ++i)
      hPageTable[i] = pool[i];
    free(pool);
  }

  printf("Page table (first 8): ");
  for (int i = 0; i < 8 && i < num_pages; ++i) printf("%d ", hPageTable[i]);
  printf("... (scrambled)\n\n");

  // Device allocations
  float *dQ, *dK_contig, *dV_contig, *dO;
  float *dK_paged, *dV_paged;
  int *dPageTable;

  CUDA_CHECK(cudaMalloc(&dQ, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK_contig, kv_bytes));
  CUDA_CHECK(cudaMalloc(&dV_contig, kv_bytes));
  CUDA_CHECK(cudaMalloc(&dO, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK_paged,
      (size_t)total_blocks * BLOCK_SIZE_KV * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV_paged,
      (size_t)total_blocks * BLOCK_SIZE_KV * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dPageTable, num_pages * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(dQ, hQ, d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK_contig, hK, kv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV_contig, hV, kv_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dPageTable, hPageTable, num_pages * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Scatter K,V into paged layout
  CUDA_CHECK(cudaMemset(dK_paged, 0,
      (size_t)total_blocks * BLOCK_SIZE_KV * d * sizeof(float)));
  CUDA_CHECK(cudaMemset(dV_paged, 0,
      (size_t)total_blocks * BLOCK_SIZE_KV * d * sizeof(float)));

  for (int p = 0; p < num_pages; ++p) {
    int phys = hPageTable[p];
    int logical_start = p * BLOCK_SIZE_KV;
    int copy_len = BLOCK_SIZE_KV;
    if (logical_start + copy_len > seq_len) copy_len = seq_len - logical_start;

    CUDA_CHECK(cudaMemcpy(
        dK_paged + (size_t)phys * BLOCK_SIZE_KV * d,
        dK_contig + (size_t)logical_start * d,
        copy_len * d * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        dV_paged + (size_t)phys * BLOCK_SIZE_KV * d,
        dV_contig + (size_t)logical_start * d,
        copy_len * d * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // --- Reference: contiguous attention ---
  int threads = 256;
  size_t smem_contig = seq_len * sizeof(float) + 64 * sizeof(float);
  contiguous_attention_decode<<<1, threads, smem_contig>>>(
      dQ, dK_contig, dV_contig, dO, seq_len, d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hO_ref, dO, d * sizeof(float), cudaMemcpyDeviceToHost));

  // --- Paged attention ---
  size_t smem_paged = (d + d + BLOCK_SIZE_KV) * sizeof(float);
  paged_attention_decode<<<1, threads, smem_paged>>>(
      dQ, dK_paged, dV_paged, dPageTable, dO, seq_len, d, num_pages);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hO_paged, dO, d * sizeof(float), cudaMemcpyDeviceToHost));

  // --- Benchmark ---
  int warmup = 5, iters = 100;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i)
    contiguous_attention_decode<<<1, threads, smem_contig>>>(
        dQ, dK_contig, dV_contig, dO, seq_len, d);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    contiguous_attention_decode<<<1, threads, smem_contig>>>(
        dQ, dK_contig, dV_contig, dO, seq_len, d);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float contig_ms;
  CUDA_CHECK(cudaEventElapsedTime(&contig_ms, start, stop));
  contig_ms /= iters;

  for (int i = 0; i < warmup; ++i)
    paged_attention_decode<<<1, threads, smem_paged>>>(
        dQ, dK_paged, dV_paged, dPageTable, dO, seq_len, d, num_pages);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    paged_attention_decode<<<1, threads, smem_paged>>>(
        dQ, dK_paged, dV_paged, dPageTable, dO, seq_len, d, num_pages);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float paged_ms;
  CUDA_CHECK(cudaEventElapsedTime(&paged_ms, start, stop));
  paged_ms /= iters;

  printf("%-30s %8.3f ms\n", "Contiguous attention:", contig_ms);
  printf("%-30s %8.3f ms  (%.2fx)\n", "Paged attention:", paged_ms,
         contig_ms / paged_ms);
  printf("\n");

  verify(hO_ref, hO_paged, d, "paged vs contiguous");

  printf("\nKey insight: paged attention achieves the SAME result as\n");
  printf("contiguous attention but allows the memory allocator to\n");
  printf("scatter KV blocks anywhere in physical memory, eliminating\n");
  printf("fragmentation and enabling near-100%% memory utilization.\n");

  free(hQ); free(hK); free(hV); free(hO_ref); free(hO_paged);
  free(hPageTable);
  CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK_contig));
  CUDA_CHECK(cudaFree(dV_contig)); CUDA_CHECK(cudaFree(dO));
  CUDA_CHECK(cudaFree(dK_paged)); CUDA_CHECK(cudaFree(dV_paged));
  CUDA_CHECK(cudaFree(dPageTable));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
