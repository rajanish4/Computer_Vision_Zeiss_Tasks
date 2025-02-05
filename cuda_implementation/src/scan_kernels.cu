#include "scan_kernels.h"

/**
 * @brief Implements a batched exclusive scan using the Blelloch scan algorithm.
 *
 * Each array is processed in chunks of size BLOCK_SIZE. The scan is performed within each block,
 * and block sums are stored for further processing.
 */
__global__ void batched_exclusive_scan(float* input, float* output, float* block_sums, int n, int stride, int num_arrays) {
    extern __shared__ float temp[]; // Shared memory for scan
    int tid = threadIdx.x;
    int array_idx = blockIdx.y;    // Array index (row or column)
    int chunk = blockIdx.x;        // Chunk index within the array

    if (array_idx >= num_arrays) return;

    int offset = chunk * BLOCK_SIZE;
    int array_start = array_idx * stride;

    // Load input into shared memory
    if (offset + tid < n) {
        temp[tid] = input[array_start + offset + tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    int log2n = 0;
    while ((1 << log2n) < BLOCK_SIZE) log2n++;
    for(int d = 0; d < log2n; d++) {
        int step = 1 << (d + 1);
        if(tid < (BLOCK_SIZE / step)){
            int index = (tid + 1) * step - 1;
            if(index < BLOCK_SIZE){
                temp[index] += temp[index - (step >> 1)];
            }
        }
        __syncthreads();
    }

    // Save the total sum and set last element to zero for exclusive scan
    if (tid == 0) {
        if (block_sums)
            block_sums[array_idx * gridDim.x + chunk] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0.0f;
    }
    __syncthreads();

    // Down-sweep phase
    for(int d = log2n - 1; d >= 0; d--){
        int step = 1 << (d + 1);
        if(tid < (BLOCK_SIZE / step)){
            int index = (tid + 1) * step - 1;
            if(index < BLOCK_SIZE){
                float t = temp[index - (step >> 1)];
                temp[index - (step >> 1)] = temp[index];
                temp[index] += t;
            }
        }
        __syncthreads();
    }

    // Write results to output
    if (offset + tid < n) {
        output[array_start + offset + tid] = temp[tid];
    }
}


/**
 * @brief Implements a batched exclusive scan on block sums using the Blelloch scan algorithm.
 *
 * This kernel scans the block sums obtained from the exclusive scan of each chunk,
 * allowing the propagation of sums across different chunks within each array.
 */
__global__ void batched_block_sums_scan(float* block_sums, float* scanned_sums, int chunks_per_array, int num_arrays) {
    extern __shared__ float temp[]; // Shared memory for scan
    int tid = threadIdx.x;
    int array_idx = blockIdx.x;

    if (array_idx >= num_arrays) return;

    // Load block sums into shared memory
    if (tid < chunks_per_array) {
        temp[tid] = block_sums[array_idx * chunks_per_array + tid];
    }
    else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    int log2n = 0;
    while ((1 << log2n) < chunks_per_array) log2n++;
    for(int d = 0; d < log2n; d++) {
        int step = 1 << (d + 1);
        if(tid < (chunks_per_array / step)){
            int index = (tid + 1) * step - 1;
            if(index < chunks_per_array){
                temp[index] += temp[index - (step >> 1)];
            }
        }
        __syncthreads();
    }

    // Save the total sum and set last element to zero for exclusive scan
    if (tid == 0) {
        if (scanned_sums)
            temp[chunks_per_array - 1] = 0.0f;
    }
    __syncthreads();

    // Down-sweep phase
    for(int d = log2n - 1; d >= 0; d--){
        int step = 1 << (d + 1);
        if(tid < (chunks_per_array / step)){
            int index = (tid + 1) * step - 1;
            if(index < chunks_per_array){
                float t = temp[index - (step >> 1)];
                temp[index - (step >> 1)] = temp[index];
                temp[index] += t;
            }
        }
        __syncthreads();
    }

    // Write scanned sums back
    if(tid < chunks_per_array){
        scanned_sums[array_idx * chunks_per_array + tid] = temp[tid];
    }
}

/**
 * @brief Adds the scanned block sums to the exclusive scan output to maintain correct cumulative sums.
 *
 * This ensures that each chunk's scan output correctly reflects the sum of all previous chunks.
 */
__global__ void batched_add_block_sums(float* output, float* scanned_sums, int n, int stride, int num_arrays, int chunks_per_array) {
    int array_idx = blockIdx.y;
    int chunk = blockIdx.x;
    int tid = threadIdx.x;

    if (array_idx >= num_arrays || chunk >= chunks_per_array) return;

    int offset = chunk * BLOCK_SIZE;
    int array_start = array_idx * stride;
    float sum = scanned_sums[array_idx * chunks_per_array + chunk];

    if (offset + tid < n) {
        output[array_start + offset + tid] += sum;
    }
}

/**
 * @brief Adds the original input elements to the exclusive scan output to convert it to an inclusive scan.
 *
 * @param input Pointer to the original input array on the device.
 * @param scan_output Pointer to the exclusive scan output array on the device.
 * @param inclusive_output Pointer to store the inclusive scan output on the device.
 * @param n Total number of elements across all arrays.
 * @param stride Number of elements per array.
 * @param num_arrays Number of arrays.
 */
__global__ void batched_add_input_to_scan(float* input, float* scan_output, float* inclusive_output, int n, int stride, int num_arrays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    // Determine which array and element this thread is processing
    int array_idx = idx / stride;
    //int element_idx = idx % stride;

    if(array_idx >= num_arrays) return;

    inclusive_output[idx] = scan_output[idx] + input[idx];
}