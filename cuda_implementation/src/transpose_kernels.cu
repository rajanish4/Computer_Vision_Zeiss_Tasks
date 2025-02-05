#include "transpose_kernels.h"

/**
 * @brief Transposes a matrix using shared memory to optimize memory accesses.
 *
 * This kernel transposes a matrix in tiles of size TILE_DIM x TILE_DIM, using shared memory to coalesce
 * memory accesses and reduce global memory latency.
 */
__global__ void transpose(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads();

    // Transpose the tile and write to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
}
