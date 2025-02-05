#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include "constants.h" 

/**
 * @brief Transposes a matrix using shared memory to optimize memory accesses.
 *
 * @param input Pointer to the input matrix on the device.
 * @param output Pointer to the transposed matrix on the device.
 * @param width Number of columns in the input matrix.
 * @param height Number of rows in the input matrix.
 */
__global__ void transpose(float* input, float* output, int width, int height);

#endif // TRANSPOSE_KERNELS_H
