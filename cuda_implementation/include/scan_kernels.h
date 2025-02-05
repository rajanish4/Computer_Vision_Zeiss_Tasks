#ifndef SCAN_KERNELS_H
#define SCAN_KERNELS_H

#include "constants.h" 

/**
 * @brief Performs a batched exclusive scan on multiple arrays using the Blelloch scan algorithm.
 *
 * @param input Pointer to the input array on the device.
 * @param output Pointer to the output array on the device.
 * @param block_sums Pointer to store the sum of each block (optional, can be nullptr).
 * @param n Number of elements per array.
 * @param stride Number of elements between consecutive arrays.
 * @param num_arrays Number of arrays to scan.
 */
__global__ void batched_exclusive_scan(float* input, float* output, float* block_sums, int n, int stride, int num_arrays);

/**
 * @brief Performs a batched exclusive scan on block sums to propagate the sums across chunks.
 *
 * @param block_sums Pointer to the input block sums on the device.
 * @param scanned_sums Pointer to the output scanned sums on the device.
 * @param chunks_per_array Number of chunks per array.
 * @param num_arrays Number of arrays to scan.
 */
__global__ void batched_block_sums_scan(float* block_sums, float* scanned_sums, int chunks_per_array, int num_arrays);

/**
 * @brief Adds the scanned block sums to the exclusive scan output to maintain correct cumulative sums.
 *
 * @param output Pointer to the exclusive scan output array on the device.
 * @param scanned_sums Pointer to the scanned block sums on the device.
 * @param n Number of elements per array.
 * @param stride Number of elements between consecutive arrays.
 * @param num_arrays Number of arrays to scan.
 * @param chunks_per_array Number of chunks per array.
 */
__global__ void batched_add_block_sums(float* output, float* scanned_sums, int n, int stride, int num_arrays, int chunks_per_array);

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
__global__ void batched_add_input_to_scan(float* input, float* scan_output, float* inclusive_output, int n, int stride, int num_arrays);

#endif // SCAN_KERNELS_H
