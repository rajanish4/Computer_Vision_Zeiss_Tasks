#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <cstdlib> // For std::atoi
#include "scan_kernels.h"
#include "transpose_kernels.h"
#include "utility.h"

#include "constants.h" // For BLOCK_SIZE, TILE_DIM

/**
 * @brief Computes the integral image using batched exclusive scans and transposes.
 *
 * The computation involves:
 * 1. Row-wise exclusive scan.
 * 2. Scanning and adding block sums to rows.
 * 3. Converting exclusive scans to inclusive scans for rows.
 * 4. Transposing the matrix.
 * 5. Column-wise exclusive scan.
 * 6. Scanning and adding block sums to columns.
 * 7. Converting exclusive scans to inclusive scans for columns.
 * 8. Transposing back to obtain the final integral image.
 *
 * @param d_input Pointer to the input matrix on the device.
 * @param d_output Pointer to store the integral image on the device.
 * @param width Number of columns in the input matrix.
 * @param height Number of rows in the input matrix.
 */
void compute_integral_image(float* d_input, float* d_output, int width, int height) {
    float *d_row_scan, *d_transposed, *d_block_sums, *d_scanned_sums;
    int row_chunks = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int col_chunks = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t block_sums_size_rows = row_chunks * height * sizeof(float);
    size_t block_sums_size_cols = col_chunks * width * sizeof(float);
    size_t total_block_sums_size = (block_sums_size_rows > block_sums_size_cols) ? block_sums_size_rows : block_sums_size_cols;

    // Allocate memory for intermediate computations
    CUDA_CHECK(cudaMalloc(&d_row_scan, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transposed, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, total_block_sums_size));
    CUDA_CHECK(cudaMalloc(&d_scanned_sums, total_block_sums_size));

    // ------------------- Row-wise Exclusive Scan -------------------
    dim3 row_blocks(row_chunks, height);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    batched_exclusive_scan<<<row_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_input, d_row_scan, d_block_sums, width, width, height
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Scan Block Sums for Rows -------------------
    dim3 scan_blocks_rows(height);
    batched_block_sums_scan<<<scan_blocks_rows, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_block_sums, d_scanned_sums, row_chunks, height
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Add Scanned Block Sums to Rows -------------------
    dim3 add_blocks_rows(row_chunks, height);
    batched_add_block_sums<<<add_blocks_rows, BLOCK_SIZE>>>(
        d_row_scan, d_scanned_sums, width, width, height, row_chunks
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Convert Exclusive Scan to Inclusive Scan for Rows -------------------
    // Launch kernel with enough blocks to cover all elements
    int total_elements = width * height;
    int threads = 512;
    int blocks = (total_elements + threads - 1) / threads;
    batched_add_input_to_scan<<<blocks, threads>>>(
        d_input, d_row_scan, d_row_scan, total_elements, width, height
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Transpose -------------------
    dim3 grid_transpose((width + TILE_DIM - 1)/TILE_DIM, (height + TILE_DIM - 1)/TILE_DIM);
    dim3 t_block(TILE_DIM, TILE_DIM);
    transpose<<<grid_transpose, t_block>>>(d_row_scan, d_transposed, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Column-wise Exclusive Scan -------------------
    dim3 col_blocks(col_chunks, width);
    batched_exclusive_scan<<<col_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_transposed, d_row_scan, d_block_sums, height, height, width
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Scan Block Sums for Columns -------------------
    dim3 scan_blocks_cols(width);
    batched_block_sums_scan<<<scan_blocks_cols, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_block_sums, d_scanned_sums, col_chunks, width
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Add Scanned Block Sums to Columns -------------------
    dim3 add_blocks_cols(col_chunks, width);
    batched_add_block_sums<<<add_blocks_cols, BLOCK_SIZE>>>(
        d_row_scan, d_scanned_sums, height, height, width, col_chunks
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Convert Exclusive Scan to Inclusive Scan for Columns -------------------
    // Launch kernel with enough blocks to cover all elements
    blocks = (total_elements + threads - 1) / threads;
    batched_add_input_to_scan<<<blocks, threads>>>(
        d_transposed, d_row_scan, d_row_scan, total_elements, height, width
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------- Transpose Back -------------------
    transpose<<<grid_transpose, t_block>>>(d_row_scan, d_output, height, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free temporary memory
    CUDA_CHECK(cudaFree(d_row_scan));
    CUDA_CHECK(cudaFree(d_transposed));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_scanned_sums));
}

/**
 * @brief Converts the flat 1D array to a 2D vector for easier manipulation and printing.
 *
 * @param flat_array Flat 1D array representing a 2D matrix.
 * @param width Number of columns.
 * @param height Number of rows.
 * @return std::vector<std::vector<float>> 2D vector representation of the array.
 */
std::vector<std::vector<float>> convert_to_2d(const std::vector<float>& flat_array, int width, int height) {
    std::vector<std::vector<float>> two_d(height, std::vector<float>(width, 0.0f));
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            two_d[y][x] = flat_array[y * width + x];
        }
    }
    return two_d;
}

int main(int argc, char* argv[]) {
    // Image size
    int W = 8192, H = 8192;
    
    // Parse command-line arguments for H, W
    if(argc == 3) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);

        if(H <= 0 || W <= 0) {
            std::cerr << "Error: Matrix dimensions must be positive integers.\n";
            return -1;
        }
    }
    else if(argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [H W]\n";
        std::cerr << "Example: " << argv[0] << " 1024 1024\n";
        return -1;
    }

    // Initialize host input and output vectors
    std::vector<float> h_input(W * H, 1.0f); // Initialize input with 1.0
    std::vector<float> h_output(W * H, 0.0f); // Initialize output

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    // Compute the integral image
    compute_integral_image(d_input, d_output, W, H);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Integral image computation took %f ms\n", milliseconds);

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, W * H * sizeof(float), cudaMemcpyDeviceToHost));

    // Convert flat array to 2D for easier printing
    std::vector<std::vector<float>> integral = convert_to_2d(h_output, W, H);

    // Print the first 4x4 integral image for verification
    printf("First 4x4 elements of the Integral Image:\n");
    for(int y = 0; y < 4 && y < H; y++) {          // Rows
        for(int x = 0; x < 4 && x < W; x++) {      // Columns
            printf("%6.1f ", integral[y][x]);
        }
        printf("\n");
    }

    // Print specific elements for verification
    printf("\nIntegral Image corners with coordinates:\n");
    printf(" Top-left (0,0): %4.1f\n", integral[0][0]);                           
    printf(" Top-right (0,%d): %4.1f\n", W - 1, integral[0][W - 1]);                      
    printf(" Bottom-left (%d,0): %4.1f\n", H - 1, integral[H - 1][0]);              
    printf(" Bottom-right (%d,%d): %4.1f\n", H - 1, W - 1, integral[H - 1][W - 1]);      

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
