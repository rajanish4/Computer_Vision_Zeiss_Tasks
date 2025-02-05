#ifndef UTILITY_H
#define UTILITY_H

#include <cuda_runtime.h>
#include <iostream> // For std::cerr
#include <cstdlib>  // For exit

/**
 * @brief Checks and handles CUDA errors.
 *
 * @param call The CUDA runtime API call to check.
 * @param file The file where the error occurred.
 * @param line The line number where the error occurred.
 */
inline void checkCudaError(cudaError_t call, const char* file, int line) {
    if (call != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(call) 
                  << " (error code " << call << ") at " 
                  << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

#endif // UTILITY_H
