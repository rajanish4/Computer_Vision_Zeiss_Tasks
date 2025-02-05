#ifndef PREFIX_SUM_H
#define PREFIX_SUM_H

#include <vector>
#include <thread>

/**
 * @brief Computes prefix sum for a subset of rows.
 * 
 * @param arr      2D array (HxW) modified in-place.
 * @param startRow Start index for row range.
 * @param endRow   End index for row range.
 */
void computeRowPrefixSum(std::vector<std::vector<float>>& arr, int startRow, int endRow);

/**
 * @brief Computes prefix sum for a subset of columns.
 * 
 * @param arr      2D array (HxW) modified in-place.
 * @param startCol Start index for column range.
 * @param endCol   End index for column range.
 */
void computeColPrefixSum(std::vector<std::vector<float>>& arr, int startCol, int endCol);

/**
 * @brief Generic function to parallelize computation using multiple threads.
 * 
 * @param worker        Function pointer to the worker function.
 * @param arr           2D array (HxW) modified in-place.
 * @param numThreads    Number of threads to utilize.
 * @param totalElements Number of elements to divide across threads.
 */
void parallelProcess(void (*worker)(std::vector<std::vector<float>>&, int, int),
                     std::vector<std::vector<float>>& arr,
                     int numThreads,
                     int totalElements);

/**
 * @brief Computes the integral image using a two-pass (row-wise and column-wise) parallel approach.
 * 
 * @param img        Input 2D image (HxW) as a constant reference.
 * @param numThreads Number of threads to utilize for parallelization.
 * @return std::vector<std::vector<float>> The computed integral image.
 */
std::vector<std::vector<float>> computeIntegralImageTwoPass(
    const std::vector<std::vector<float>>& img,
    int numThreads);

#endif // PREFIX_SUM_H
