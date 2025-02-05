#include "prefix_sum.h"
#include <vector>
#include <thread>

// Function to compute prefix sum for a range of rows
void computeRowPrefixSum(std::vector<std::vector<float>>& arr, int startRow, int endRow) {
    int W = arr[0].size();
    for (int y = startRow; y < endRow; y++) {
        float runningSum = 0.0f;
        for (int x = 0; x < W; x++) {
            runningSum += arr[y][x];
            arr[y][x] = runningSum;
        }
    }
}

// Function to compute prefix sum for a range of columns
void computeColPrefixSum(std::vector<std::vector<float>>& arr, int startCol, int endCol) {
    int H = arr.size();
    for (int x = startCol; x < endCol; x++) {
        float runningSum = 0.0f;
        for (int y = 0; y < H; y++) {
            runningSum += arr[y][x];
            arr[y][x] = runningSum;
        }
    }
}

// Generic function to parallelize work across threads
void parallelProcess(void (*worker)(std::vector<std::vector<float>>&, int, int),
                     std::vector<std::vector<float>>& arr,
                     int numThreads,
                     int totalElements) {
    
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    int elementsPerThread = totalElements / numThreads;
    int remainder = totalElements % numThreads;
    int currentElement = 0;

    for (int t = 0; t < numThreads; t++) {
        int blockSize = elementsPerThread + (t < remainder ? 1 : 0);
        if (blockSize == 0) break;

        int start = currentElement;
        int end = currentElement + blockSize;
        currentElement = end;

        threads.emplace_back(worker, std::ref(arr), start, end);
    }

    for (auto& th : threads) {
        th.join();
    }
}

// Compute the integral image using two-pass parallel prefix sum
std::vector<std::vector<float>> computeIntegralImageTwoPass(
    const std::vector<std::vector<float>>& img, int numThreads) {
    
    int H = img.size();
    if (H == 0) return {};
    int W = img[0].size();

    // Copy input image
    std::vector<std::vector<float>> arr = img;

    // Compute row-wise prefix sum in parallel
    parallelProcess(computeRowPrefixSum, arr, numThreads, H);

    // Compute column-wise prefix sum in parallel
    parallelProcess(computeColPrefixSum, arr, numThreads, W);

    return arr;
}
