#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // For std::setw, std::fixed, std::setprecision
#include "prefix_sum.h"
#include <cstdlib> // For std::atoi

int main(int argc, char* argv[])
{
    // Default matrix size
    int H = 8192;
    int W = 8192;
    
    // Default number of threads
    int numThreads = std::thread::hardware_concurrency(); // = 12
    // std::cout << "Number of threads: " << numThreads << "\n";
    if(numThreads == 0) numThreads = 4; // Fallback to 4 if detection fails

    // Parse command-line arguments for H, W, and optional numThreads
    if(argc == 3 || argc == 4)
    {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);

        if(argc == 4)
        {
            numThreads = std::atoi(argv[3]);
            if(numThreads <= 0)
            {
                std::cerr << "Error: Number of threads must be a positive integer.\n";
                return -1;
            }
        }

        if(H <= 0 || W <= 0)
        {
            std::cerr << "Error: Matrix dimensions must be positive integers.\n";
            return -1;
        }
    }
    else if(argc != 1)
    {
        std::cerr << "Usage: " << argv[0] << " [H W [numThreads]]\n";
        std::cerr << "Example: " << argv[0] << " 1024 1024 8\n";
        return -1;
    }

    // Initialize host input and output vectors
    std::vector<std::vector<float>> img(H, std::vector<float>(W, 1.0f)); // Input matrix filled with 1.0

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Compute the integral image using 2-pass multi-core approach
    auto integral = computeIntegralImageTwoPass(img, numThreads);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // ------------------- Print the First 4x4 Elements -------------------
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "First 4x4 elements of the Integral Image:\n";
    for(int y = 0; y < 4 && y < H; y++) {          // Rows
        for(int x = 0; x < 4 && x < W; x++) {      // Columns
            std::cout << std::setw(6) << integral[y][x] << " ";
        }
        std::cout << "\n";
    }

    // ------------------- Print Specific Elements with Coordinates -------------------
    std::cout << "\nIntegral Image corners with coordinates:\n";
    std::cout << " Top-left (0,0): "       << integral[0][0]            << "\n";
    std::cout << " Top-right (0," << W-1 << "): " << integral[0][W-1]            << "\n";
    std::cout << " Bottom-left (" << H-1 << ",0): " << integral[H-1][0]        << "\n";
    std::cout << " Bottom-right (" << H-1 << "," << W-1 << "): " 
              << integral[H-1][W-1] << "\n";

    // ------------------- Print Timing -------------------
    std::cout << "\nIntegral image computation took "
              << elapsed.count() << " ms.\n";

    return 0;
}
