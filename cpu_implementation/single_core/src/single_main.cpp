#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // For std::setw, std::fixed, std::setprecision
#include "integral_image.h"
#include <cstdlib> // For std::atoi

int main(int argc, char* argv[])
{
    // Default matrix size
    int H = 8192;
    int W = 8192;

    // Parse command-line arguments for H and W
    if(argc == 3)
    {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);

        if(H <= 0 || W <= 0)
        {
            std::cerr << "Error: Matrix dimensions must be positive integers.\n";
            return -1;
        }
    }
    else if(argc != 1)
    {
        std::cerr << "Usage: " << argv[0] << " [H W]\n";
        std::cerr << "Example: " << argv[0] << " 1024 1024\n";
        return -1;
    }

    // Initialize host input and output vectors
    std::vector<std::vector<float>> img(H, std::vector<float>(W, 1.0f)); // Input matrix filled with 1.0

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Compute the integral image using single-core approach
    std::vector<std::vector<float>> integral = computeIntegralImage(img);

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
    std::cout << " Top-left (0,0): " << integral[0][0] << "\n";
    std::cout << " Top-right (0," << W-1 << "): " << integral[0][W-1] << "\n";
    std::cout << " Bottom-left (" << H-1 << ",0): " << integral[H-1][0] << "\n";
    std::cout << " Bottom-right (" << H-1 << "," << W-1 << "): " 
              << integral[H-1][W-1] << "\n";

    // ------------------- Print Timing -------------------
    std::cout << "\nIntegral image computation took "
              << elapsed.count() << " ms.\n";

    return 0;
}

