#include "integral_image.h"
#include <vector>
#include <iostream>

std::vector<std::vector<float>> computeIntegralImage(const std::vector<std::vector<float>>& img)
{
    int H = static_cast<int>(img.size());
    if (H == 0) {
        return {};
    }
    int W = static_cast<int>(img[0].size());

    // Verify all rows have the same number of columns
    for(const auto& row : img)
    {
        if(row.size() != W)
        {
            std::cerr << "Error: Inconsistent row sizes detected.\n";
            return {};
        }
    }

    // Initialize the integral image with zeros
    std::vector<std::vector<float>> integralImg(H, std::vector<float>(W, 0.0f));

    for(int y = 0; y < H; y++)
    {
        float rowSum = 0.0f;
        for(int x = 0; x < W; x++)
        {
            rowSum += img[y][x];
            if(y == 0)
            {
                integralImg[y][x] = rowSum;
            }
            else
            {
                integralImg[y][x] = rowSum + integralImg[y - 1][x];
            }
        }
    }

    return integralImg;
}
