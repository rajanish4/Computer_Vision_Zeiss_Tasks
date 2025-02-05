#ifndef INTEGRAL_IMAGE_H
#define INTEGRAL_IMAGE_H

#include <vector>

/**
 * @brief Computes the integral image of 'img' (size HxW) on a single CPU core in O(H*W) time.
 * 
 * @param img 2D vector containing the input image (H rows, W columns).
 * @return 2D vector containing the integral image (same dimensions).
 */
std::vector<std::vector<float>> computeIntegralImage(const std::vector<std::vector<float>>& img);

#endif // INTEGRAL_IMAGE_H
