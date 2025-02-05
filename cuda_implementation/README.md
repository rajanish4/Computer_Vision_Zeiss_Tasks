# CUDA Integral Image Computation

## Overview

This project implements the computation of an **Integral Image** using CUDA.

## Components

### 1. **Kernels:**

- **Exclusive Scan (`scan_kernels.cu`):**
  - Implements a batched exclusive scan using the Blelloch algorithm.
  - Processes multiple arrays (e.g., rows or columns) in parallel.

- **Block Sums Scan (`scan_kernels.cu`):**
  - Scans the sums of each block obtained from the exclusive scans.
  - Facilitates the propagation of sums across different chunks within arrays.

- **Add Block Sums (`scan_kernels.cu`):**
  - Adds the scanned block sums to the exclusive scan outputs to maintain cumulative sums.

- **Add Input to Scan (`scan_kernels.cu`):**
  - Converts exclusive scans to inclusive scans by adding original input elements.

- **Transpose (`transpose_kernels.cu`):**
  - Transposes a matrix using shared memory to optimize memory accesses.
  - Facilitates column-wise operations by treating columns as rows after transposition.

### 2. **Utility (`utility.cu`):**

- Provides error checking mechanisms for CUDA API calls.
- Ensures that CUDA operations are executed successfully, aiding in debugging and reliability.

### 3. **Main (`main.cu`):**

- Orchestrates the overall computation of the integral image.
- Handles memory allocations, kernel launches, and result verification.
- Demonstrates the functionality with a sample matrix and outputs the integral image.

## Compilation

Ensure that CUDA is properly installed on the system. The Makefile assumes that CUDA is installed in the default directory. If CUDA is installed elsewhere, update the `CUDA_PATH` in the Makefile accordingly.

### **Steps to Compile:**

1. **Compile the Project:**
  ```cmd
  make

2. **Run the Executable:**
  ```cmd
  integral_image.exe

3. **Clean the Build:**
  ```cmd
  make clean