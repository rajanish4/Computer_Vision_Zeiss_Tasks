# Zeiss Tasks

## Task 1: Schematic Sketch

Consider an image `I` of size `H x W`. We compute the integral image `IΣ` in a single pass:

1. **Row-Wise Computation:**
   - Initialize `rowSum` to `0` for each row `y` (0 to H-1).
   - For each column `x` (0 to W-1):
     - `rowSum += I[y][x]`  // accumulate in the row
     - If `y == 0`: `IΣ[y][x] = rowSum` (no row above on first row)
     - Else: `IΣ[y][x] = rowSum + IΣ[y-1][x]`

### Computational Work (Time Complexity)

- **Outer loop**: over `y` in `[0..H-1]`
- **Inner loop**: over `x` in `[0..W-1]`
- Each iteration performs a constant number of operations.
- **Total Complexity:** `O(H × W)`

Thus, for an image of size `H x W`, the single-core computation of the 2D integral image `IΣ(y,x)` is **linear** in the number of pixels.

---

## Task 2: Two-Pass Integral Image (Multi-Core)

### 1. Row-Wise Pass
- Split the image into **blocks of rows**.
- Each block is assigned to **one thread**.
- Each thread computes a **1D prefix sum** row-wise.

```
+---------------------------+
|  Block0: rows [0..r0)    |
|    (Thread 0)            |
|--------------------------|
|  Block1: rows [r0..r1)   |
|    (Thread 1)            |
|--------------------------|
|  Block2: rows [r1..r2)   |
|    (Thread 2)            |
|     ... etc.             |
+---------------------------+
```

### 2. Column-Wise Pass
- Split the image into **blocks of columns**.
- Each thread handles **one block of columns**.
- Compute a **1D prefix sum top-to-bottom** in each column.

```
+---------+---------+----------+
| Block0  | Block1  | Block2   |
| cols    | cols    | cols     |
+---------+---------+----------+
(Thread 0) (Thread1) (Thread2)
```

### 3. Summary
- **Single-Core Complexity:** `O(H × W)`
- **Multi-Core, Two-Pass Complexity:** `O(H × W)`, but wall-clock time is reduced.
- **Parallel Speedup:** `T` threads ideally divide the workload into `1/T` of the time.
- **Real-world factors** affect efficiency (memory bandwidth, load balancing, etc.).

---

## Task 3: Expected Speedup for Multi-Core Implementation

### 1. **Ideal Theoretical Speedup**
- With `p` cores, the best case speedup is `p`.
- **Ideal Time:** `T_serial / p`.

### 2. **Real-World Limitations**
- **Amdahl’s Law**: Speedup is limited by the serial fraction.
- **Memory Bandwidth**: Multiple cores accessing large data sets can bottleneck.
- **Synchronization Overhead**: Threads require coordination.
- **Load Imbalance**: Uneven work distribution can slow down execution.
- **Cache Effects**: More cores may lead to increased cache misses.

### 3. **Conclusion**
- **Best Case:** Speedup ~ `p` (ideal case).
- **Typical Case:** Speedup `< p` due to real-world inefficiencies.

---

## Task 4: Parallelizing Integral Image Computation on GPU using CUDA

### Steps:

```
Input Image
│
├── Row-Wise Exclusive Scan (Batched Exclusive Scan)
│   ├── Chunked rows processed in parallel
│   └── Block sums aggregated and adjusted
│
├── Convert Exclusive to Inclusive Scan (batched_add_input_to_scan)
│   └── Add input values to exclusive scan results
│
├── Transpose Matrix
│   └── Columns become rows
│
├── Column-Wise Exclusive Scan (Same as row-wise on transposed data)
│   ├── Chunked "rows" (original columns) processed
│   └── Block sums adjusted
│
├── Convert Exclusive to Inclusive Scan (batched_add_input_to_scan)
│   └── Add transposed input values to column-wise exclusive scan results
│
└── Transpose Back
    └── Final integral image
```

### Key CUDA Methods:
- **Batched Processing:** Rows/columns treated independently for massive parallelism.
- **Shared Memory:** Reduces global memory latency.
- **Blelloch Scan:** `O(log n)` optimal GPU parallelism.
- **Block Sums Handling:** Aggregates and scans partial sums.
- **Efficient Transpose:** Uses shared memory tiles to minimize bank conflicts.

### Advantages of GPU Implementation:
- **Massive Parallelism:** Thousands of threads process in parallel.
- **Memory Optimization:** Efficient shared memory usage.
- **Scalability:** Handles large images (e.g., `8192x8192`) efficiently.

---

## Task 5: Theoretical GPU vs CPU Speedup

### Breakdown of Each Kernel's Complexity

1. **Row-wise Exclusive Scan:** `O(log W)`
2. **Block Sums Scan:** `O(log (W / BLOCK_SIZE)) ≈ O(log W)`
3. **Add Block Sums:** `O(1)`
4. **Convert Exclusive Scan to Inclusive:** `O(1)`
5. **Transpose:** `O(1)`
6. **Column-wise Exclusive Scan:** `O(log H)`
7. **Block Sums Scan (Columns):** `O(log H)`
8. **Add Block Sums (Columns):** `O(1)`
9. **Convert Exclusive Scan to Inclusive (Columns):** `O(1)`
10. **Final Transpose:** `O(1)`

### Final GPU Complexity Calculation

total GPU complexity:

`O(log W + log H)`

For a **square image (W = H = N)**, this simplifies further to:

`O(log N)`

### CPU Complexity

total CPU complexity: `O(N²)`

### Complexity-Based Speedup Comparison

| Approach  | Complexity |
|-----------|------------|
| **CPU (Single-Core)**  | O(N²) |
| **GPU (CUDA)**  | O(log N) |

Thus, the **theoretical speedup** from CPU to GPU is:

`O(N² / log N)`

### Practical Limitations Affecting GPU Performance:

1.**Memory Bandwidth Limitations**
   - GPU global memory access is slower than shared memory.
   - Excessive memory transfers between host (CPU) and device (GPU) can reduce gains.

2.**Kernel Launch Overheads**
   - Each CUDA kernel has a launch overhead.
   - Frequent kernel calls (e.g., for row and column scans) increase total execution time.

3.**Limited Streaming Multiprocessors (SMs)**
   - The GPU executes a limited number of thread blocks concurrently.
   - Large images may cause some blocks to wait, leading to serialization effects.

4.**Shared Memory Bank Conflicts**
   - Transpose operations rely on shared memory tiling.
   - Bank conflicts can cause delays when multiple threads access the same memory bank.

---

## Task 6: Extending Integral Image Computation to 3D Volumes

The integral image computation can be extended to **three-dimensional (3D) volumes** by applying the same **prefix sum strategy** across an additional depth dimension. This follows a **three-pass approach**:

**Pass 1: Row-Wise Exclusive Scan (X-Direction)**
   - Each thread computes a prefix sum along the **X-axis** independently for each **(y, z) plane**.

**Pass 2: Column-Wise Exclusive Scan (Y-Direction)**
   - The volume is **transposed** so that the **Y-axis** becomes rows.

**Pass 3: Depth-Wise Exclusive Scan (Z-Direction)**
   - The volume is **transposed** again to process the **Z-axis as rows**.

**Final Transpose Back**
   - The computed integral volume is restored to its **original layout**.

### Challenges in 3D Integral Image Computation
1.**Integer Overflow Risk**:
   - The sum of pixel values across large volumes can exceed the maximum representable value of **32-bit integers**, leading to incorrect computations.

2.**Floating-Point Precision Errors**:
   - Using `float` for integral image storage can result in **rounding errors**, causing discrepancies in summed values, especially in deep volumetric datasets.

3.**Memory Bandwidth Bottleneck**:
   - **Global memory accesses** for large volumes can become a limiting factor, as multiple passes require heavy read/write operations.
  
---

## Usage

### Requirements:
1. Install `g++`, `cuda`, and `python` (optional for benchmarking).
2. Optional for plotting benchmark:
   - Install `matplotlib`, `seaborn`, and `pandas`.

### Compilation:
Run Makefile:
```
make clean
make
```

### Benchmarking & Plotting:
```
python benchmark_all.py  # (default: 20 runs for each image size)
python plot_benchmark.py
```
Results will be stored inside the `results` folder.

---


## Results:

### System: 
   - CPU: Intel Core i7-9750H (16 GB)
   - GPU: NVIDIA GeForce GTX 1650 (4 GB)

### Average performance for 8192x8192 image size:
   - Single-core CPU = 931.9 ms
   - Multi-core CPU = 577.6 ms (12 threads)
   - GPU = 108.5 ms


