import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_benchmark_data(file_path):
    """
    Reads a benchmark CSV file and returns a DataFrame with average and std elapsed time per image size.
    """
    try:
        df = pd.read_csv(file_path)
        # Compute average and std elapsed time for each image size
        agg_df = df.groupby(['Image_Height', 'Image_Width'])['Elapsed_Time_ms'].agg(['mean', 'std']).reset_index()
        # Create a new column for Image Size as a string (e.g., "256x256")
        agg_df['Image_Size'] = agg_df['Image_Height'].astype(str) + 'x' + agg_df['Image_Width'].astype(str)
        return agg_df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def plot_benchmark(cpu_single, cpu_multi, cuda, output_path):
    """
    Plots the benchmark results for single-core, multi-core, and CUDA implementations with error bars.
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))

    # Plot CPU Single-Core
    if not cpu_single.empty:
        plt.errorbar(cpu_single['Image_Size'], cpu_single['mean'], yerr=cpu_single['std'],
                     marker='o', label='CPU Single-Core', capsize=5)
    
    # Plot CPU Multi-Core
    if not cpu_multi.empty:
        plt.errorbar(cpu_multi['Image_Size'], cpu_multi['mean'], yerr=cpu_multi['std'],
                     marker='s', label='CPU Multi-Core', capsize=5)
    
    # Plot CUDA
    if not cuda.empty:
        plt.errorbar(cuda['Image_Size'], cuda['mean'], yerr=cuda['std'],
                     marker='^', label='CUDA', capsize=5)
    
    plt.title('Integral Image Computation Benchmark Comparison')
    plt.xlabel('Image Size (HxW)')
    plt.ylabel('Average Elapsed Time (ms)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Benchmark comparison plot saved as '{output_path}'.")


def main():
    # Define paths to benchmark CSV files
    results_folder = 'results'
    out_path = os.path.join(results_folder, 'benchmark_comparison.png')
    cpu_single_core_path = os.path.join(results_folder, 'benchmark_results_single_core.csv')
    cpu_multi_core_path = os.path.join(results_folder, 'benchmark_results_multi_core.csv')
    cuda_path = os.path.join(results_folder, 'benchmark_results_cuda.csv')

    # Read benchmark data
    print("Reading benchmark data...")
    cpu_single = read_benchmark_data(cpu_single_core_path)
    cpu_multi = read_benchmark_data(cpu_multi_core_path)
    cuda = read_benchmark_data(cuda_path)

    # Check if any data was loaded
    if cpu_single.empty and cpu_multi.empty and cuda.empty:
        print("No benchmark data found. Please ensure the CSV files exist in the specified directories.")
        return

    # Plot the benchmark comparison
    plot_benchmark(cpu_single, cpu_multi, cuda, out_path)

if __name__ == "__main__":
    main()
