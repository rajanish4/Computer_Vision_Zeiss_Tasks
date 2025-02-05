import subprocess
import csv
import re
from statistics import mean
import sys
import os

def run_benchmark(executable, H, W, runs=20):
    """
    Runs the specified executable with given H and W, captures the elapsed time.

    Args:
        executable (str): Path to the executable.
        H (int): Image height.
        W (int): Image width.
        runs (int): Number of runs.

    Returns:
        list of float: Elapsed times in milliseconds.
    """
    times = []
    for i in range(runs):
        try:
            # Run the executable with H and W as arguments
            result = subprocess.run([executable, str(H), str(W)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=300)  # Timeout after 5 minutes

            if result.returncode != 0:
                print(f"Run {i+1}/{runs} for {os.path.basename(executable)} {H}x{W} failed with error:")
                print(result.stderr)
                continue

            # Extract the elapsed time using regex
            match = re.search(r'Integral image computation took ([\d.]+) ms', result.stdout)
            if match:
                elapsed_time = float(match.group(1))
                times.append(elapsed_time)
                print(f"Run {i+1}/{runs} for {os.path.basename(executable)} {H}x{W}: {elapsed_time} ms")
            else:
                print(f"Run {i+1}/{runs} for {os.path.basename(executable)} {H}x{W}: Elapsed time not found.")

        except subprocess.TimeoutExpired:
            print(f"Run {i+1}/{runs} for {os.path.basename(executable)} {H}x{W}: Timed out.")
        except Exception as e:
            print(f"Run {i+1}/{runs} for {os.path.basename(executable)} {H}x{W}: Encountered an error: {e}")

    return times

def save_to_csv(file_name, data):
    """
    Saves benchmarking data to a CSV file.

    Args:
        file_name (str): Name of the CSV file.
        data (list of dict): Data to save.
    """
    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Image_Height', 'Image_Width', 'Run_Number', 'Elapsed_Time_ms'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def benchmark_all(image_sizes, runs):
    """
    Benchmarks all implementations and saves results to CSV files.
    """
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Define executables and corresponding CSV filenames
    implementations = {
        'integral_image_singlecore': 'benchmark_results_single_core.csv',
        'integral_image_multicore': 'benchmark_results_multi_core.csv',
        'integral_image_cuda': 'benchmark_results_cuda.csv'
    }

    # Iterate over each implementation
    for exec_name, csv_file in implementations.items():
        executable = os.path.join('.', 'bin', exec_name)
        if sys.platform.startswith('win'):
            executable += '.exe'  # Adjust for Windows

        if not os.path.isfile(executable):
            print(f"Executable {executable} not found. Skipping.")
            continue

        all_data = []
        print(f"\nBenchmarking {exec_name}...")
        for (H, W) in image_sizes:
            print(f"\nImage Size: {H}x{W}")
            times = run_benchmark(executable, H, W, runs=runs)
            for run_num, elapsed in enumerate(times, start=1):
                all_data.append({
                    'Image_Height': H,
                    'Image_Width': W,
                    'Run_Number': run_num,
                    'Elapsed_Time_ms': elapsed
                })
            if times:
                avg_time = mean(times)
                print(f"Average elapsed time for {H}x{W}: {avg_time:.3f} ms")
            else:
                print(f"No successful runs for {H}x{W}.")

        # Save the results to CSV
        if all_data:
            save_to_csv(os.path.join(results_folder, csv_file), all_data)
            print(f"\nResults for {exec_name} saved to {os.path.join(results_folder, csv_file)}.")
        else:
            print(f"No data to save for {exec_name}.")

if __name__ == "__main__":
    # Define image sizes to test
    image_sizes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192)
    ]
    runs=20 # how many times to run each algorithm for each image size

    benchmark_all(image_sizes=image_sizes, runs=runs)
